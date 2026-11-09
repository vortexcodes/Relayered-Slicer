"""Onshape REST client: pull a whole assembly (parts, placements, mates, mass) into an Assembly.

Auth: an Onshape API key pair (https://dev-portal.onshape.com/keys), passed as
ONSHAPE_ACCESS_KEY / ONSHAPE_SECRET_KEY, sent as HTTP Basic auth (Onshape's documented
scheme for API keys). Enterprise stacks: set ONSHAPE_BASE_URL (e.g. https://acme.onshape.com).

Only GET requests are made; nothing in the Onshape document is modified.
"""
from __future__ import annotations

import base64
import io
import os
import re
import time
import urllib.parse
from dataclasses import dataclass

import numpy as np

from .assembly import Assembly, Mate, Part, mesh_from_stl_bytes

URL_RE = re.compile(
    r"/documents/(?P<did>[0-9a-f]{24})/(?P<wvm>[wvm])/(?P<wvmid>[0-9a-f]{24})/e/(?P<eid>[0-9a-f]{24})"
)


@dataclass
class ElementRef:
    did: str
    wvm: str
    wvmid: str
    eid: str

    @classmethod
    def from_url(cls, url: str) -> "ElementRef":
        m = URL_RE.search(url)
        if not m:
            raise ValueError(
                "not an Onshape element URL; expected .../documents/<did>/w/<wid>/e/<eid> "
                "(open the assembly tab in Onshape and copy the address bar)"
            )
        return cls(**m.groupdict())

    def path(self) -> str:
        return f"/d/{self.did}/{self.wvm}/{self.wvmid}/e/{self.eid}"


class OnshapeError(RuntimeError):
    pass


class OnshapeClient:
    def __init__(self, access_key=None, secret_key=None, base_url=None, session=None):
        import requests

        access_key = access_key or os.environ.get("ONSHAPE_ACCESS_KEY")
        secret_key = secret_key or os.environ.get("ONSHAPE_SECRET_KEY")
        if not access_key or not secret_key:
            raise OnshapeError(
                "Onshape API keys missing: set ONSHAPE_ACCESS_KEY and ONSHAPE_SECRET_KEY "
                "(create a key pair at https://dev-portal.onshape.com/keys, read scope is enough)"
            )
        self.base = (base_url or os.environ.get("ONSHAPE_BASE_URL") or "https://cad.onshape.com").rstrip("/")
        self._base_host = urllib.parse.urlparse(self.base).hostname or ""
        self.s = session or requests.Session()
        token = base64.b64encode(f"{access_key}:{secret_key}".encode()).decode()
        self._auth = {"Authorization": f"Basic {token}"}

    # -- transport ---------------------------------------------------------------------------
    def _trusted(self, url: str) -> bool:
        host = urllib.parse.urlparse(url).hostname or ""
        return host == self._base_host or host.endswith(".onshape.com")

    def get(self, path: str, params=None, accept="application/json", raw=False):
        url = path if path.startswith("http") else self.base + path
        for attempt in range(8):
            headers = {"Accept": accept}
            if self._trusted(url):  # never send the key to a host outside Onshape
                headers.update(self._auth)
            r = self.s.get(url, params=params, headers=headers, allow_redirects=False, timeout=180)
            if r.status_code in (301, 302, 303, 307, 308):
                # Exports redirect to a regional host; re-send auth ourselves (requests drops it).
                url = urllib.parse.urljoin(url, r.headers["Location"])
                params = None
                continue
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            if r.status_code >= 400:
                raise OnshapeError(f"GET {url} -> {r.status_code}: {r.text[:300]}")
            return r.content if raw else r.json()
        raise OnshapeError(f"GET {url}: too many redirects/retries")

    # -- endpoints ---------------------------------------------------------------------------
    def assembly_definition(self, ref: ElementRef) -> dict:
        return self.get(
            f"/api/assemblies{ref.path()}",
            params={
                "includeMateFeatures": "true",
                "includeMateConnectors": "true",
                "includeNonSolids": "false",
            },
        )

    @staticmethod
    def _part_path(p: dict) -> str:
        pid = urllib.parse.quote(p["partId"], safe="")
        return f"/d/{p['documentId']}/m/{p['documentMicroversion']}/e/{p['elementId']}/partid/{pid}"

    def part_stl(self, p: dict, chord_mm: float = 0.05, angle_deg: float = 10.0) -> bytes:
        params = {
            "mode": "binary",
            "units": "millimeter",
            "chordTolerance": chord_mm / 1000.0,  # metres
            "angleTolerance": np.radians(angle_deg),
        }
        if p.get("configuration"):
            params["configuration"] = p["configuration"]
        return self.get(f"/api/parts{self._part_path(p)}/stl", params=params,
                        accept="application/vnd.onshape.v1+octet-stream", raw=True)

    def part_mass(self, p: dict) -> tuple[float | None, np.ndarray | None]:
        params = {"configuration": p["configuration"]} if p.get("configuration") else None
        try:
            mp = self.get(f"/api/parts{self._part_path(p)}/massproperties", params=params)
        except OnshapeError:
            return None, None
        body = next(iter(mp.get("bodies", {}).values()), None)
        if not body or not body.get("hasMass", False):
            return None, None
        mass = float(body["mass"][0])                          # kg
        com = np.array(body["centroid"][:3], float) * 1000.0  # m -> mm (part frame)
        return mass, com


# ------------------------------------------------------------------------------------------------
# Assembly definition -> Assembly
# ------------------------------------------------------------------------------------------------

def _xform(t16) -> np.ndarray:
    """Onshape 4x4 row-major transform, translation in metres -> 4x4 with translation in mm."""
    T = np.array(t16, float).reshape(4, 4)
    T[:3, 3] *= 1000.0
    return T


def _part_key(d: dict) -> tuple:
    return (d.get("documentId"), d.get("documentMicroversion"), d.get("elementId"),
            d.get("partId"), d.get("configuration", ""))


def _asm_key(d: dict) -> tuple:
    return (d.get("documentId"), d.get("documentMicroversion"), d.get("elementId"),
            d.get("configuration", ""))


def assembly_from_definition(defn: dict, fetch_stl, fetch_mass=None, log=print) -> Assembly:
    """Build an Assembly from an Onshape assembly-definition JSON.

    fetch_stl(part_dict) -> STL bytes;  fetch_mass(part_dict) -> (kg|None, com_mm|None).
    Separated from the HTTP client so the conversion can be tested offline.
    """
    root = defn["rootAssembly"]
    occ_T = {tuple(o["path"]): _xform(o["transform"]) for o in root.get("occurrences", [])}
    occ_fixed = {tuple(o["path"]): bool(o.get("fixed", False)) for o in root.get("occurrences", [])}
    occ_hidden = {tuple(o["path"]): bool(o.get("hidden", False)) for o in root.get("occurrences", [])}
    subasm = {_asm_key(s): s for s in defn.get("subAssemblies", [])}
    part_meta = {_part_key(p): p for p in defn.get("parts", [])}

    parts: dict[tuple, Part] = {}
    mesh_cache: dict[tuple, tuple] = {}
    mates: list[Mate] = []

    def walk(instances, features, prefix: tuple, fixed_parent: bool):
        for inst in instances:
            if inst.get("suppressed"):
                continue
            path = prefix + (inst["id"],)
            if inst["type"] == "Assembly":
                sa = subasm.get(_asm_key(inst))
                if sa is None:
                    log(f"  ! sub-assembly {inst.get('name')} missing from definition, skipped")
                    continue
                walk(sa.get("instances", []), sa.get("features", []), path,
                     fixed_parent or occ_fixed.get(path, False))
            elif inst["type"] == "Part":
                if occ_hidden.get(path):
                    continue
                key = _part_key(inst)
                meta = part_meta.get(key, inst)
                if key not in mesh_cache:
                    log(f"  - downloading {inst.get('name')}")
                    V, F = mesh_from_stl_bytes(fetch_stl(meta))
                    mass, com = fetch_mass(meta) if fetch_mass else (None, None)
                    mesh_cache[key] = (V, F, mass, com)
                V, F, mass, com = mesh_cache[key]
                T = occ_T.get(path, np.eye(4))
                Vw = V @ T[:3, :3].T + T[:3, 3]
                comw = None if com is None else T[:3, :3] @ com + T[:3, 3]
                standard = bool(inst.get("isStandardContent") or meta.get("isStandardContent"))
                name = inst.get("name", inst["id"])
                pid = "/".join(path)
                parts[path] = Part(
                    id=pid, name=name, vertices=Vw, faces=F, mass=mass, com=comw,
                    fixed=fixed_parent or occ_fixed.get(path, False),
                    printable=not standard,
                    material="steel" if standard else None,
                )
        for feat in features:
            if feat.get("featureType") != "mate" or feat.get("suppressed"):
                continue
            fd = feat.get("featureData", {})
            ents = fd.get("matedEntities", [])
            if len(ents) != 2:
                continue
            ends = []
            for e in ents:
                p = prefix + tuple(e["matedOccurrence"])
                cs = e["matedCS"]
                T = occ_T.get(p, np.eye(4))
                R = T[:3, :3]
                origin = R @ (np.array(cs["origin"], float) * 1000.0) + T[:3, 3]
                ax = [R @ np.array(cs[k], float) for k in ("xAxis", "yAxis", "zAxis")]
                ends.append((p, origin, np.column_stack(ax)))
            (pa, oa, Ra), (pb, _, _) = ends
            mates.append(Mate(
                name=fd.get("name", feat.get("id", "mate")),
                type=fd.get("mateType", "FASTENED").upper(),
                part_a="/".join(pa), part_b="/".join(pb),
                origin=oa, frame=_orthonormal(Ra),
            ))

    walk(root.get("instances", []), root.get("features", []), (), False)

    # A mate may name a sub-assembly occurrence path that is a prefix of a part path (rigid
    # sub-assemblies); resolve to the first part underneath it.
    ids = {p.id for p in parts.values()}
    for m in mates:
        for attr in ("part_a", "part_b"):
            pid = getattr(m, attr)
            if pid not in ids:
                cand = sorted(i for i in ids if i.startswith(pid + "/"))
                if cand:
                    setattr(m, attr, cand[0])
    mates = [m for m in mates if m.part_a in ids and m.part_b in ids and m.part_a != m.part_b]
    return Assembly(parts=list(parts.values()), mates=mates, name=defn.get("name", "Onshape assembly"))


def _orthonormal(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        Q[:, 2] *= -1
    return Q


def load_onshape(url: str, client: OnshapeClient | None = None, log=print) -> Assembly:
    client = client or OnshapeClient()
    ref = ElementRef.from_url(url)
    log(f"Onshape: reading assembly {ref.eid}")
    defn = client.assembly_definition(ref)
    asm = assembly_from_definition(defn, client.part_stl, client.part_mass, log=log)
    log(f"Onshape: {len(asm.parts)} parts, {len(asm.mates)} mates")
    return asm
