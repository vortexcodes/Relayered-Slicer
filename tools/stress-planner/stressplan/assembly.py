"""Assembly data model (units: mm, kg, N) + local bundle I/O.

A *bundle* is a folder with assembly.json and one STL per part, so an assembly pulled from
Onshape once can be re-analysed offline, versioned, or built by hand without Onshape:

    {
      "name": "Arm",
      "parts": [{"id": "base", "name": "Base", "stl": "base.stl", "fixed": true,
                 "printable": true, "mass": 0.12, "com": [0,0,10]}],
      "mates": [{"name": "Pivot", "type": "REVOLUTE", "part_a": "base", "part_b": "arm",
                 "origin": [0,0,40], "z_axis": [0,1,0], "x_axis": [1,0,0]}]
    }

All coordinates are in the assembly frame, millimetres. STLs are stored already placed.
"""
from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DENSITY = {  # kg/mm^3
    None: 1.24e-6,      # PLA-ish default for printed parts
    "PLA": 1.24e-6, "PETG": 1.27e-6, "ABS": 1.04e-6, "ASA": 1.07e-6, "PA": 1.14e-6,
    "PA-CF": 1.20e-6, "steel": 7.85e-6, "aluminum": 2.70e-6,
}


@dataclass
class Part:
    id: str
    name: str
    vertices: np.ndarray            # (N,3) mm, assembly frame
    faces: np.ndarray               # (M,3) int
    mass: float | None = None       # kg
    com: np.ndarray | None = None   # mm, assembly frame
    fixed: bool = False             # grounded (Onshape "Fix")
    printable: bool = True          # False for bolts/bearings/standard content: rigid, no FEA
    material: str | None = None

    def volume(self) -> float:
        v0, v1, v2 = (self.vertices[self.faces[:, i]] for i in range(3))
        return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)

    def centroid(self) -> np.ndarray:
        v0, v1, v2 = (self.vertices[self.faces[:, i]] for i in range(3))
        vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0
        if abs(vol.sum()) < 1e-12:
            return self.vertices.mean(0)
        return ((v0 + v1 + v2) / 4.0 * vol[:, None]).sum(0) / vol.sum()

    def resolved_mass(self) -> tuple[float, np.ndarray]:
        m = self.mass if self.mass is not None else abs(self.volume()) * DENSITY.get(self.material, DENSITY[None])
        c = self.com if self.com is not None else self.centroid()
        return m, c


@dataclass
class Mate:
    name: str
    type: str                       # FASTENED, REVOLUTE, SLIDER, CYLINDRICAL, PIN_SLOT, PLANAR, BALL, PARALLEL
    part_a: str
    part_b: str
    origin: np.ndarray              # mm, assembly frame
    frame: np.ndarray               # 3x3, columns = mate x, y, z axes (z = mate axis)


@dataclass
class Assembly:
    parts: list[Part]
    mates: list[Mate] = field(default_factory=list)
    name: str = "assembly"

    def part(self, key: str) -> Part:
        for p in self.parts:
            if p.id == key or p.name == key:
                return p
        low = key.lower()
        hits = [p for p in self.parts if low in p.name.lower()]
        if len(hits) == 1:
            return hits[0]
        raise KeyError(f"no part named {key!r}; parts: {[p.name for p in self.parts]}")

    def mate(self, key: str) -> Mate:
        for m in self.mates:
            if m.name == key:
                return m
        raise KeyError(f"no mate named {key!r}; mates: {[m.name for m in self.mates]}")

    # -- bundle I/O --------------------------------------------------------------------------
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        js = {"name": self.name, "parts": [], "mates": []}
        used = set()
        for p in self.parts:
            stem = "".join(c if c.isalnum() else "_" for c in p.name).strip("_") or "part"
            fn, k = f"{stem}.stl", 1
            while fn in used:
                k += 1
                fn = f"{stem}_{k}.stl"
            used.add(fn)
            write_stl(folder / fn, p.vertices, p.faces)
            js["parts"].append({
                "id": p.id, "name": p.name, "stl": fn, "fixed": p.fixed, "printable": p.printable,
                "material": p.material, "mass": p.mass,
                "com": None if p.com is None else [float(x) for x in p.com],
            })
        for m in self.mates:
            js["mates"].append({
                "name": m.name, "type": m.type, "part_a": m.part_a, "part_b": m.part_b,
                "origin": [float(x) for x in m.origin],
                "x_axis": [float(x) for x in m.frame[:, 0]],
                "z_axis": [float(x) for x in m.frame[:, 2]],
            })
        (folder / "assembly.json").write_text(json.dumps(js, indent=2))
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> "Assembly":
        folder = Path(folder)
        js = json.loads((folder / "assembly.json").read_text())
        parts = []
        for pj in js["parts"]:
            V, F = read_stl(folder / pj["stl"])
            parts.append(Part(
                id=pj.get("id", pj["name"]), name=pj["name"], vertices=V, faces=F,
                mass=pj.get("mass"), com=None if pj.get("com") is None else np.array(pj["com"], float),
                fixed=bool(pj.get("fixed", False)), printable=bool(pj.get("printable", True)),
                material=pj.get("material"),
            ))
        mates = []
        for mj in js.get("mates", []):
            mates.append(Mate(
                name=mj["name"], type=mj.get("type", "FASTENED").upper(),
                part_a=mj["part_a"], part_b=mj["part_b"], origin=np.array(mj["origin"], float),
                frame=frame_from_axes(mj.get("z_axis", [0, 0, 1]), mj.get("x_axis")),
            ))
        asm = cls(parts=parts, mates=mates, name=js.get("name", folder.name))
        ids = {p.id for p in parts}
        for m in asm.mates:  # allow mates to reference parts by name
            for attr in ("part_a", "part_b"):
                if getattr(m, attr) not in ids:
                    setattr(m, attr, asm.part(getattr(m, attr)).id)
        return asm


def frame_from_axes(z, x=None) -> np.ndarray:
    z = np.asarray(z, float)
    z = z / np.linalg.norm(z)
    if x is None or abs(np.dot(np.asarray(x, float), z)) > 0.99 * np.linalg.norm(x):
        x = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.asarray(x, float) - np.dot(x, z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


# ------------------------------------------------------------------------------------------------
# STL
# ------------------------------------------------------------------------------------------------

def mesh_from_stl_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    if data[:5].lower() == b"solid" and b"facet" in data[:400]:
        tris = []
        for line in data.decode(errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("vertex"):
                tris.append([float(t) for t in line.split()[1:4]])
        tri = np.array(tris, float).reshape(-1, 3, 3)
    else:
        n = struct.unpack("<I", data[80:84])[0]
        rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                            count=n, offset=84)
        tri = rec["v"].astype(float)
    return weld(tri)


def weld(tri: np.ndarray, tol: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    pts = tri.reshape(-1, 3)
    q = np.round(pts / tol).astype(np.int64)
    _, idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    V = pts[idx]
    F = inv.reshape(-1, 3)
    keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    return V, F[keep]


def read_stl(path) -> tuple[np.ndarray, np.ndarray]:
    return mesh_from_stl_bytes(Path(path).read_bytes())


def write_stl(path, V: np.ndarray, F: np.ndarray) -> None:
    tri = V[F].astype(np.float32)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    buf = io.BytesIO()
    buf.write(b"stressplan".ljust(80, b" "))
    buf.write(struct.pack("<I", len(F)))
    rec = np.zeros(len(F), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = n, tri
    buf.write(rec.tobytes())
    Path(path).write_bytes(buf.getvalue())
