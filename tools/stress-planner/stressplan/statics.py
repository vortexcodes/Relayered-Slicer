"""Load cases and assembly-level rigid-body statics.

Every part is a rigid body. Each mate transmits only the wrench components its type can carry
(a revolute carries no moment about its axis, a slider no force along it, ...). Grounded parts
get one extra fastened connection to the world at their fixture. Solving equilibrium of every
body gives the reaction wrench at every mate -- i.e. *where and how hard the rest of the
assembly pushes on each part*. Those become the boundary loads of each part's FEA, so every
part's FEA is self-equilibrated and needs no artificial supports.

Statically indeterminate assemblies (several fastened mates in a loop) have infinitely many
reaction sets; we take the minimum-norm one (moments scaled by the assembly size so forces and
moments weigh alike). That is a heuristic, not a stiffness-based split -- noted in the report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .assembly import Assembly, Mate, frame_from_axes

# Transmitted components per mate type, mate frame: 0-2 force x,y,z ; 3-5 moment x,y,z.
MATE_DOF = {
    "FASTENED":    (0, 1, 2, 3, 4, 5),
    "REVOLUTE":    (0, 1, 2, 3, 4),
    "SLIDER":      (0, 1, 3, 4, 5),
    "CYLINDRICAL": (0, 1, 3, 4),
    "PIN_SLOT":    (1, 2, 3, 4),
    "PLANAR":      (2, 3, 4),
    "BALL":        (0, 1, 2),
    "PARALLEL":    (3, 4),
}


@dataclass
class Material:
    name: str = "PLA"
    E: float = 3500.0          # MPa
    nu: float = 0.35
    S_xy: float = 50.0         # MPa, tensile strength along the bead (in-layer)
    S_z: float = 25.0          # MPa, tensile strength across layers (interlayer bond)
    S_shear_z: float = 20.0    # MPa, interlayer shear strength

    @classmethod
    def preset(cls, name: str) -> "Material":
        presets = {  # conservative, typical datasheet-ish values for printed parts
            "PLA":   cls("PLA", 3500, 0.35, 50, 25, 20),
            "PETG":  cls("PETG", 2100, 0.38, 45, 25, 20),
            "ABS":   cls("ABS", 2200, 0.35, 35, 15, 13),
            "ASA":   cls("ASA", 2200, 0.35, 38, 17, 14),
            "PA-CF": cls("PA-CF", 6000, 0.35, 90, 30, 25),
            "PC":    cls("PC", 2300, 0.37, 60, 30, 24),
        }
        return presets[name.upper()]


@dataclass
class PointLoad:
    part: str
    point: np.ndarray            # mm
    force: np.ndarray            # N
    moment: np.ndarray           # N*mm
    radius: float | None = None  # load patch radius, mm (None = automatic)
    label: str = ""


@dataclass
class LoadCase:
    name: str = "load case"
    gravity: np.ndarray | None = field(default_factory=lambda: np.array([0, 0, -9.81]))  # m/s^2
    loads: list[PointLoad] = field(default_factory=list)
    fixed: list[str] | None = None            # override grounded parts (else Onshape "Fix")
    fixtures: dict = field(default_factory=dict)  # part -> {"point":[..], "radius": r} or "bottom"
    material: Material = field(default_factory=Material)
    load_factor: float = 1.0

    @classmethod
    def from_dict(cls, d: dict, asm: Assembly) -> "LoadCase":
        mat = d.get("material", "PLA")
        material = Material.preset(mat) if isinstance(mat, str) else Material(**mat)
        g = d.get("gravity", [0, 0, -9.81])
        lc = cls(name=d.get("name", "load case"), gravity=None if g is None else np.array(g, float),
                 fixed=d.get("fixed"), fixtures=d.get("fixtures", {}), material=material,
                 load_factor=float(d.get("load_factor", 1.0)))
        for i, L in enumerate(d.get("loads", [])):
            part = asm.part(L["part"])
            if "at_mate" in L:
                pt = asm.mate(L["at_mate"]).origin
            elif "point" in L:
                pt = np.array(L["point"], float)
            else:
                pt = part.centroid()
            lc.loads.append(PointLoad(
                part=part.id, point=np.asarray(pt, float),
                force=np.array(L.get("force", [0, 0, 0]), float),
                moment=np.array(L.get("moment", [0, 0, 0]), float),
                radius=L.get("radius"), label=L.get("label", f"load {i + 1}"),
            ))
        return lc

    @classmethod
    def load(cls, path, asm: Assembly) -> "LoadCase":
        return cls.from_dict(json.loads(Path(path).read_text()), asm)


@dataclass
class Wrench:
    """A wrench applied to one part at a point (assembly frame)."""
    point: np.ndarray
    force: np.ndarray
    moment: np.ndarray
    source: str                 # "mate:<name>", "ground", "load:<label>"
    radius: float | None = None
    axis: np.ndarray | None = None   # mate axis, used to pick the contact patch


@dataclass
class StaticsResult:
    wrenches: dict[str, list[Wrench]]       # part id -> applied wrenches (excl. gravity)
    gravity: dict[str, np.ndarray]          # part id -> body force vector N (applied over volume)
    reactions: list[dict]                   # table for the report
    residual: float                         # equilibrium residual, N
    unbalanced: list[str]                   # parts that cannot be balanced (mechanism)
    indeterminate: bool


def _skew_cross(r, e):
    return np.cross(r, e)


def ground_fixture(part, spec) -> tuple[np.ndarray, float | None]:
    """Where a grounded part is held. Default: its lowest face region (it sits on something)."""
    if isinstance(spec, dict) and "point" in spec:
        return np.array(spec["point"], float), spec.get("radius")
    V = part.vertices
    zmin = V[:, 2].min()
    band = V[V[:, 2] <= zmin + max(0.5, 0.02 * np.ptp(V[:, 2]))]
    return np.array([band[:, 0].mean(), band[:, 1].mean(), zmin]), None


def solve_statics(asm: Assembly, lc: LoadCase, log=print) -> StaticsResult:
    parts = asm.parts
    idx = {p.id: i for i, p in enumerate(parts)}
    n = len(parts)
    lf = lc.load_factor

    fixed_ids = set()
    if lc.fixed:
        fixed_ids = {asm.part(k).id for k in lc.fixed}
    else:
        fixed_ids = {p.id for p in parts if p.fixed}
    if not fixed_ids:
        # Nothing grounded in Onshape and no override: ground the heaviest part.
        heaviest = max(parts, key=lambda p: p.resolved_mass()[0])
        fixed_ids = {heaviest.id}
        log(f"  ! no fixed part in assembly; grounding '{heaviest.name}' (set \"fixed\" in the load file)")

    allP = np.vstack([p.vertices for p in parts])
    L = float(np.linalg.norm(np.ptp(allP, axis=0))) or 1.0
    refs = [p.resolved_mass()[1] for p in parts]

    # External loads (assembly frame) per body: resultant force and moment about the body ref.
    b = np.zeros(6 * n)
    gravity = {}
    for i, p in enumerate(parts):
        if lc.gravity is not None:
            m, c = p.resolved_mass()
            Fg = m * lc.gravity * lf
            gravity[p.id] = Fg
            b[6 * i:6 * i + 3] += Fg
            b[6 * i + 3:6 * i + 6] += np.cross(c - refs[i], Fg)
    for ld in lc.loads:
        i = idx[ld.part]
        b[6 * i:6 * i + 3] += ld.force * lf
        b[6 * i + 3:6 * i + 6] += ld.moment * lf + np.cross(ld.point - refs[i], ld.force * lf)

    # Unknown reaction components.
    connections = []  # (label, body_a or None, body_b, origin, frame, dofs, axis, radius)
    for m in asm.mates:
        t = m.type if m.type in MATE_DOF else "FASTENED"
        if m.type not in MATE_DOF:
            log(f"  ! mate '{m.name}' type {m.type} not modelled; treated as FASTENED")
        connections.append((f"mate:{m.name}", idx[m.part_a], idx[m.part_b], m.origin, m.frame,
                            MATE_DOF[t], m.frame[:, 2], None))
    for pid in sorted(fixed_ids):
        p = parts[idx[pid]]
        pt, rad = ground_fixture(p, lc.fixtures.get(p.name, lc.fixtures.get(pid, "bottom")))
        connections.append(("ground", None, idx[pid], pt, np.eye(3), MATE_DOF["FASTENED"], None, rad))

    cols, col_meta = [], []
    for ci, (label, ia, ib, o, R, dofs, axis, rad) in enumerate(connections):
        for k in dofs:
            col = np.zeros(6 * n)
            e = R[:, k % 3]
            for body, sgn in ((ib, 1.0), (ia, -1.0)):
                if body is None:
                    continue
                if k < 3:
                    col[6 * body:6 * body + 3] += sgn * e
                    col[6 * body + 3:6 * body + 6] += sgn * np.cross(o - refs[body], e)
                else:
                    col[6 * body + 3:6 * body + 6] += sgn * e * L
            cols.append(col)
            col_meta.append((ci, k))
    A = np.column_stack(cols) if cols else np.zeros((6 * n, 0))

    # Scale moment rows by 1/L so residuals are in N.
    S = np.ones(6 * n)
    for i in range(n):
        S[6 * i + 3:6 * i + 6] = 1.0 / L
    x, *_ = np.linalg.lstsq(A * S[:, None], -b * S, rcond=None)
    res = (A @ x + b) * S
    rank = np.linalg.matrix_rank(A * S[:, None]) if A.size else 0
    indeterminate = A.shape[1] > rank

    unbalanced = []
    scale = max(1.0, np.abs(b * S).max())
    for i, p in enumerate(parts):
        if np.linalg.norm(res[6 * i:6 * i + 6]) > 1e-6 * scale + 1e-9:
            unbalanced.append(p.name)

    # Collect reaction wrenches.
    per_conn = {}
    for (ci, k), val in zip(col_meta, x):
        f, mo = per_conn.setdefault(ci, (np.zeros(3), np.zeros(3)))
        R = connections[ci][4]
        if k < 3:
            f += val * R[:, k]
        else:
            mo += val * L * R[:, k % 3]

    wrenches: dict[str, list[Wrench]] = {p.id: [] for p in parts}
    reactions = []
    for ci, (label, ia, ib, o, R, dofs, axis, rad) in enumerate(connections):
        f, mo = per_conn.get(ci, (np.zeros(3), np.zeros(3)))
        wrenches[parts[ib].id].append(Wrench(o.copy(), f.copy(), mo.copy(), label, rad, axis))
        if ia is not None:
            wrenches[parts[ia].id].append(Wrench(o.copy(), -f, -mo, label, rad, axis))
        reactions.append({
            "connection": label, "part_a": parts[ia].name if ia is not None else "world",
            "part_b": parts[ib].name, "force_N": float(np.linalg.norm(f)),
            "moment_Nmm": float(np.linalg.norm(mo)),
            "F": [float(v) for v in f], "M": [float(v) for v in mo], "at": [float(v) for v in o],
        })
    for ld in lc.loads:
        wrenches[ld.part].append(Wrench(ld.point, ld.force * lf, ld.moment * lf, f"load:{ld.label}", ld.radius))

    return StaticsResult(wrenches=wrenches, gravity=gravity, reactions=reactions,
                         residual=float(np.abs(res).max()) if res.size else 0.0,
                         unbalanced=unbalanced, indeterminate=bool(indeterminate))
