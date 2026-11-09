"""Built-in demo assembly: a shelf bracket bolted to a wall plate, with a hook at the tip.

Modelled the way people draw it in CAD (vertical leg up the wall, +Z up). Printed as modelled the
leg's bending tension runs straight across the layers -- the classic place FDM brackets snap --
so it shows clearly what re-orienting and multiplanar cutting buy.
"""
from __future__ import annotations

import numpy as np

from .assembly import Assembly, Mate, Part, frame_from_axes


def _ear_clip(P):
    """Triangulate a simple CCW polygon (small, so O(n^3) is fine)."""
    idx = list(range(len(P)))
    tris = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    while len(idx) > 3:
        for k in range(len(idx)):
            i, j, l = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            if cross(P[i], P[j], P[l]) <= 0:
                continue
            if any(cross(P[i], P[j], P[m]) >= 0 and cross(P[j], P[l], P[m]) >= 0 and cross(P[l], P[i], P[m]) >= 0
                   for m in idx if m not in (i, j, l)):
                continue
            tris.append((i, j, l))
            idx.pop(k)
            break
    tris.append(tuple(idx))
    return tris


def extrude_yz(poly_yz, x0, x1):
    """Extrude a simple CCW polygon given in the YZ plane along +X (outward normals)."""
    P = [tuple(map(float, p)) for p in poly_yz]
    n = len(P)
    V = np.array([(x0, y, z) for y, z in P] + [(x1, y, z) for y, z in P])
    F = []
    for a, b, c in _ear_clip(P):
        F.append((a, c, b))              # x0 cap faces -X
        F.append((a + n, b + n, c + n))  # x1 cap faces +X
    for i in range(n):
        j = (i + 1) % n
        F += [(i, j, j + n), (i, j + n, i + n)]
    return V, np.array(F)


def box(lo, hi):
    import trimesh

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(hi - lo)
    return np.asarray(m.vertices) + (lo + hi) / 2, np.asarray(m.faces)


def demo_assembly() -> Assembly:
    plate_V, plate_F = box((-30, -6, 0), (30, 0, 120))
    t, leg_h, arm_l = 8.0, 100.0, 80.0
    # L bracket with a 12 mm inner fillet-chamfer at the corner.
    L = [(0, 0), (t, 0), (t, leg_h - t - 12), (t + 12, leg_h - t), (arm_l, leg_h - t),
         (arm_l, leg_h), (0, leg_h)]
    br_V, br_F = extrude_yz(L, -10, 10)
    hook_V, hook_F = box((-5, 64, 60), (5, 76, leg_h - t))
    parts = [
        Part("wall", "Wall plate", plate_V, plate_F, fixed=True, printable=False, material="steel"),
        Part("bracket", "Bracket", br_V, br_F, material="PETG"),
        Part("hook", "Hook", hook_V, hook_F, material="PETG"),
    ]
    y = frame_from_axes([0, 1, 0])
    z = frame_from_axes([0, 0, 1])
    mates = [
        Mate("Bolt low", "FASTENED", "wall", "bracket", np.array([0, 0, 20.0]), y),
        Mate("Bolt high", "FASTENED", "wall", "bracket", np.array([0, 0, 70.0]), y),
        Mate("Hook joint", "FASTENED", "bracket", "hook", np.array([0, 70, leg_h - t]), z),
    ]
    return Assembly(parts, mates, name="Shelf bracket (demo)")


DEMO_LOADS = {
    "name": "15 kg hanging from the hook",
    "gravity": [0, 0, -9.81],
    "material": "PETG",
    "load_factor": 1.0,
    "fixtures": {"Wall plate": {"point": [0, -6, 60], "radius": 80}},
    "loads": [{"part": "Hook", "point": [0, 70, 60], "force": [0, 0, -147], "label": "hanging load"}],
}
