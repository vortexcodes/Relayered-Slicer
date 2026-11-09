"""Outputs: a 3MF Unlayered Slicer opens with the cut planes already placed, VTK for ParaView,
and a JSON plan.

The 3MF carries Metadata/cut_planes.xml in the slicer's own schema (see
src/libslic3r/Format/bbs_3mf.cpp, _extract_cut_planes_from_archive): one <object id=k> per model
object (1-based, in file order) with <plane px py pz nx ny nz [bnx bny bnz]/> in the object's
local frame. Meshes are written centred on their bounding box so a later
ModelObject::center_around_origin() is a no-op and cannot shift the mesh away from the planes.
Planes are listed newest-first with user_ordered=1: in the slicer's manual order the first plane
peels last and claims its whole far side, which is exactly how the planner evaluated them.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Default Extension="json" ContentType="application/json"/>
</Types>
"""
RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


def _f(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".") if abs(x) > 1e-12 else "0"


def write_3mf(path, objects: list[dict], extra: dict | None = None) -> Path:
    """objects: [{"name", "vertices" (local, centred), "faces", "planes": [...], "position": (x,y,z)}]"""
    model = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
             ' <metadata name="Application">Unlayered stress planner</metadata>',
             ' <resources>']
    for i, ob in enumerate(objects, 1):
        model.append(f'  <object id="{i}" name={quoteattr(ob["name"])} type="model"><mesh><vertices>')
        model.extend(f'<vertex x="{_f(v[0])}" y="{_f(v[1])}" z="{_f(v[2])}"/>' for v in ob["vertices"])
        model.append('</vertices><triangles>')
        model.extend(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in ob["faces"])
        model.append('</triangles></mesh></object>')
    model.append(' </resources>')
    model.append(' <build>')
    for i, ob in enumerate(objects, 1):
        x, y, z = ob["position"]
        model.append(f'  <item objectid="{i}" transform="1 0 0 0 1 0 0 0 1 {_f(x)} {_f(y)} {_f(z)}" printable="1"/>')
    model.append(' </build>')
    model.append('</model>')

    cp = ['<?xml version="1.0" encoding="utf-8"?>', '<objects>']
    for i, ob in enumerate(objects, 1):
        if not ob.get("planes"):
            continue
        cp.append(f'<object id="{i}" enabled="1" user_ordered="1">')
        for pl in reversed(ob["planes"]):  # newest first == peels last (claims its far side)
            p, n = pl["point"], pl["normal"]
            attrs = (f'px="{_f(p[0])}" py="{_f(p[1])}" pz="{_f(p[2])}" '
                     f'nx="{_f(n[0])}" ny="{_f(n[1])}" nz="{_f(n[2])}"')
            bn = pl.get("build_normal")
            if bn is not None and np.linalg.norm(np.asarray(bn) - np.asarray(n)) > 1e-9:
                attrs += f' bnx="{_f(bn[0])}" bny="{_f(bn[1])}" bnz="{_f(bn[2])}"'
            cp.append(f'<plane {attrs}/>')
        cp.append('</object>')
    cp.append('</objects>')

    path = Path(path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("3D/3dmodel.model", "\n".join(model))
        if any(ob.get("planes") for ob in objects):
            z.writestr("Metadata/cut_planes.xml", "\n".join(cp))
        if extra:
            z.writestr("Metadata/stress_plan.json", json.dumps(extra, indent=1))
    return path


def write_vtk(path, fea, plan, R: np.ndarray) -> Path:
    """Legacy ASCII VTK unstructured grid (print frame) with per-cell stress fields."""
    nodes = fea.mesh.nodes @ R.T
    tets = fea.mesh.tets
    S = np.einsum("ij,mjk,lk->mil", R, fea.stress, R)
    w, v = np.linalg.eigh(S)
    p1 = v[:, :, -1] * w[:, -1:]  # max principal direction scaled by its value
    out = ["# vtk DataFile Version 3.0", "stress planner", "ASCII", "DATASET UNSTRUCTURED_GRID",
           f"POINTS {len(nodes)} double"]
    out += [f"{a:.5g} {b:.5g} {c:.5g}" for a, b, c in nodes]
    out.append(f"CELLS {len(tets)} {5 * len(tets)}")
    out += [f"4 {a} {b} {c} {d}" for a, b, c, d in tets]
    out.append(f"CELL_TYPES {len(tets)}")
    out += ["10"] * len(tets)
    out.append(f"CELL_DATA {len(tets)}")
    for name, arr in (("von_mises_MPa", fea.vm), ("failure_index", plan.fi),
                      ("safety_factor", 1 / np.maximum(plan.fi, 1e-9)), ("segment", plan.seg)):
        out += [f"SCALARS {name} double 1", "LOOKUP_TABLE default"] + [f"{x:.5g}" for x in arr]
    out.append("VECTORS sigma1_dir double")
    out += [f"{a:.5g} {b:.5g} {c:.5g}" for a, b, c in p1]
    out.append("VECTORS best_build_dir double")
    out += [f"{a:.4g} {b:.4g} {c:.4g}" for a, b, c in plan.pref_dir]
    out.append("POINT_DATA %d" % len(nodes))
    out.append("VECTORS displacement double")
    out += [f"{a:.5g} {b:.5g} {c:.5g}" for a, b, c in fea.u @ R.T]
    Path(path).write_text("\n".join(out))
    return Path(path)
