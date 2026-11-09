"""Assembly -> statics -> per-part FEA -> plan -> 3MF / VTK / report."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import fea as fea_mod
from .assembly import Assembly, write_stl
from .export import write_3mf, write_vtk
from .planner import PlanOptions, Surface, plan_part
from .report import write_report
from .statics import LoadCase, solve_statics


def run(asm: Assembly, lc: LoadCase, out_dir, opt: PlanOptions | None = None,
        resolution: float = 30.0, only: list[str] | None = None, log=print) -> dict:
    opt = opt or PlanOptions()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log(f"[1/3] statics: {len(asm.parts)} parts, {len(asm.mates)} mates, load case '{lc.name}'")
    st = solve_statics(asm, lc, log=log)
    if st.unbalanced:
        log(f"  ! cannot balance {st.unbalanced}: a part is free to move under the loads "
            f"(missing mate, or mate type too loose). Results for it are not meaningful.")
    if st.indeterminate:
        log("  i statically indeterminate (redundant mates); reactions use the minimum-norm split")
    for r in sorted(st.reactions, key=lambda r: -r["force_N"])[:12]:
        log(f"    {r['connection']:<28} {r['part_a']:>16} -> {r['part_b']:<16} "
            f"F={r['force_N']:8.1f} N  M={r['moment_Nmm']:9.0f} N*mm")

    targets = [p for p in asm.parts if p.printable]
    if only:
        keep = {asm.part(k).id for k in only}
        targets = [p for p in targets if p.id in keep]
    log(f"[2/3] FEA + planning for {len(targets)} printable part(s)")

    objects, summary, viz = [], [], []
    x_cursor = 0.0
    for p in targets:
        log(f"  * {p.name}")
        mesh = fea_mod.tet_mesh(p.vertices, p.faces, resolution=resolution)
        log(f"    mesh: {len(mesh.nodes)} nodes, {len(mesh.tets)} tets, h={mesh.h:.2f} mm")
        res = fea_mod.solve(mesh, lc.material, st.wrenches[p.id], st.gravity.get(p.id), log=log)
        log(f"    von Mises p99 {np.quantile(res.vm, 0.99):.2f} MPa, max {res.vm.max():.2f} MPa, "
            f"max displacement {np.linalg.norm(res.u, axis=1).max():.3f} mm")
        plan = plan_part(res, Surface.from_mesh(p.vertices, p.faces), lc.material, opt, log=log)

        # Print frame, centred on the bounding box (object-local frame of the 3MF).
        Vp = p.vertices @ plan.R.T
        lo, hi = Vp.min(0), Vp.max(0)
        centre = (lo + hi) / 2
        Vl = Vp - centre
        planes_local = [{"point": pl["point"] - centre, "normal": pl["normal"],
                         "build_normal": pl["build_normal"]} for pl in plan.planes]
        size = hi - lo
        pos = (x_cursor + size[0] / 2 + 10, 100.0, size[2] / 2)
        x_cursor += size[0] + 15
        objects.append({"name": p.name, "vertices": Vl, "faces": p.faces, "planes": planes_local,
                        "position": pos})
        safe = "".join(c if c.isalnum() else "_" for c in p.name).strip("_") or "part"
        write_stl(out / f"{safe}_print_orientation.stl", Vl, p.faces)
        write_vtk(out / f"{safe}_stress.vtu.vtk", res, plan, plan.R)

        m = plan.metrics
        summary.append({
            "part": p.name, **m,
            "rotation_asm_to_print": plan.R.tolist(),
            "cut_planes_local": [{k: [float(x) for x in v] for k, v in pl.items()} for pl in planes_local],
            "segment_build_dirs": [[float(x) for x in d] for d in plan.seg_dirs],
            "mesh": {"nodes": int(len(mesh.nodes)), "tets": int(len(mesh.tets)), "h_mm": mesh.h},
            "load_imbalance_N": res.imbalance,
        })
        viz.append(_viz_payload(p.name, res, plan, centre, planes_local, m))
        log(f"    safety factor: as modelled {m['sf_as_modelled']:.2f} | best 3-axis "
            f"{m['sf_best_3axis']:.2f} | multiplanar {m['sf_plan']:.2f} "
            f"({m['n_segments']} segment{'s' if m['n_segments'] > 1 else ''})")

    log("[3/3] writing outputs")
    result = {
        "assembly": asm.name, "load_case": lc.name, "material": lc.material.__dict__,
        "load_factor": lc.load_factor,
        "statics": {"reactions": st.reactions, "residual_N": st.residual,
                    "unbalanced": st.unbalanced, "indeterminate": st.indeterminate},
        "parts": summary, "seconds": round(time.time() - t0, 1),
    }
    if objects:
        write_3mf(out / "stress_plan.3mf", objects, extra=result)
    (out / "plan.json").write_text(json.dumps(result, indent=2))
    write_report(out / "report.html", result, viz)
    log(f"done in {result['seconds']} s -> {out}/stress_plan.3mf, report.html, plan.json")
    return result


def _viz_payload(name, res, plan, centre, planes_local, metrics, max_lines=1500):
    """Compact data for the HTML viewer, print frame, centred (matches the 3MF)."""
    R = plan.R
    sf = res.mesh.surf_faces
    used, inv = np.unique(sf, return_inverse=True)
    V = res.mesh.nodes[used] @ R.T - centre
    # Per-vertex safety factor: worst (min) over the tets that touch the vertex.
    sf_el = 1 / np.maximum(plan.fi, 1e-9)
    node_sf = np.full(len(res.mesh.nodes), np.inf)
    seg_node = np.zeros(len(res.mesh.nodes), int)
    for a in range(4):
        np.minimum.at(node_sf, res.mesh.tets[:, a], sf_el)
        seg_node[res.mesh.tets[:, a]] = plan.seg
    # Principal-stress glyphs at the most stressed elements.
    order = np.argsort(res.vm)[::-1][:max_lines]
    S = np.einsum("ij,mjk,lk->mil", R, res.stress[order], R)
    w, v = np.linalg.eigh(S)
    k = np.argmax(np.abs(w), axis=1)
    dirs = v[np.arange(len(order)), :, k]
    sign = np.sign(w[np.arange(len(order)), k])
    cen = res.centroids[order] @ R.T - centre
    rnd = lambda a, n=2: np.round(np.asarray(a, float), n).ravel().tolist()
    return {
        "name": name, "metrics": metrics,
        "vertices": rnd(V), "faces": inv.reshape(-1, 3).ravel().tolist(),
        "sf": rnd(node_sf[used], 3), "seg": seg_node[used].tolist(),
        "glyph_pos": rnd(cen), "glyph_dir": rnd(dirs, 3), "glyph_sign": sign.astype(int).tolist(),
        "glyph_len": float(res.mesh.h * 1.6),
        "planes": [{"point": rnd(pl["point"]), "normal": rnd(pl["normal"], 4)} for pl in planes_local],
    }
