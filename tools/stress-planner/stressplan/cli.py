"""Command line.

  python -m stressplan demo                      -> runs the built-in bracket demo
  python -m stressplan onshape URL -l loads.json -> pulls the assembly from Onshape and plans it
  python -m stressplan local BUNDLE -l loads.json-> same, from a saved bundle folder
  python -m stressplan fetch URL BUNDLE          -> only download the Onshape assembly to a bundle
  python -m stressplan template BUNDLE|URL       -> print a loads.json skeleton for an assembly
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .assembly import Assembly
from .planner import PlanOptions
from .statics import LoadCase


def _plan_args(ap):
    ap.add_argument("-l", "--loads", help="load case JSON (see examples/)")
    ap.add_argument("-o", "--out", default="stress_out", help="output folder")
    ap.add_argument("--max-planes", type=int, default=3, help="cut planes per part (0 = orientation only)")
    ap.add_argument("--max-tilt", type=float, default=90.0, help="max segment tilt from the bed normal, deg")
    ap.add_argument("--overhang", type=float, default=45.0, help="overhang angle needing support, deg")
    ap.add_argument("--support-weight", type=float, default=0.25,
                    help="penalty for support area vs. strength (0 = strength only)")
    ap.add_argument("--target-sf", type=float, default=3.0,
                    help="stop adding cut planes once a part reaches this safety factor (0 = maximise)")
    ap.add_argument("--keep-orientation", action="store_true", help="keep each part as modelled (+Z up)")
    ap.add_argument("--resolution", type=float, default=30.0, help="FEA elements across the part diagonal")
    ap.add_argument("--only", nargs="*", help="only these part names")


def _opts(a) -> PlanOptions:
    return PlanOptions(max_planes=a.max_planes, max_tilt_deg=a.max_tilt, overhang_deg=a.overhang,
                       support_weight=a.support_weight, keep_orientation=a.keep_orientation,
                       target_sf=a.target_sf)


def template(asm: Assembly) -> dict:
    fixed = [p.name for p in asm.parts if p.fixed]
    return {
        "name": "describe the load case",
        "gravity": [0, 0, -9.81],
        "material": "PETG",
        "load_factor": 2.0,
        "fixed": fixed or ["<part that is bolted to the world>"],
        "fixtures": {n: {"point": [0, 0, 0], "radius": 10} for n in (fixed or [])},
        "loads": [{"part": asm.parts[0].name, "point": [0, 0, 0], "force": [0, 0, -100],
                   "label": "what pushes here", "radius": 5}],
        "_parts": [p.name for p in asm.parts],
        "_mates": [f"{m.name} ({m.type}): {m.part_a} <-> {m.part_b}" for m in asm.mates],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stressplan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run the built-in bracket demo")
    _plan_args(d)
    o = sub.add_parser("onshape", help="plan an Onshape assembly")
    o.add_argument("url")
    o.add_argument("--save-bundle", help="also save the downloaded assembly here")
    _plan_args(o)
    lo = sub.add_parser("local", help="plan a saved bundle")
    lo.add_argument("bundle")
    _plan_args(lo)
    f = sub.add_parser("fetch", help="download an Onshape assembly to a bundle")
    f.add_argument("url")
    f.add_argument("bundle")
    t = sub.add_parser("template", help="print a loads.json skeleton")
    t.add_argument("source", help="bundle folder or Onshape URL")
    a = ap.parse_args(argv)

    from .pipeline import run

    if a.cmd == "demo":
        from .demo import DEMO_LOADS, demo_assembly
        asm = demo_assembly()
        lc = LoadCase.load(a.loads, asm) if a.loads else LoadCase.from_dict(DEMO_LOADS, asm)
        run(asm, lc, a.out, _opts(a), resolution=a.resolution, only=a.only)
        return 0
    if a.cmd in ("onshape", "fetch") or (a.cmd == "template" and a.source.startswith("http")):
        from .onshape import load_onshape
        asm = load_onshape(a.url if a.cmd != "template" else a.source)
        if a.cmd == "fetch":
            print(f"saved {asm.save(a.bundle)}")
            return 0
        if a.cmd == "onshape" and a.save_bundle:
            asm.save(a.save_bundle)
    elif a.cmd == "template" or a.cmd == "local":
        asm = Assembly.load(a.bundle if a.cmd == "local" else a.source)
    if a.cmd == "template":
        print(json.dumps(template(asm), indent=2))
        return 0
    if not a.loads:
        print("no --loads given: using gravity only. Run `stressplan template ...` for a skeleton.",
              file=sys.stderr)
        lc = LoadCase(name="gravity only")
    else:
        lc = LoadCase.load(a.loads, asm)
    run(asm, lc, a.out, _opts(a), resolution=a.resolution, only=a.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
