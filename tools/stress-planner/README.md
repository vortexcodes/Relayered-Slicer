# Stress planner

Pull an assembly from Onshape, work out the forces on every part, and get a print plan that
keeps each part's tension inside its layers. The output is a `.3mf` that Unlayered Slicer
opens with the multiplanar cut planes already placed.

```
Onshape assembly ──► rigid-body statics ──► per-part FEA ──► orientation + cut planes ──► stress_plan.3mf
 (parts, placements,   (mate reactions:       (tet mesh,         (strength vs. support,      (+ report.html,
  mates, mass, Fix)     who pushes where)      principal stress)   peel-order planes)          VTK, plan.json)
```

How it works and which ideas came from other multi-axis slicers: [docs/stress-planner.md](../../docs/stress-planner.md).

## Install

Python 3.10+.

```bash
cd tools/stress-planner
```

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

## Try it (no Onshape needed)

```bash
python -m stressplan demo -o out_demo
```

This plans a shelf bracket with a 15 kg load. Open `out_demo/report.html` to see the result. Then
open `out_demo/stress_plan.3mf` in Unlayered Slicer. Add `--keep-orientation` to see cut planes
used on a part that has to print upright.

## Your Onshape assembly

1. Create an API key pair at <https://dev-portal.onshape.com/keys>. Read scope is enough.
2. Export the keys:
   ```bash
   export ONSHAPE_ACCESS_KEY=... ONSHAPE_SECRET_KEY=...
   ```
   On an enterprise stack, also set `ONSHAPE_BASE_URL=https://yourco.onshape.com`.
3. Open the **assembly** tab in Onshape and copy the URL. Then download the assembly and get a skeleton load file:
   ```bash
   python -m stressplan fetch "https://cad.onshape.com/documents/<did>/w/<wid>/e/<eid>" my_bundle
   ```
   ```bash
   python -m stressplan template my_bundle > loads.json
   ```
4. Edit `loads.json` to say where the real-world forces go (see below), then run:
   ```bash
   python -m stressplan local my_bundle -l loads.json -o out
   ```
   You can also go straight from the URL with `python -m stressplan onshape "<url>" -l loads.json -o out`.

What it reads from Onshape: every part (STL), its placement, every mate (type and mate
connector frame), mass properties, and the **Fix** flag. Fixed parts are grounded. Standard
content, such as bolts and bearings, stays in the force path as rigid bodies but is not analysed
or exported. Nothing in the document is changed: the tool only makes GET requests.

## The load file

An assembly knows its geometry and how its parts connect. It does not know what the outside
world does to it. That part is up to you:

```jsonc
{
  "name": "15 kg on the hook, x2 safety",
  "material": "PETG",              // PLA PETG ABS ASA PC PA-CF, or {"E":..,"nu":..,"S_xy":..,"S_z":..,"S_shear_z":..}
  "load_factor": 2.0,              // multiply every load (design margin)
  "gravity": [0, 0, -9.81],        // assembly frame, m/s^2; null to ignore self-weight
  "fixed": ["Wall plate"],         // optional: overrides Onshape's Fix flags
  "fixtures": {"Wall plate": {"point": [0, -6, 60], "radius": 80}},  // where a fixed part is held (default: its bottom face)
  "loads": [
    {"part": "Hook", "point": [0, 70, 60], "force": [0, 0, -147], "label": "bag"},
    {"part": "Arm", "at_mate": "Grip", "force": [20, 0, 0], "moment": [0, 0, 500]}
  ]
}
```

Units are mm, N and N·mm, all in the assembly's coordinate frame. To find a point, click a
vertex in Onshape and read its coordinates, or use a mate connector and `at_mate`. Part names can
be any unique part of the name.

## Options

| flag | default | |
|---|---|---|
| `--max-planes N` | 3 | Cut planes per part. 0 gives the best orientation only (3-axis). |
| `--target-sf X` | 3 | Stops adding planes once a part reaches this safety factor. 0 means always maximise. |
| `--max-tilt DEG` | 90 | Steepest segment build direction. Match your machine's A axis, or the slicer's *Max tilt angle* in virtual mode. |
| `--support-weight W` | 0.25 | Cost of support area against strength. 0 means strength only. |
| `--overhang DEG` | 45 | Overhang angle that needs support. |
| `--keep-orientation` | off | Keeps each part as modelled (+Z up) and only adds cut planes. |
| `--resolution R` | 30 | FEA elements across the part diagonal. Higher is slower and finer. |
| `--only NAME…` | all | Analyses only these parts. |

## Outputs (`-o out`)

| file | |
|---|---|
| `stress_plan.3mf` | Every printable part in its planned orientation, with `Metadata/cut_planes.xml`. Open it in Unlayered Slicer, then **Printer settings ▸ Motion ability ▸ Non-planar printing** must be on. |
| `report.html` | Safety factors (as modelled / best 3-axis / multiplanar), mate reactions, and a 3D view of the stress lines, safety factor and segments. |
| `<part>_stress.vtu.vtk` | Full field for ParaView: von Mises, failure index, safety factor, segment, σ₁ direction, best build direction, displacement. |
| `<part>_print_orientation.stl` | The part in its planned orientation. |
| `plan.json` | Everything above as numbers, including the rotation from assembly frame to print frame. |

## Tests

```bash
python -m pytest -q tests
```

The tests check the FEA against beam theory, the statics against hand-computed reactions and mechanism
detection, the planner's choices, the 3MF schema the slicer reads, and the Onshape JSON
conversion. They use a fake assembly definition, so they run offline.

## Limits (read before trusting a number)

- **Loads are yours.** Everything downstream is only as good as `loads.json`.
- **Redundant mates.** When several mates share a load path, as with two bolts, the split between
  them is the minimum-norm one, not a stiffness-based one. The report says when this applies.
- **Contact is a patch.** Each mate's force is spread over the surface nodes around the mate
  connector. Stress right at those patches is unreliable, so it is left out of the score.
- **Linear tets, linear material.** Use the safety factors to compare orientations, not as a rating.
  The interlayer strengths in the presets are conservative typical values. Put in your own
  measured values for real work.
- **Piecewise-planar layers.** Unlayered Slicer slices each segment with flat layers, so the
  planner picks segment directions. It does not produce continuously curved stress-following
  layers. See the roadmap in the docs.
