# Stress-aware multiplanar planning

`tools/stress-planner` turns "this assembly, these real-world loads" into "print each part in this
orientation, cut it here, and build each segment in this direction". Unlayered Slicer then slices
the plan with its existing multiplanar engine. This page explains the method and what it borrows
from other multi-axis slicers.

## Why

An FDM part is weakest across its layers. Interlayer tensile strength is typically 30–70 % of
in-layer strength, and most printed parts that break snap along a layer line. A five-axis machine
can choose a layer direction per region. The strongest part keeps the **principal tension
direction inside the layers** wherever stress is high. Finding those regions means knowing the
stress field, which means knowing the loads. In a real product the loads arrive through the
other parts of the assembly.

## Pipeline

1. **Assembly import (Onshape).** `/api/assemblies` with mate features and connectors gives
   every part occurrence (4×4 placement), every mate (type and mate-connector frame on both parts),
   the Fix flags and sub-assemblies. Parts come in as STL from `/api/parts/.../stl` and mass from
   `/massproperties`. Redirects to regional export hosts are followed with the key re-sent only to
   `*.onshape.com`.
2. **Assembly statics.** Each part is a rigid body. Each mate is a reaction wrench restricted to
   the components its type transmits. For example, a revolute carries no moment about its axis,
   a slider no force along it, a ball only forces, and a planar only the normal force and two
   moments. Grounded parts get one fastened connection to the world at their fixture.
   Equilibrium of all bodies, `A r = −f_ext`, gives every mate's reaction. A part that cannot be
   balanced is reported as a mechanism. With redundant mates the minimum-norm solution is used,
   with moments scaled by the assembly size.
3. **Per-part FEA.** TetGen meshes the part and linear tets are solved with SciPy. Each mate
   reaction is applied over the surface patch around its connector with an RBE3-style
   distribution, which gives the exact resultant force and moment and adds no stiffness.
   Self-weight is applied as a body force. Because statics already balanced the part, rigid
   motion is removed with six Lagrange constraints instead of fixed nodes, so there are no fake
   support hot spots. Output: the stress tensor, von Mises and the principal frame per element.
4. **Failure index per build direction.** For a layer normal `d`, the traction on the layer plane
   is `t = σd`, with peel `σn = d·σ·d` and interlayer shear `τ = |t − σn d|`.
   `FI = max( √((σn⁺/S_z)² + (τ/S_τz)²), σ_vm/S_xy )` and the safety factor is `SF = 1/FI`.
   This is evaluated for every element against hundreds of candidate directions at once.
5. **Orientation (3-axis baseline).** The planner scores flat-face resting orientations first,
   then a Fibonacci sphere of directions. The objective is the volume-weighted 99th-percentile FI,
   which ignores load-patch singularities, times `1 + w · support_area_fraction`.
6. **Multiplanar refinement.** In the chosen print frame, the planner looks at the critical
   elements (top 5 % FI) and the directions they would prefer, within the tilt limit. It then
   tries planes normal to the most-preferred directions at a sweep of offsets. A new plane claims
   its whole far side, as in Unlayered Slicer's peel semantics. Elements on the cut face are also
   checked against the plane normal, because a cut is a layer interface. Bed-level vertices must
   stay on the root's side. The best plane is accepted if it improves the objective by ≥ 3 %. The
   planner repeats until the part reaches the target SF, gains stop, or `--max-planes` is reached.
7. **Export.** Meshes are written centred in the print frame. `Metadata/cut_planes.xml` uses the
   schema read by `_BBS_3MF_Importer::_extract_cut_planes_from_archive`, with `user_ordered="1"`
   and the newest plane first, so the slicer's peel order matches what was evaluated.

## What came from other multi-axis slicers

Only ideas were borrowed, no code. Each item names the source and what changed here.

| Source | Idea used | Adaptation here |
|---|---|---|
| **S³-Slicer** (Zhang, Fang et al., SIGGRAPH Asia 2022) | Multi-objective layer optimisation driven by FEA: **strength reinforcement** (layer normals ⟂ max principal stress) traded off against **support-free** and surface-quality terms. | The same strength + support trade-off, evaluated per discrete segment direction instead of a continuous quaternion deformation field, because Unlayered's core is piecewise-planar. |
| **Reinforced FDM** (Fang et al., SIGGRAPH 2020) | Align filament with principal stress lines. Stress anisotropy is the reason curved layers help. | The σ₁ field drives the critical-element direction votes. It is drawn as stress lines in the report and exported as `sigma1_dir` in VTK. |
| **S4 Slicer** (open source, GPL-3.0) | Per-region build direction from a rotation field, constrained by a maximum overhang and tilt, so "printable" is part of the optimisation, not an afterthought. | Overhang penalty per segment direction, plus `--max-tilt` limiting candidate directions. |
| **Open5x** (Hong, Gardner et al., 2022) | Conformal or multi-directional printing inside the CAD workflow. Start from the design environment, not from a loose STL. | The import starts from the Onshape assembly, so loads come from the real mating parts. |
| **Multi-axis motion planning work** (Zhang et al.) | Collision-free sequencing of multi-axis segments. | Left to Unlayered Slicer's existing hotend collision check after slicing. |
| **Unlayered Slicer** (this repo) | Declarative cut planes, peel semantics, per-segment build normals, Bed clip. | The planner's output format and its rules: root on the bed, newest plane claims its far side. |

## Roadmap

- **Several load cases.** Take the worst FI across them, for example a drop, a hanging load and
  assembly torque.
- **Stiffness-based split for redundant mates.** Solve the assembly as one FE model with contact
  springs at the mates instead of the minimum-norm statics.
- **Strength modifiers.** Write modifier volumes into the 3MF that add walls and infill where SF
  is low, and set the infill direction per segment along σ₁.
- **In-slicer panel.** A web-app page (see `WEB_APP_BRIDGE.md`) that calls `slicer.setCutPlanes`
  live, so the plan can be applied to an object already on the plate.
- **Curved layers.** If the slicing core ever gets a deform-slice-undeform path (S³/S4 style), the
  per-element `best_build_dir` field is already the input such a path needs.
