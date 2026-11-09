"""Strength-driven build-direction planner for multiplanar / five-axis FDM.

Failure model (transversely isotropic FDM part). For a build direction d (= layer normal):
  * across the layers the weak interlayer bond sees the traction t = sigma . d, split into
      sigma_n = d . sigma . d        (peel)      and   tau = |t - sigma_n d|   (interlayer shear)
    FI_z  = sqrt( (max(sigma_n,0)/S_z)^2 + (tau/S_shear_z)^2 )
  * inside the layers the beads behave close to bulk:  FI_xy = sigma_vm / S_xy
  FI(d) = max(FI_z, FI_xy);  safety factor = 1 / FI.
This is the "keep principal tension inside the layers" goal of stress-aligned slicers
(S^3-Slicer's strength term, Reinforced FDM), evaluated per element against discrete
directions instead of a continuous curved-layer field.

Plan = one part orientation (the root segment builds +Z on the bed) plus up to N cut planes in
Unlayered Slicer's peel semantics: the most recently added plane claims everything on its far
side and builds along its normal. Planes are chosen greedily to cut down the robust peak failure
index (volume-weighted 99th percentile, load-patch singularities excluded), with a support-area
penalty (overhang along each segment's build direction) so a stronger plan that needs a forest
of supports does not win by default. Each cut face is itself a layer interface, so elements on
the cut are also checked against the plane normal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fea import FEAResult
from .statics import Material


def fibonacci_dirs(n: int, max_tilt_deg: float = 180.0) -> np.ndarray:
    """Roughly uniform unit vectors whose angle from +Z is <= max_tilt_deg."""
    zmin = np.cos(np.radians(min(max_tilt_deg, 180.0)))
    i = np.arange(n) + 0.5
    z = 1.0 - (1.0 - zmin) * i / n
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    phi = i * np.pi * (3 - np.sqrt(5))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def failure_index(stress: np.ndarray, vm: np.ndarray, dirs: np.ndarray, mat: Material,
                  chunk: int = 64) -> np.ndarray:
    """(M,K) failure index of each element for each build direction."""
    dirs = np.atleast_2d(dirs)
    out = np.empty((len(stress), len(dirs)), np.float32)
    fi_xy = (vm / mat.S_xy)[:, None]
    for k0 in range(0, len(dirs), chunk):
        d = dirs[k0:k0 + chunk]
        t = np.einsum("mij,kj->mki", stress, d)
        sn = np.einsum("mki,ki->mk", t, d)
        tau2 = np.maximum(np.einsum("mki,mki->mk", t, t) - sn * sn, 0.0)
        fi_z = np.sqrt((np.maximum(sn, 0) / mat.S_z) ** 2 + tau2 / mat.S_shear_z ** 2)
        out[:, k0:k0 + chunk] = np.maximum(fi_z, fi_xy)
    return out


def wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(cw, q * cw[-1]), len(x) - 1)])


def flat_rest_dirs(surf: "Surface", min_frac: float = 0.01) -> np.ndarray:
    """Build directions that put a large planar face of the part flat on the bed."""
    key = np.round(surf.normals * 50).astype(int)  # ~1 deg bins
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    area = np.bincount(inv.ravel(), weights=surf.areas)
    good = np.where(area >= min_frac * surf.areas.sum())[0]
    good = good[np.argsort(area[good])[::-1]][:24]
    dirs = []
    for g in good:
        m = inv.ravel() == g
        n = (surf.normals[m] * surf.areas[m, None]).sum(0)
        n /= np.linalg.norm(n)
        # the face must be on the convex bottom along -n (otherwise it is a pocket, not a base)
        if np.min(surf.centroids @ -n) >= np.min(surf.centroids[m] @ -n) - 1e-3:
            dirs.append(-n)
    return np.array(dirs).reshape(-1, 3)


def rotation_to_z(d: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector d to +Z."""
    d = d / np.linalg.norm(d)
    z = np.array([0, 0, 1.0])
    v = np.cross(d, z)
    c = float(np.dot(d, z))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


@dataclass
class Surface:
    centroids: np.ndarray
    normals: np.ndarray
    areas: np.ndarray

    @classmethod
    def from_mesh(cls, V, F):
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n = np.cross(b - a, c - a)
        ar = np.linalg.norm(n, axis=1)
        return cls((a + b + c) / 3, n / np.maximum(ar, 1e-20)[:, None], ar / 2)

    def rotated(self, R):
        return Surface(self.centroids @ R.T, self.normals @ R.T, self.areas)


@dataclass
class PlanOptions:
    max_planes: int = 3
    max_tilt_deg: float = 90.0
    overhang_deg: float = 45.0
    support_weight: float = 0.25    # how much a full-support part costs, relative to strength
    min_gain: float = 0.03          # a plane must improve the objective by >= 3 %
    min_segment_frac: float = 0.04  # each segment >= 4 % of the part volume
    n_orient: int = 600
    n_tilt: int = 300
    max_elems: int = 40_000
    keep_orientation: bool = False  # True: do not re-orient, keep the part as modelled (+Z up)
    target_sf: float = 3.0          # stop adding cut planes once the part reaches this (0 = never stop)


@dataclass
class Plan:
    R: np.ndarray                          # assembly frame -> print frame rotation
    planes: list[dict]                     # print frame: {"point","normal","build_normal"} in add order
    seg: np.ndarray                        # per element (all elements) segment id
    seg_dirs: list[np.ndarray]             # print frame build direction per segment
    fi: np.ndarray                         # per element FI under the plan
    pref_dir: np.ndarray                   # per element best direction (print frame)
    metrics: dict = field(default_factory=dict)


def _objective(fi, w, surf_over_area, total_area, opt):
    p99 = wquantile(fi, w, 0.99)
    return p99 * (1 + opt.support_weight * surf_over_area / total_area), p99


def _overhang_area(surf: Surface, face_seg, seg_dirs, seg_base_off, opt, bed_tol):
    thr = -np.cos(np.radians(opt.overhang_deg))
    area = 0.0
    for s, d in enumerate(seg_dirs):
        m = face_seg == s
        if not m.any():
            continue
        proj = surf.centroids[m] @ d
        down = surf.normals[m] @ d < thr
        base = seg_base_off[s]
        rest = np.abs(proj - base) < bed_tol      # faces lying on the bed / own cut plane
        area += surf.areas[m][down & ~rest].sum()
    return float(area)


def plan_part(fea: FEAResult, surf_asm: Surface, mat: Material, opt: PlanOptions, log=print) -> Plan:
    M = len(fea.vm)
    rng = np.random.default_rng(0)
    # Subsample elements for the search; keep every highly stressed one.
    if M > opt.max_elems:
        top = np.argsort(fea.vm)[-opt.max_elems // 4:]
        rest = rng.choice(np.setdiff1d(np.arange(M), top), opt.max_elems - len(top), replace=False)
        sub = np.concatenate([top, rest])
        wscale = np.ones(len(sub))
        wscale[len(top):] = (M - len(top)) / len(rest)
    else:
        sub, wscale = np.arange(M), np.ones(M)

    # Exclude elements touching load patches (point-load singularities) from the score.
    on_patch = np.isin(fea.mesh.tets, fea.load_nodes).any(1)
    valid = ~on_patch
    if valid[sub].sum() < 0.5 * len(sub):
        valid[:] = True
    S, vm, cen = fea.stress[sub], fea.vm[sub], fea.centroids[sub]
    w = fea.volumes[sub] * wscale * valid[sub]
    if w.sum() <= 0:
        w = fea.volumes[sub] * wscale

    total_area = surf_asm.areas.sum()
    bed_tol = max(0.5, fea.mesh.h * 0.5)

    # ---- 1. orientation (root build direction) -----------------------------------------
    up = np.array([0, 0, 1.0])
    fi_up = failure_index(S, vm, up, mat)[:, 0]
    oh_up = _overhang_area(surf_asm, np.zeros(len(surf_asm.areas), int), [up],
                           [surf_asm.centroids[:, 2].min()], opt, bed_tol)
    J_up, p99_up = _objective(fi_up, w, oh_up, total_area, opt)

    if opt.keep_orientation:
        d_best, J_best, p99_best = up, J_up, p99_up
    else:
        # Resting on a flat face is what a real print needs, so those directions come first and
        # a free (sampled) direction has to beat them by 2 % to be picked.
        rest = flat_rest_dirs(surf_asm)
        cand = np.vstack([rest, fibonacci_dirs(opt.n_orient)])
        FI = failure_index(S, vm, cand, mat)
        best = (J_up, p99_up, up)
        for k, d in enumerate(cand):
            handicap = 1.0 if k < len(rest) else 1.02
            p99 = wquantile(FI[:, k], w, 0.99)
            if p99 * handicap >= best[0]:
                continue  # cannot win even with zero support
            proj = surf_asm.centroids @ d
            oh = _overhang_area(surf_asm, np.zeros(len(proj), int), [d], [proj.min()], opt, bed_tol)
            J, p99 = _objective(FI[:, k], w, oh, total_area, opt)
            if J * handicap < best[0]:
                best = (J * handicap, p99, d)
        J_best, p99_best, d_best = best
    R = rotation_to_z(d_best)
    log(f"    orientation: build dir {np.round(d_best, 3)}  SF as modelled {1 / p99_up:.2f} -> "
        f"best 3-axis {1 / p99_best:.2f}")

    # ---- 2. multiplanar refinement in the print frame -----------------------------------
    Sp = np.einsum("ij,mjk,lk->mil", R, S, R)
    cp = cen @ R.T
    surf = surf_asm.rotated(R)
    tilt = fibonacci_dirs(opt.n_tilt, opt.max_tilt_deg)
    tilt = tilt[tilt[:, 2] > 0.02] if opt.max_tilt_deg >= 90 else tilt
    FIc = failure_index(Sp, vm, tilt, mat)
    fi = failure_index(Sp, vm, up, mat)[:, 0]
    seg = np.zeros(len(sub), int)
    face_seg = np.zeros(len(surf.areas), int)
    seg_dirs, seg_off = [up], [surf.centroids[:, 2].min()]
    planes = []
    # Unlayered Slicer flips any plane whose far side holds the lowest vertex (the root must rest
    # on the bed), so every bed-level vertex has to stay on the near side of every plane.
    nodes_p = fea.mesh.nodes @ R.T
    bed_pts = nodes_p[nodes_p[:, 2] <= nodes_p[:, 2].min() + 0.05]
    vol_all = (fea.volumes[sub] * wscale)
    vtot = vol_all.sum()
    oh = _overhang_area(surf, face_seg, seg_dirs, seg_off, opt, bed_tol)
    J, p99 = _objective(fi, w, oh, total_area, opt)
    J3, p993 = J, p99
    band_h = max(fea.mesh.h, 0.5)

    for it in range(opt.max_planes):
        if opt.target_sf > 0 and 1 / p99 >= opt.target_sf:
            log(f"    SF {1 / p99:.2f} >= target {opt.target_sf:g}: no cut planes needed")
            break
        crit = fi >= wquantile(fi, w, 0.95)
        crit &= w > 0
        if not crit.any():
            break
        pref = np.argmin(FIc[crit], axis=1)
        votes = np.bincount(pref, weights=w[crit], minlength=len(tilt))
        top = np.argsort(votes)[::-1][:10]
        top = [k for k in top if votes[k] > 0]
        best = None
        for k in top:
            d = tilt[k]
            if any(np.dot(d, sd) > 0.985 for sd in seg_dirs):
                continue  # same direction as an existing segment: no gain from a cut
            s = cp @ d
            lo, hi = np.quantile(s, [0.05, 0.95])
            for off in np.linspace(lo, hi, 16):
                if (bed_pts @ d).max() >= off - 0.2:
                    continue
                far = s > off
                vf = vol_all[far].sum() / vtot
                if vf < opt.min_segment_frac or vf > 1 - opt.min_segment_frac:
                    continue
                new_seg = seg.copy()
                new_seg[far] = len(seg_dirs)
                remain = np.bincount(new_seg, weights=vol_all, minlength=len(seg_dirs) + 1) / vtot
                if (remain[:len(seg_dirs)] < opt.min_segment_frac).any():
                    continue
                fi_new = fi.copy()
                fi_new[far] = FIc[far, k]
                band = np.abs(s - off) < band_h
                fi_new[band] = np.maximum(fi_new[band], FIc[band, k])
                fs = face_seg.copy()
                fs[surf.centroids @ d > off] = len(seg_dirs)
                oh_n = _overhang_area(surf, fs, seg_dirs + [d], seg_off + [off], opt, bed_tol)
                Jn, p99n = _objective(fi_new, w, oh_n, total_area, opt)
                if best is None or Jn < best[0]:
                    best = (Jn, p99n, k, off, far, fi_new, fs)
        if best is None or best[0] > J * (1 - opt.min_gain):
            break
        J, p99, k, off, far, fi, face_seg = best
        d = tilt[k]
        seg[far] = len(seg_dirs)
        seg_dirs.append(d)
        seg_off.append(off)
        bandc = cp[np.abs(cp @ d - off) < band_h * 2]
        c0 = bandc.mean(0) if len(bandc) else cp[far].mean(0)
        point = c0 - (np.dot(c0, d) - off) * d
        planes.append({"point": point, "normal": d.copy(), "build_normal": d.copy()})
        tilt_deg = np.degrees(np.arccos(np.clip(d[2], -1, 1)))
        log(f"    + cut plane {len(planes)}: tilt {tilt_deg:.0f} deg, SF -> {1 / p99:.2f}")

    # ---- expand to every element ---------------------------------------------------------
    cp_all = fea.centroids @ R.T
    seg_all = np.zeros(M, int)
    for i, pl in enumerate(planes):
        seg_all[cp_all @ pl["normal"] > np.dot(pl["point"], pl["normal"])] = i + 1
    S_all = np.einsum("ij,mjk,lk->mil", R, fea.stress, R)
    fi_all = np.empty(M, np.float32)
    for s_id, d in enumerate(seg_dirs):
        m = seg_all == s_id
        if m.any():
            fi_all[m] = failure_index(S_all[m], fea.vm[m], d, mat)[:, 0]
    pref_all = tilt[np.argmin(failure_index(S_all, fea.vm, tilt, mat), axis=1)]

    w_all = fea.volumes * valid
    metrics = {
        "sf_as_modelled": 1 / p99_up, "sf_best_3axis": 1 / p993, "sf_plan": 1 / p99,
        "sf_plan_min_all": float(1 / max(fi_all.max(), 1e-12)),
        "sf_plan_p99_all": 1 / wquantile(fi_all, w_all if w_all.sum() > 0 else fea.volumes, 0.99),
        "support_area_frac": float(_overhang_area(surf, face_seg, seg_dirs, seg_off, opt, bed_tol) / total_area),
        "n_segments": len(seg_dirs), "build_dir_asm": [float(x) for x in d_best],
        "von_mises_max": float(fea.vm.max()), "von_mises_p99": wquantile(fea.vm, fea.volumes, 0.99),
        "max_disp_mm": float(np.linalg.norm(fea.u, axis=1).max()),
    }
    return Plan(R=R, planes=planes, seg=seg_all, seg_dirs=seg_dirs, fi=fi_all, pref_dir=pref_all,
                metrics=metrics)
