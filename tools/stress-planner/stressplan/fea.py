"""Linear-elastic tetrahedral FEA of one part under self-equilibrated loads.

Units: mm, N, MPa. Linear tets (constant stress per element) -- coarse but fast, and what the
planner needs is the *direction field* of principal stress and where it is high, not a certified
peak value. Rigid-body motion is removed with six Lagrange constraints (zero mean translation and
rotation) instead of fixed nodes, so there are no fake support hot spots: the statics step has
already balanced every load on the part, including the ground reaction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .statics import Material, Wrench


@dataclass
class TetMesh:
    nodes: np.ndarray        # (N,3)
    tets: np.ndarray         # (M,4)
    surf_faces: np.ndarray   # (K,3) boundary triangles (outward)
    h: float                 # target edge length

    @property
    def surf_nodes(self) -> np.ndarray:
        return np.unique(self.surf_faces)


@dataclass
class FEAResult:
    mesh: TetMesh
    u: np.ndarray            # (N,3) displacement, mm
    stress: np.ndarray       # (M,3,3) Cauchy stress, MPa
    vm: np.ndarray           # (M,) von Mises
    p_val: np.ndarray        # (M,3) principal stresses, descending
    p_dir: np.ndarray        # (M,3,3) principal directions, columns match p_val
    centroids: np.ndarray    # (M,3)
    volumes: np.ndarray      # (M,)
    load_nodes: np.ndarray   # node indices that carry patch loads (singular zones)
    imbalance: float         # |sum F| of applied loads, N (should be ~0)


def tet_mesh(V: np.ndarray, F: np.ndarray, resolution: float = 30.0, max_tets: int = 120_000) -> TetMesh:
    import tetgen
    import trimesh

    m = trimesh.Trimesh(V, F, process=True)
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
    trimesh.repair.fix_normals(m)
    diag = float(np.linalg.norm(m.extents))
    vol = abs(float(m.volume)) if m.is_volume else float(np.prod(m.extents)) * 0.3
    h = max(diag / resolution, (vol * 8.5 / max_tets) ** (1 / 3))
    maxvol = h ** 3 / (6 * np.sqrt(2))
    tg = tetgen.TetGen(np.asarray(m.vertices, float), np.asarray(m.faces, np.int32))
    try:
        nodes, tets, *_ = tg.tetrahedralize(quality=True, minratio=1.5, mindihedral=10,
                                            maxvolume=maxvol, fixedvolume=True)
    except RuntimeError:
        tg = tetgen.TetGen(np.asarray(m.vertices, float), np.asarray(m.faces, np.int32))
        tg.make_manifold(verbose=False)
        nodes, tets, *_ = tg.tetrahedralize(quality=True, minratio=2.0, maxvolume=maxvol, fixedvolume=True)
    tets = np.asarray(tets, np.int64)
    # Orient all tets positively.
    X = nodes[tets]
    det = np.einsum("ij,ij->i", X[:, 1] - X[:, 0], np.cross(X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]))
    tets[det < 0] = tets[det < 0][:, [0, 2, 1, 3]]
    tets = tets[np.abs(det) > 1e-12]
    return TetMesh(nodes=np.asarray(nodes, float), tets=tets, surf_faces=boundary_faces(tets), h=h)


def boundary_faces(tets: np.ndarray) -> np.ndarray:
    # Faces opposite each vertex, ordered so the normal points out of a positively oriented tet.
    f = np.vstack([tets[:, [1, 2, 3]], tets[:, [0, 3, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 1]]])
    key = np.sort(f, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return f[cnt[inv.ravel()] == 1]


def _elasticity(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[np.arange(3, 6), np.arange(3, 6)] = mu
    return D


def _bmatrices(nodes, tets):
    X = nodes[tets]                                   # (M,4,3)
    J = X[:, 1:] - X[:, :1]                           # rows: x1-x0, x2-x0, x3-x0
    vol = np.linalg.det(J) / 6.0
    G = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    dN = G @ np.linalg.inv(np.transpose(J, (0, 2, 1)))  # (M,4,3) gradients of N_a
    M = len(tets)
    B = np.zeros((M, 6, 12))
    for a in range(4):
        dx, dy, dz = dN[:, a, 0], dN[:, a, 1], dN[:, a, 2]
        c = 3 * a
        B[:, 0, c] = dx
        B[:, 1, c + 1] = dy
        B[:, 2, c + 2] = dz
        B[:, 3, c + 1] = dz; B[:, 3, c + 2] = dy      # gamma_yz
        B[:, 4, c] = dz;     B[:, 4, c + 2] = dx      # gamma_xz
        B[:, 5, c] = dy;     B[:, 5, c + 1] = dx      # gamma_xy
    return B, vol


def _patch_nodes(mesh: TetMesh, w: Wrench, min_nodes=12) -> np.ndarray:
    sn = mesh.surf_nodes
    P = mesh.nodes[sn]
    if w.source == "ground" and w.radius is None:
        z0 = P[:, 2].min()
        band = max(mesh.h * 0.75, 0.3)
        sel = sn[P[:, 2] <= z0 + band]
        if len(sel) >= 3:
            return sel
    d = np.linalg.norm(P - w.point, axis=1)
    if w.radius is not None:
        sel = sn[d <= w.radius]
        if len(sel) >= 3:
            return sel
    d0 = d.min()
    diag = float(np.linalg.norm(np.ptp(mesh.nodes, axis=0)))
    r = d0 + 1.5 * mesh.h
    while True:
        sel = sn[d <= r]
        if len(sel) >= min_nodes or r > 0.3 * diag:
            return sel if len(sel) >= 3 else sn[np.argsort(d)[:min_nodes]]
        r *= 1.35


def distribute_wrench(nodes_xyz: np.ndarray, point, force, moment) -> np.ndarray:
    """Spread a wrench over patch nodes (RBE3-style, equal weights): exact resultant force and
    moment, no added stiffness."""
    c = nodes_xyz.mean(0)
    r = nodes_xyz - c
    Mc = moment + np.cross(point - c, force)
    I = (r * r).sum() * np.eye(3) - r.T @ r
    omega = np.linalg.pinv(I) @ Mc
    return force / len(r) + np.cross(omega, r)


def solve(mesh: TetMesh, mat: Material, wrenches: list[Wrench], body_force: np.ndarray | None = None,
          log=print) -> FEAResult:
    nodes, tets = mesh.nodes, mesh.tets
    N, M = len(nodes), len(tets)
    D = _elasticity(mat.E, mat.nu)
    B, vol = _bmatrices(nodes, tets)
    Ke = np.einsum("mji,jk,mkl->mil", B, D, B) * vol[:, None, None]   # (M,12,12)
    dofs = (3 * tets[:, :, None] + np.arange(3)).reshape(M, 12)
    rows = np.repeat(dofs, 12, axis=1).ravel()
    cols = np.tile(dofs, (1, 12)).ravel()
    K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(3 * N, 3 * N))

    f = np.zeros((N, 3))
    load_nodes = []
    for w in wrenches:
        if np.linalg.norm(w.force) < 1e-12 and np.linalg.norm(w.moment) < 1e-12:
            continue
        sel = _patch_nodes(mesh, w)
        f[sel] += distribute_wrench(nodes[sel], w.point, w.force, w.moment)
        load_nodes.append(sel)
    if body_force is not None and np.linalg.norm(body_force) > 0:
        share = (vol / vol.sum())[:, None] * body_force / 4.0
        for a in range(4):
            np.add.at(f, tets[:, a], share)
    imbalance = float(np.linalg.norm(f.sum(0)))

    # Rigid-body modes as constraints: sum u = 0, sum x cross u = 0.
    c = nodes.mean(0)
    x = nodes - c
    C = np.zeros((6, 3 * N))
    for k in range(3):
        C[k, k::3] = 1.0
    C[3, 1::3], C[3, 2::3] = -x[:, 2], x[:, 1]
    C[4, 0::3], C[4, 2::3] = x[:, 2], -x[:, 0]
    C[5, 0::3], C[5, 1::3] = -x[:, 1], x[:, 0]
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    # Make the load exactly balanced (removes round-off and gravity/patch mismatch).
    fv = f.ravel()
    Q, _ = np.linalg.qr(Cn.T)
    fv = fv - Q @ (Q.T @ fv)

    if 3 * N <= 150_000:
        scale = K.diagonal().mean()
        A = sp.bmat([[K, sp.csr_matrix(Cn.T * scale)], [sp.csr_matrix(Cn * scale), None]], format="csc")
        sol = spla.spsolve(A, np.concatenate([fv, np.zeros(6)]))
        u = sol[:3 * N]
    else:
        d = K.diagonal()
        P = sp.diags(1.0 / np.where(d > 0, d, 1.0))
        u, info = spla.cg(K, fv, M=P, rtol=1e-8, maxiter=20_000)
        if info:
            log(f"    ! CG did not fully converge ({info})")
        u = u - Q @ (Q.T @ u)
    u = u.reshape(N, 3)

    strain = np.einsum("mij,mj->mi", B, u[tets].reshape(M, 12))
    s = strain @ D.T
    S = np.empty((M, 3, 3))
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    S[:, 1, 2] = S[:, 2, 1] = s[:, 3]
    S[:, 0, 2] = S[:, 2, 0] = s[:, 4]
    S[:, 0, 1] = S[:, 1, 0] = s[:, 5]
    vm = np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
                 + 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    w_, v_ = np.linalg.eigh(S)
    p_val, p_dir = w_[:, ::-1], v_[:, :, ::-1]
    return FEAResult(mesh=mesh, u=u, stress=S, vm=vm, p_val=p_val, p_dir=p_dir,
                     centroids=nodes[tets].mean(1), volumes=vol,
                     load_nodes=np.unique(np.concatenate(load_nodes)) if load_nodes else np.zeros(0, int),
                     imbalance=imbalance)
