"""Sharp-corner rig: a periodic array of one polygon (square, isosceles triangle of apex angle beta, rhombus with two sharp edges; any rotation) in Stokes flow driven by a body force on the fluid, against a Fourier
volume-penalisation reference of the same cell (stokes_array_ref.py, here with a polygon indicator).  The trailing edge of an airfoil is the motivating case (docs/plan-next-steps.md, "Decisions and order").

Two checks:

  operator   static: the reference velocity field interpolated to the particles of a cut lattice (uniform density, so the pressure force vanishes), the SPH viscous acceleration (fluid pair sum + wall closure) against
             nu lap u of the reference; the error per particle class, in units of the body force f: FACE (nearest wall point on an edge, farther than H from every vertex), NEAR-VERTEX FACE (nearest point on an edge
             within H of a vertex), VERTEX (nearest point = a vertex: the corner quadrant).  No time integration, no pressure: the wall closure alone.
  array      dynamic: the steady superficial velocity U (both components: a rotated body gives a transverse flow) against the reference, K = F / (mu U_x) and U_y / U_x.

Reference: the body force f acts on the fluid only (F = f (1 - c), the SPH driver); `fourier` returns U, the force on the solid and the fields, cached in .tmp by (shape, N).  The polygon is centred in the unit cell and must
satisfy extent + support < 1/2 (Body._checkCompact).

    python scripts/studies/corner_rig.py operator --shape triangle --beta 30 --n 48 --visc morris
    python scripts/studies/corner_rig.py array    --shape triangle --beta 30 --n 48 --visc morris --consistent
"""
import argparse
import hashlib
import math
import os
import time

import numpy as np
import torch

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries import paths
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.scene.scene import Body, Scene, SurfaceRep
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D, DeltaSPHConfig
from warpSPHBoundaries.sim.loads import LoadHistory

XI = 2.821384729


# ----------------------------------------------------------------------------------------------------------------- geometry
def polygon(shape, c, beta=30.0, rot=0.0, aspect=0.3, sides=64):
    """vertices [n, 2] (counter-clockwise, centred on the centroid at the origin) of a polygon of area c: square; isosceles triangle with apex angle beta (deg) pointing along +x (the trailing edge of a body
    in a stream along +x); rhombus with the acute angle beta at both ends of the x axis (two sharp edges); then rotated by rot (deg, counter-clockwise)."""
    b = math.radians(beta)
    if shape == "square":
        h = 0.5 * math.sqrt(c)
        V = np.array([(-h, -h), (h, -h), (h, h), (-h, h)])
    elif shape == "triangle":
        # base of half-width w at x = 0, apex at x = L with tan(b / 2) = w / L; area = w L
        L = math.sqrt(c / math.tan(b / 2))
        w = L * math.tan(b / 2)
        V = np.array([(0.0, -w), (L, 0.0), (0.0, w)])
        V = V - V.mean(0)
    elif shape == "rhombus":
        # diagonals 2a (along x) and 2e with tan(b / 2) = e / a; area = 2 a e
        a = math.sqrt(c / (2 * math.tan(b / 2)))
        e = a * math.tan(b / 2)
        V = np.array([(-a, 0.0), (0.0, -e), (a, 0.0), (0.0, e)])
    elif shape == "ngon":
        # regular polygon with `sides` vertices (64: a disk up to angles of 174 deg, the corner-free baseline of the same rig); area = sides / 2 R^2 sin(2 pi / sides)
        m = int(sides)
        R = math.sqrt(2 * c / (m * math.sin(2 * math.pi / m)))
        th = 2 * math.pi * np.arange(m) / m
        V = np.stack([R * np.cos(th), R * np.sin(th)], 1)
    else:
        raise ValueError(shape)
    r = math.radians(rot)
    R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    V = V @ R.T
    s = 0.5 * np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1])
    return V if s > 0 else V[::-1].copy()


def signed_distance(P, V):
    """signed distance of points P [M, 2] to the closed polygon V [n, 2]: + outside (fluid), - inside; plus the index of the nearest feature: vertex k -> k, edge k (V[k] -> V[k+1], interior point) -> n + k."""
    n = len(V)
    best = np.full(len(P), np.inf)
    feat = np.zeros(len(P), dtype=int)
    for k in range(n):
        a, b = V[k], V[(k + 1) % n]
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
        d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        f = np.where(t <= 0.0, k, np.where(t >= 1.0, (k + 1) % n, n + k))
        m = d < best
        best[m], feat[m] = d[m], f[m]
    inside = np.zeros(len(P), dtype=bool)
    for k in range(n):                                                                  # even-odd ray casting along +x
        a, b = V[k], V[(k + 1) % n]
        cross = ((a[1] > P[:, 1]) != (b[1] > P[:, 1]))
        xint = a[0] + (P[:, 1] - a[1]) * (b[0] - a[0]) / np.where(b[1] != a[1], b[1] - a[1], 1.0)
        inside ^= cross & (P[:, 0] < xint)
    return np.where(inside, -best, best), feat


# ----------------------------------------------------------------------------------------------------------------- reference
def fourier(V, N=384, eta=1e6, tol=1e-10, shift=0.26):
    """Stokes flow in the unit cell around the polygon V (centred at (1/2, 1/2)), unit body force along x on the fluid only, nu = 1, volume penalisation with a smoothed indicator (width 1.5 / N), Fourier-CG.
    `shift` (grid cells) moves the smoothed indicator outward: the penalised flow vanishes 0.26 cells INSIDE the transition (measured: the zero of u extrapolated along a face normal, square, N = 576), which near the
    wall is not negligible for the operator check (the closure reads the profile curvature from w(d) / d^2 at d = dx / 2: a 0.021 dx wall offset moves u by 4 % there); with the shift the zero lies on the polygon.
    Returns dict(U [2] superficial velocity, F [2] force on the solid per cell, u [2, N, N] velocity, lap [2, N, N] laplacian of u (4th-order differences), lapErr, c) (cached)."""
    key = hashlib.sha1(np.round(V, 12).tobytes() + str((N, eta, tol, "fd4", shift)).encode()).hexdigest()[:16]
    d = paths.tmp_dir()
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"corner_ref_{key}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    sd, _ = signed_distance(np.stack([X.ravel() - 0.5, Y.ravel() - 0.5], 1), V)
    chi = (0.5 * (1 - np.tanh((sd - shift / N) / (0.35 * 1.5 / N)))).reshape(N, N)
    k = 2 * np.pi * np.fft.fftfreq(N, 1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    K2s = K2.copy()
    K2s[0, 0] = 1.0

    def proj(fx, fy):
        dd = (KX * fx + KY * fy) / K2s
        return fx - KX * dd, fy - KY * dd

    def A(u):
        fx, fy = np.fft.fft2(u[0]), np.fft.fft2(u[1])
        lx = np.real(np.fft.ifft2(K2 * fx)) + eta * chi * u[0]
        ly = np.real(np.fft.ifft2(K2 * fy)) + eta * chi * u[1]
        px, py = proj(np.fft.fft2(lx), np.fft.fft2(ly))
        return np.real(np.fft.ifft2(px)), np.real(np.fft.ifft2(py))

    def prec(u):
        s = 1.0 / (K2 + eta * chi.mean() + 1e-30)
        return tuple(np.real(np.fft.ifft2(s * np.fft.fft2(q))) for q in u)

    bx, by = proj(np.fft.fft2(1 - chi), np.zeros((N, N)))
    b = (np.real(np.fft.ifft2(bx)), np.real(np.fft.ifft2(by)))
    u = (np.zeros((N, N)), np.zeros((N, N)))
    r_ = b
    z = prec(r_)
    p = z
    rz = sum((q * bb).sum() for q, bb in zip(r_, z))
    b2 = sum((q * q).sum() for q in b)
    for it in range(40000):
        Ap = A(p)
        al = rz / sum((x_ * y_).sum() for x_, y_ in zip(p, Ap))
        u = (u[0] + al * p[0], u[1] + al * p[1])
        r_ = (r_[0] - al * Ap[0], r_[1] - al * Ap[1])
        if sum((x_ * x_).sum() for x_ in r_) < tol ** 2 * b2:
            break
        z = prec(r_)
        rz2 = sum((x_ * y_).sum() for x_, y_ in zip(r_, z))
        p = (z[0] + rz2 / rz * p[0], z[1] + rz2 / rz * p[1])
        rz = rz2
    h = 1.0 / N
    # local finite-difference Laplacians (the spectral one rings over several cells from the kink of u at the wall: -0.81 instead of -1.11 half a particle spacing from a face at N = 576);
    # 4th order is the reference, |4th - 2nd| its local uncertainty
    lap2 = np.stack([(np.roll(q, 1, 0) + np.roll(q, -1, 0) + np.roll(q, 1, 1) + np.roll(q, -1, 1) - 4 * q) / h ** 2 for q in u])
    lap = np.stack([(-(np.roll(q, 2, 0) + np.roll(q, -2, 0) + np.roll(q, 2, 1) + np.roll(q, -2, 1)) + 16 * (np.roll(q, 1, 0) + np.roll(q, -1, 0) + np.roll(q, 1, 1) + np.roll(q, -1, 1)) - 60 * q)
                    / (12 * h ** 2) for q in u])
    out = dict(U=np.array([u[0].mean(), u[1].mean()]), F=np.array([(eta * chi * u[0]).mean(), (eta * chi * u[1]).mean()]), u=np.stack(u), lap=lap, lapErr=np.abs(lap - lap2), c=np.array(chi.mean()),
               iters=np.array(it))
    np.savez(path, **out)
    return out


def wall_offset(V, ref, face_mid, normal, dx):
    """zero of the reference velocity along a face normal (cubic fit over 0.3 ... 3 dx), in units of dx: 0 = the penalised wall lies on the polygon."""
    s = np.linspace(0.3, 3.0, 28) * dx
    P = 0.5 + np.asarray(face_mid)[None] + s[:, None] * np.asarray(normal)[None]
    t = np.array([-normal[1], normal[0]])
    u = sample(ref["u"][0], P) * t[0] + sample(ref["u"][1], P) * t[1]
    r = np.roots(np.polyfit(s, u, 3))
    r = r[np.isreal(r)].real
    return float(r[np.argmin(np.abs(r))] / dx)


def sample(field, P):
    """bilinear periodic interpolation of a cell-centred field [N, N] (unit cell) at points P [M, 2]."""
    N = field.shape[0]
    g = P * N - 0.5
    i0 = np.floor(g).astype(int)
    a = g - i0
    i0 %= N
    i1 = (i0 + 1) % N
    f = field
    return (f[i0[:, 0], i0[:, 1]] * (1 - a[:, 0]) * (1 - a[:, 1]) + f[i1[:, 0], i0[:, 1]] * a[:, 0] * (1 - a[:, 1]) + f[i0[:, 0], i1[:, 1]] * (1 - a[:, 0]) * a[:, 1]
            + f[i1[:, 0], i1[:, 1]] * a[:, 0] * a[:, 1])


def offset_curve(V, s, ds):
    """points at arc spacing ~ds on the outward offset of the CONVEX counter-clockwise polygon V at distance s: the edges shifted by s n plus circular arcs of radius s around the vertices."""
    n = len(V)
    pieces = []
    for k in range(n):
        a, b = V[k], V[(k + 1) % n]
        t = (b - a) / np.linalg.norm(b - a)
        nk = np.array([t[1], -t[0]])                                                       # outward normal of a counter-clockwise polygon
        pieces.append(("seg", a + s * nk, b + s * nk))
        c = V[(k + 1) % n]                                                                 # the arc around the next vertex, from this edge's normal to the next edge's normal
        t2 = (V[(k + 2) % n] - c) / np.linalg.norm(V[(k + 2) % n] - c)
        n2 = np.array([t2[1], -t2[0]])
        a0, a1 = math.atan2(nk[1], nk[0]), math.atan2(n2[1], n2[0])
        if a1 < a0:
            a1 += 2 * math.pi
        pieces.append(("arc", c, (a0, a1)))
    lens = [np.linalg.norm(p[2] - p[1]) if p[0] == "seg" else s * (p[2][1] - p[2][0]) for p in pieces]
    L = float(sum(lens))
    m = max(int(round(L / ds)), 3)
    out = []
    for u in (np.arange(m) + 0.5) * L / m:
        for p, l in zip(pieces, lens):
            if u <= l or p is pieces[-1]:
                f = min(u / l, 1.0) if l > 0 else 0.0
                if p[0] == "seg":
                    out.append(p[1] + f * (p[2] - p[1]))
                else:
                    ang = p[2][0] + f * (p[2][1] - p[2][0])
                    out.append(p[1] + s * np.array([math.cos(ang), math.sin(ang)]))
                break
            u -= l
    return np.array(out)


def layered(V, n, layers):
    """body-fitted sampling: `layers` offset curves at (k + 1/2) dx (spacing dx along each), the lattice beyond layers * dx; the seam between the two lies outside the support of every particle that sees the wall
    when layers * dx > H.  Convex polygons only."""
    dx = 1.0 / n
    pts = [offset_curve(V, (k + 0.5) * dx, dx) for k in range(layers)]
    X, Y = np.meshgrid(dx * (np.arange(n) + .5), dx * (np.arange(n) + .5), indexing="ij")
    lat = np.stack([X.ravel(), Y.ravel()], 1)
    sd, _ = signed_distance(lat - 0.5, V)
    lat = lat[sd >= layers * dx]
    return np.concatenate([np.concatenate(pts) + 0.5, lat])


# ----------------------------------------------------------------------------------------------------------------- SPH set-up
def setup(a, V, f, consistent=False):
    dx = 1.0 / a.n
    H = a.H * dx
    if a.layers > 0:
        pos = layered(V, a.n, a.layers)                                                 # body-fitted layers near the wall, the lattice beyond
    else:
        X, Y = np.meshgrid(dx * (np.arange(a.n) + .5), dx * (np.arange(a.n) + .5), indexing="ij")
        pos = np.stack([X.ravel(), Y.ravel()], 1)
        sd, _ = signed_distance(pos - 0.5, V)
        pos = pos[sd >= 0.5 * dx]                                                       # the cut lattice, first row half a spacing from the wall
    alpha = a.nu * 8 * XI / (a.c0 * H)                                                  # nominal nu = alpha c0 H / (8 xi) (Morris: calibrated to it; alpha form: the angular mean)
    scene = Scene([Body(bodyId=0, center=(0.5, 0.5), reps=[SurfaceRep.polygon(V + 0.0)])], a.device)
    cfg = DeltaSPHConfig(gravity=(0, 0), c0=a.c0, alpha=alpha, periodic=Periodic((0, 0), (1, 1)), bodyForce=(f, 0.0), pressureConsistent=consistent, graphStep=not a.eager, shifting=True,
                         wallViscosityForm=a.wall, fluidViscosity=a.visc, morrisCalibration=a.cal, cornerWedgeTables=a.wedge, complementMoments=a.complement, wallPressureViscous=a.wpv)
    sim = DeltaSPH2D(pos, np.zeros_like(pos), np.ones(len(pos)), dx, scene, cfg, a.device, support=H)
    return sim, pos, dx, H


def classes(pos, V, H):
    """particle classes by the nearest feature: 0 interior (farther than H from the wall), 1 face, 2 near-vertex face (within H of a vertex whose angle is not 180 deg), 3 vertex quadrant."""
    sd, feat = signed_distance(pos - 0.5, V)
    n = len(V)
    dv = np.min(np.linalg.norm((pos - 0.5)[:, None, :] - V[None], axis=2), axis=1)
    cl = np.zeros(len(pos), dtype=int)
    near = sd < H
    cl[near & (feat >= n)] = 1
    cl[near & (feat >= n) & (dv < H)] = 2
    cl[near & (feat < n)] = 3
    return cl


def operator(a):
    V = polygon(a.shape, a.c, a.beta, a.rot, sides=a.sides)
    ref = fourier(V, a.N)
    sim, pos, dx, H = setup(a, V, 0.0)
    nu = a.nu
    # reference: unit force, nu = 1  ->  u_ref (nu = 1); for our nu and f = 1: u = u_ref / nu, nu lap u = lap_ref; scale the field so that the speeds are O(0.05)
    s = 0.05 / max(np.abs(ref["U"]).max(), 1e-30)
    v = np.stack([sample(ref["u"][0], pos), sample(ref["u"][1], pos)], 1) * s
    exact = np.stack([sample(ref["lap"][0], pos), sample(ref["lap"][1], pos)], 1) * s * nu      # nu lap u of the scaled field = s nu lap u_ref
    uncert = np.hypot(sample(ref["lapErr"][0], pos), sample(ref["lapErr"][1], pos)) * s * nu
    fscale = s * nu                                                                             # the body force that drives the scaled field (f = nu / nu_ref * s with nu_ref = 1)
    sim.v = torch.as_tensor(v, dtype=torch.float64, device=sim.dev)
    acc, _, _ = sim.rhs(sim.x, sim.v, sim.rho)
    acc = acc.cpu().numpy()
    # sampling quality: partition-of-unity deficit 1 - (sum_j V_j W_ij + lambda_wall,i); the wall closure assumes the fluid side of the wall is sampled by particles of volume dx^2 (a cut lattice leaves a gap)
    from scipy.spatial import cKDTree
    tree = cKDTree(pos, boxsize=1.0)
    pairs = tree.query_pairs(H, output_type="ndarray")
    rr = np.linalg.norm(((pos[pairs[:, 0]] - pos[pairs[:, 1]]) + 0.5) % 1.0 - 0.5, axis=1)
    Wv = sim.W(torch.as_tensor(rr), H).numpy() * dx * dx
    sumW = np.full(len(pos), float(sim.W(torch.zeros(1, dtype=torch.float64), H)[0]) * dx * dx)
    np.add.at(sumW, pairs[:, 0], Wv)
    np.add.at(sumW, pairs[:, 1], Wv)
    lam = sim._surface_state(sim.x, sim.rho)["lam"].sum(0).cpu().numpy()
    pu = 1.0 - (sumW + lam)
    sd, _ = signed_distance(pos - 0.5, V)
    cl = classes(pos, V, H)
    err = np.linalg.norm(acc - exact, axis=1) / fscale
    names = ["interior", "face", "near-vertex face", "vertex quadrant"]
    row = np.round(sd / dx - 0.5).astype(int)
    seam = (sd > (a.layers - 1) * dx) & (sd < (a.layers + 1) * dx) if a.layers else np.zeros(len(pos), dtype=bool)
    if a.layers:
        print(f"  seam of layers / lattice (excluded from interior): n={int(seam.sum())}  err {np.median(err[seam]):.4f}  PU deficit {np.median(pu[seam]):+.4f}")
    cl = np.where(seam & (cl == 0), -1, cl)
    mag = np.linalg.norm(exact, axis=1) / fscale
    print(f"{a.shape} beta={a.beta:g} rot={a.rot:g} c={a.c:g} n={a.n} H/dx={a.H:g} {a.visc}/{a.wall}{' complement' if a.complement else ''}{' wedge' if a.wedge else ''} {'layers ' + str(a.layers) if a.layers else 'cut lattice'}: medians per class and wall row, in units of the body force f "
          f"(|exact| = |nu lap u|, err = |a_sph - nu lap u|, ref unc = |FD4 - FD2| of the reference)")
    print(f"  {'interior':17s} n={int((cl == 0).sum()):5d}  err {np.median(err[cl == 0]):7.4f}  (90 % {np.percentile(err[cl == 0], 90):.4f})  PU deficit {np.median(pu[cl == 0]):+.4f}")
    for k in (1, 2, 3):
        for r in range(4):
            m = (cl == k) & (row == r)
            if m.any():
                print(f"  {names[k]:17s} row {r}: n={int(m.sum()):3d}  |exact| {np.median(mag[m]):7.3f}  err {np.median(err[m]):7.3f}  rel {np.median(err[m] / mag[m].clip(1e-9)):6.3f}  ref unc {np.median(uncert[m] / fscale):6.3f}"
                      f"  PU deficit {np.median(pu[m]):+.4f}  d/dx {np.median(sd[m]) / dx:.2f}")
    np.savez(os.path.join(paths.tmp_dir(), f"corner_operator_{a.shape}_b{a.beta:g}_r{a.rot:g}_n{a.n}_{a.visc}.npz"), pos=pos, acc=acc, exact=exact, cl=cl, err=err, V=V, row=row, uncert=uncert)


def array(a):
    V = polygon(a.shape, a.c, a.beta, a.rot, sides=a.sides)
    ref = fourier(V, a.N)
    sim, pos, dx, H = setup(a, V, a.f, a.consistent)
    c = float(ref["c"])
    Uref = ref["U"] * a.f / a.nu                                                         # superficial velocity for our f and nu (linear)
    hist = LoadHistory()
    t0, k = time.time(), 0
    while sim.time < a.time:
        sim.step()
        hist.record(sim)
        k += 1
    U = sim.v.sum(0).cpu().numpy() * dx * dx
    F = hist.total()[-500:, :2].mean(0)
    Fbal = sim.cfg.rho0 * a.f * len(pos) * dx * dx
    K, Kref = F[0] / (a.nu * U[0]), (1.0 - c) / float(ref["U"][0])                          # reference: f = nu = 1, force on the fluid only
    print(f"{a.shape} beta={a.beta:g} rot={a.rot:g} c={c:.4f} n={a.n} H/dx={a.H:g} {a.visc}/{a.wall}{' consistent' if a.consistent else ''}: U = ({U[0]:.5f}, {U[1]:+.5f}) ref ({Uref[0]:.5f}, {Uref[1]:+.5f})  "
          f"U_x / ref = {U[0] / Uref[0]:.4f}  U_y / U_x = {U[1] / U[0]:+.4f} (ref {Uref[1] / Uref[0]:+.4f})  K / K_ref = {K / Kref:.4f}  F / balance = {F[0] / Fbal:.4f}  "
          f"fill N dx^2 / (1 - c) = {len(pos) * dx * dx / (1 - c):.4f}  ({time.time() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("operator", "array", "reference"))
    ap.add_argument("--shape", default="triangle", choices=("square", "triangle", "rhombus", "ngon"))
    ap.add_argument("--sides", type=int, default=64, help="ngon: number of vertices (64: the corner-free baseline)")
    ap.add_argument("--beta", type=float, default=30.0, help="apex / edge angle in degrees (triangle, rhombus)")
    ap.add_argument("--rot", type=float, default=0.0, help="rotation in degrees (counter-clockwise)")
    ap.add_argument("--c", type=float, default=0.08, help="solid fraction")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--H", type=float, default=4.0)
    ap.add_argument("--N", type=int, default=576, help="Fourier reference resolution (576: K within ~0.3 % of the extrapolated value for the square and the 30 deg triangle)")
    ap.add_argument("--nu", type=float, default=0.0185)
    ap.add_argument("--c0", type=float, default=10.0)
    ap.add_argument("--f", type=float, default=0.03)
    ap.add_argument("--time", type=float, default=40.0)
    ap.add_argument("--wall", default="noslipMoment")
    ap.add_argument("--visc", default="morris", choices=("alpha", "morris"))
    ap.add_argument("--cal", type=float, default=0.985)
    ap.add_argument("--consistent", action="store_true")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--wedge", action="store_true", help="cfg.cornerWedgeTables (wedge moment tables at the corners)")
    ap.add_argument("--complement", action="store_true", help="cfg.complementMoments (discrete-complement wall moments)")
    ap.add_argument("--wpv", action="store_true", help="cfg.wallPressureViscous (the viscous term in the wall pressure condition)")
    ap.add_argument("--layers", type=int, default=0, help="body-fitted sampling: this many offset-curve layers at (k + 1/2) dx (0: the cut lattice); > H / dx keeps the seam out of the wall particles' support")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.mode == "reference":
        V = polygon(a.shape, a.c, a.beta, a.rot, sides=a.sides)
        for N in (192, 384, 576):
            r = fourier(V, N)
            print(f"{a.shape} beta={a.beta:g} rot={a.rot:g} N={N}: U = ({r['U'][0]:.6f}, {r['U'][1]:+.6f})  K_fluidonly = {(1 - float(r['c'])) / r['U'][0]:.4f}  c = {float(r['c']):.4f}  iters {int(r['iters'])}", flush=True)
    elif a.mode == "operator":
        operator(a)
    else:
        array(a)


if __name__ == "__main__":
    main()
