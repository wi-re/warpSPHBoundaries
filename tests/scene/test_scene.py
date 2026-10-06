"""Scene layer: bodies with pose + OBB, surface / volume / implicit / SDF representations, per-type adjacency and operations."""
import numpy as np
import pytest
import torch
import warp as wp

from warpSPHCore import GradientScheme, KernelFunctions, OperationProperties, ParticleState, WarpOperation
from warpSPHBoundaries.scene.implicitBodies import DiskBody, HalfPlaneBody
from warpSPHBoundaries.scene.scene import Body, BodyField, ImplicitRep, Scene, SdfRep, SurfaceRep, VolumeRep, sceneOperation

DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
TD = torch.float64
LSHAPE = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
LTRI = np.array([[0, 1, 4], [0, 4, 5], [1, 2, 3], [1, 3, 4]])          # not a partition by itself, replaced below
# the L as 3 unit squares (6 triangles) on a shared vertex set
LV = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2]], dtype=float)
LE = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [3, 4, 7], [3, 7, 6]])


def props(op, kernel=KernelFunctions.Wendland4, mode=GradientScheme.Naive):
    return OperationProperties(kernel=kernel, operation=op, gradientMode=mode)


def state(pos, dev, sup=1.0, rho=None, mass=None, seed=0):
    n = len(pos)
    rng = np.random.default_rng(seed)
    t = lambda a: torch.as_tensor(a, dtype=TD, device=dev)
    return ParticleState(positions=t(pos), supports=t(np.full(n, sup) if np.isscalar(sup) else sup), masses=t(np.ones(n) if mass is None else mass),
                         kinds=torch.zeros(n, dtype=torch.int32, device=dev), densities=t(np.ones(n) if rho is None else rho))


def body_pair(dev, center=(0.3, -0.2), angle=0.7, v=(0.4, -0.1), w=0.9):
    kw = dict(center=center, angle=angle, linearVelocity=v, angularVelocity=w)
    a = Body(bodyId=0, reps=[SurfaceRep.polygon(LSHAPE)], **kw)
    b = Body(bodyId=0, reps=[VolumeRep(LV, LE)], **kw)
    return Scene([a], dev), Scene([b], dev, volumeMode="nodal")


def particles(dev, n=60, seed=1, sup=0.9):
    rng = np.random.default_rng(seed)
    # world positions around the (moved, rotated) L, some inside the solid
    pos = rng.uniform(-2.2, 2.4, (n, 2))
    return state(pos, dev, sup=rng.uniform(0.5, sup, n), rho=rng.uniform(0.9, 1.1, n), mass=rng.uniform(0.5, 1.5, n))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("op", [WarpOperation.Density, WarpOperation.Interpolate, WarpOperation.Gradient, WarpOperation.Divergence, WarpOperation.Curl])
def test_surface_matches_volume_for_rigid_fields(device, op):
    sa, sb = body_pair(device)
    ps = particles(device)
    fld = [BodyField.rigid(sa.bodies[0], rho=1.3)]
    fldv = [BodyField.rigid(sb.bodies[0], rho=1.3)]
    pr = props(op)
    if op in (WarpOperation.Gradient,):
        # vector field gradient (2x2) uses the rigid velocity
        pass
    oa = sceneOperation(ps, pr, sa, bodyFields=fld)
    ob = sceneOperation(ps, pr, sb, bodyFields=fldv)
    assert oa.shape == ob.shape
    np.testing.assert_allclose(oa.cpu().numpy(), ob.cpu().numpy(), atol=2e-10 * max(1, float(ob.abs().max())), rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", [GradientScheme.Naive, GradientScheme.Symmetric, GradientScheme.Difference, GradientScheme.Summation])
def test_gradient_modes_scalar_linear_field(device, mode):
    sa, sb = body_pair(device)
    ps = particles(device)
    rng = np.random.default_rng(5)
    qv = torch.as_tensor(rng.normal(size=len(ps.positions)), dtype=TD, device=device)
    f = BodyField(torch.tensor(0.7, dtype=TD, device=device), torch.tensor([0.3, -0.5], dtype=TD, device=device), rho=1.2)
    pr = props(WarpOperation.Gradient, mode=mode)
    oa = sceneOperation(ps, pr, sa, queryValues=qv, bodyFields=[f])
    ob = sceneOperation(ps, pr, sb, queryValues=qv, bodyFields=[f])
    np.testing.assert_allclose(oa.cpu().numpy(), ob.cpu().numpy(), atol=3e-10 * max(1, float(ob.abs().max())), rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_reaction_conservation_and_torque_vs_quadrature(device):
    sa, sb = body_pair(device)
    ps = particles(device, n=40)
    f = BodyField(torch.tensor(1.7, dtype=TD, device=device), None, rho=1.0)
    pr = props(WarpOperation.Gradient)
    out, rea = sceneOperation(ps, pr, sa, bodyFields=[f], returnReaction=True)
    m = ps.masses
    np.testing.assert_allclose(rea.force.cpu().numpy()[0], -(m[:, None] * out).sum(0).cpu().numpy(), atol=1e-12)
    assert rea.torqueExact == [True]
    # torque by dense quadrature of  -m_i A  int (x'-c) x grad_x W dA'   over the world-frame polygon (fan of the 3 unit squares, Gauss 12x12)
    body = sa.bodies[0]
    X = body.pose.toWorld(torch.as_tensor(LV, dtype=TD, device=device)).cpu().numpy()
    gx, gw = np.polynomial.legendre.leggauss(14)
    gx, gw = (gx + 1) / 2, gw / 2
    pts, wts = [], []
    for e in LE:
        a, b, c = X[e]
        area = 0.5 * abs((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0])
        for i, u in enumerate(gx):
            for j, v in enumerate(gx):
                l1, l2 = u, v * (1 - u)
                pts.append(a + l1 * (b - a) + l2 * (c - a))
                wts.append(gw[i] * gw[j] * (1 - u) * 2 * area)
    pts, wts = np.array(pts), np.array(wts)
    cen = body.center.cpu().numpy()
    tau = 0.0
    for i in range(len(ps.positions)):
        x = ps.positions[i].cpu().numpy()
        h = float(ps.supports[i])
        y = pts - x
        r = np.linalg.norm(y, axis=1)
        q = r / h
        # w4: W = 9/pi (1-q)^6 (1+6q+35/3 q^2) / h^2 ; dW/dr
        W = lambda qq: 9 / np.pi * (1 - qq) ** 6 * (1 + 6 * qq + 35 / 3 * qq ** 2) * (qq < 1) / h ** 2
        dW = (-9 / np.pi * (6 * (1 - q) ** 5 * (1 + 6 * q + 35 / 3 * q ** 2) - (1 - q) ** 6 * (6 + 70 / 3 * q)) * (q < 1)) / h ** 3
        gradx = -(dW / np.maximum(r, 1e-300))[:, None] * y            # grad_x W(x - x') = -W'(r) y / r, y = x' - x
        arm = pts - cen
        tau += -float(ps.masses[i]) * 1.7 * np.sum(wts * (arm[:, 0] * gradx[:, 1] - arm[:, 1] * gradx[:, 0]))
    # kinks of the kernel at the support radius limit the quadrature accuracy
    assert abs(float(rea.torque[0]) - tau) < 5e-4 * max(1.0, abs(tau))


@pytest.mark.parametrize("device", DEVICES)
def test_broadphase_is_exact_and_poses_move_without_rebuilds(device):
    sa, _ = body_pair(device)
    body = sa.bodies[0]
    ps = particles(device, n=200, seed=4)
    allowed = torch.ones(200, dtype=torch.bool, device=device)
    cand, lpos = sa.candidates(body, ps.positions, ps.supports, allowed)
    # brute force: distance from the local point to the OBB
    lo, hi = body.obb()
    L = body.pose.toLocal(ps.positions)
    d = ((lo - L).clamp(min=0) + (L - hi).clamp(min=0)).norm(dim=1)
    ref = torch.nonzero(d < ps.supports).flatten()
    assert torch.equal(cand, ref)
    # moving the body: the rep's acceleration structures are reused (identity), results equal a freshly constructed body at the new pose
    pr = props(WarpOperation.Density)
    sceneOperation(ps, pr, sa)
    rep = body.reps[0]
    cl = rep._cl
    body.move(0.37)
    out_moved = sceneOperation(ps, pr, sa)
    assert rep._cl is cl
    fresh = Scene([Body(bodyId=0, center=tuple(body.center.cpu().numpy()), angle=float(body.angle), reps=[SurfaceRep.polygon(LSHAPE)])], device)
    out_fresh = sceneOperation(ps, pr, fresh)
    np.testing.assert_allclose(out_moved.cpu().numpy(), out_fresh.cpu().numpy(), atol=1e-13)


@pytest.mark.parametrize("device", DEVICES)
def test_rotation_covariance(device):
    """rotating bodies AND particles together by an angle rotates the vector outputs and leaves scalar outputs unchanged."""
    th = 0.9
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pos = particles(device, n=50, seed=8).positions.cpu().numpy()
    sup = np.linspace(0.5, 0.9, 50)
    f0 = BodyField(torch.tensor(0.9, dtype=TD, device=device), None)
    outs = []
    for angle, center, P in [(0.0, np.array([0.1, 0.2]), pos), (th, Rm @ np.array([0.1, 0.2]), pos @ Rm.T)]:
        sc = Scene([Body(center=tuple(center), angle=angle, reps=[SurfaceRep.polygon(LSHAPE)])], device)
        ps = state(P, device, sup=sup)
        outs.append((sceneOperation(ps, props(WarpOperation.Density), sc).cpu().numpy(),
                     sceneOperation(ps, props(WarpOperation.Gradient), sc, bodyFields=[f0]).cpu().numpy()))
    np.testing.assert_allclose(outs[0][0], outs[1][0], atol=1e-12)
    np.testing.assert_allclose(outs[0][1] @ Rm.T, outs[1][1], atol=1e-11)


@pytest.mark.parametrize("device", DEVICES)
def test_winding_indicator(device):
    rng = np.random.default_rng(2)
    rep = SurfaceRep.polygon(LSHAPE).to(device)
    p = torch.as_tensor(rng.uniform(-1, 3, (500, 2)), dtype=TD, device=device)
    ind = rep.indicator(p).cpu().numpy()
    x, y = p[:, 0].cpu().numpy(), p[:, 1].cpu().numpy()
    ref = (((x > 0) & (x < 2) & (y > 0) & (y < 1)) | ((x > 0) & (x < 1) & (y > 0) & (y < 2))).astype(float)
    np.testing.assert_array_equal(ind, ref)
    hole = SurfaceRep.polygon(LSHAPE, solid="outside").to(device)
    np.testing.assert_array_equal(hole.indicator(p).cpu().numpy(), 1 - ref)


# ----------------------------------------------------------------------------------------------------------------------- implicit / SDF
def exact_disk_lambda(center, R, pos, sup, device):
    """independent reference: a very fine area-preserving surface polygon of the disk (exact edge terms)."""
    sc = Scene([Body(center=(0, 0), reps=[SurfaceRep.regularPolygon(center, R, 4096)])], device)
    ps = state(pos, device, sup=sup)
    return sceneOperation(ps, props(WarpOperation.Density), sc).cpu().numpy()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("R,tol", [(4.0, 3e-4), (0.7, 2e-6), (0.1, 1e-5)])
def test_implicit_disk_hard_switch(device, R, tol):
    c = np.array([0.4, -0.3])
    ang = np.linspace(0, 2 * np.pi, 41)[:-1]
    d = np.tile([0.05, 0.3, 0.6], len(ang) // 3 + 1)[:len(ang)]
    pos = c + ((R + d) * np.stack([np.cos(ang), np.sin(ang)], 1).T).T
    body = Body(center=(1.0, 2.0), angle=0.5, reps=[ImplicitRep(DiskBody(tuple(c), R))])           # centre given in the LOCAL frame
    sc = Scene([body], device)
    world = body.pose.toWorld(torch.as_tensor(pos, dtype=TD, device=device)).cpu().numpy()
    ps = state(world, device, sup=1.0)
    out = sceneOperation(ps, props(WarpOperation.Density), sc).cpu().numpy()
    ref = exact_disk_lambda(c, R, pos, 1.0, device) if False else None
    sc2 = Scene([Body(center=(1.0, 2.0), angle=0.5, reps=[SurfaceRep.regularPolygon(c, R, 8192)])], device)
    ref = sceneOperation(ps, props(WarpOperation.Density), sc2).cpu().numpy()
    assert np.abs(out - ref).max() < tol


@pytest.mark.parametrize("device", DEVICES)
def test_sdf_disk_and_box_with_fallback(device):
    R, h = 3.0, 1.0
    c = np.array([0.0, 0.0])
    fn = lambda p: (p - torch.as_tensor(c, dtype=TD)).norm(dim=1) - R
    sdf = SdfRep.fromFunction(fn, (-R - 2, -R - 2), (R + 2, R + 2), h / 16)
    sc = Scene([Body(center=(0.5, 0.5), angle=0.3, reps=[sdf])], device)
    ang = np.linspace(0, 2 * np.pi, 31)[:-1]
    pos = (R + np.tile([0.05, 0.4, 0.7], 10)) * np.stack([np.cos(ang), np.sin(ang)], 1).T
    pos = pos.T
    body = sc.bodies[0]
    world = body.pose.toWorld(torch.as_tensor(pos, dtype=TD, device=device)).cpu().numpy()
    ps = state(world, device, sup=h)
    out = sceneOperation(ps, props(WarpOperation.Density), sc).cpu().numpy()
    ref = sceneOperation(ps, props(WarpOperation.Density),
                         Scene([Body(center=(0.5, 0.5), angle=0.3, reps=[SurfaceRep.regularPolygon(c, R, 8192)])], device)).cpu().numpy()
    assert np.abs(out - ref).max() < 5e-4                      # tier-3 model error at R = 3h plus the sampling error of the SDF
    # box with sharp corners: SDF is not smooth near them -> fallback surface elements (exact), tier 3 elsewhere
    box = SurfaceRep.box((-2, -1), (2, 1))
    def boxsdf(p):
        q = p.abs() - torch.tensor([2.0, 1.0], dtype=TD)
        return q.clamp(min=0).norm(dim=1) + q.max(dim=1).values.clamp(max=0)
    sdfb = SdfRep.fromFunction(boxsdf, (-5, -4), (5, 4), 1 / 32, fallback=box)
    rng = np.random.default_rng(11)
    P = rng.uniform(-3.2, 3.2, (300, 2))
    ps = state(P, device, sup=0.9)
    o = sceneOperation(ps, props(WarpOperation.Density), Scene([Body(reps=[sdfb])], device)).cpu().numpy()
    r = sceneOperation(ps, props(WarpOperation.Density), Scene([Body(reps=[box])], device)).cpu().numpy()
    # tier 3 (curvature 0 on the faces) is exact on flat parts; near corners the fallback is exact: the overall error is small
    assert np.abs(o - r).max() < 2e-3
    assert np.abs(o - r).mean() < 1e-4


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_types_accumulate_and_stats(device):
    """a scene with a surface body, a volume body, an implicit disk and a half plane: output = sum of the single-body outputs."""
    bodies = [Body(bodyId=0, center=(0, 0), reps=[SurfaceRep.polygon(LSHAPE)]),
              Body(bodyId=1, center=(4, 0), angle=0.4, reps=[VolumeRep(LV, LE)]),
              Body(bodyId=2, center=(0, 4), reps=[ImplicitRep(DiskBody((0, 0), 2.5))]),
              Body(bodyId=3, center=(0, 0), reps=[ImplicitRep(HalfPlaneBody((0, -2.0), (0.0, 1.0)))])]
    ps = particles(device, n=120, seed=6, sup=0.9)
    P = ps.positions.cpu().numpy() * 2.5
    ps = state(P, device, sup=np.linspace(0.5, 0.9, 120))
    pr = props(WarpOperation.Density)
    total = sceneOperation(ps, pr, Scene(bodies, device))
    parts = sum(sceneOperation(ps, pr, Scene([b], device)) for b in bodies)
    np.testing.assert_allclose(total.cpu().numpy(), parts.cpu().numpy(), atol=1e-13)
    adj = Scene(bodies, device).pairMoments(ps, pr)
    assert len(adj.stats["candidates"]) == 4 and adj.stats["pairs"] > 0


@pytest.mark.parametrize("device", DEVICES)
def test_per_query_fields_equal_the_body_field_sampled_at_the_queries(device):
    """BodyField(perQuery) with a0_i = A(x_i) and the same gradient reproduces the constant-body linear field exactly (surface and implicit paths)."""
    ps = particles(device, n=60, seed=9)
    pos = ps.positions
    for scene in [Scene([Body(center=(0.2, 0.1), angle=0.4, reps=[SurfaceRep.polygon(LSHAPE)])], device)]:
        body = scene.bodies[0]
        a0 = torch.tensor(0.7, dtype=TD, device=device)
        a1 = torch.tensor([0.3, -0.5], dtype=TD, device=device)
        ref = sceneOperation(ps, props(WarpOperation.Gradient), scene, bodyFields=[BodyField(a0, a1)])
        Ax = a0 + (pos - body.center) @ a1
        got = sceneOperation(ps, props(WarpOperation.Gradient), scene, bodyFields=[BodyField(Ax, a1, perQuery=True)])
        np.testing.assert_allclose(got.cpu().numpy(), ref.cpu().numpy(), atol=1e-12)
        # symmetric mode with queryValues as well
        qv = torch.as_tensor(np.random.default_rng(3).normal(size=60), dtype=TD, device=device)
        pr = props(WarpOperation.Gradient, mode=GradientScheme.Symmetric)
        ref = sceneOperation(ps, pr, scene, queryValues=qv, bodyFields=[BodyField(a0, a1, rho=1.1)])
        got = sceneOperation(ps, pr, scene, queryValues=qv, bodyFields=[BodyField(Ax, a1, rho=1.1, perQuery=True)])
        np.testing.assert_allclose(got.cpu().numpy(), ref.cpu().numpy(), atol=1e-12)
    # implicit disk, constant per-query field (Neumann-type wall value p_b = p_i)
    sc = Scene([Body(center=(0.0, 0.0), reps=[ImplicitRep(DiskBody((0.0, 0.0), 3.0))])], device)
    pr = props(WarpOperation.Gradient)
    P = torch.as_tensor(np.random.default_rng(2).uniform(-4, 4, (50, 2)), dtype=TD, device=device)
    ps2 = state(P.cpu().numpy(), device, sup=1.0)
    pq = torch.as_tensor(np.random.default_rng(4).normal(size=50), dtype=TD, device=device)
    got = sceneOperation(ps2, pr, sc, bodyFields=[BodyField(pq, None, perQuery=True)])
    unit = sceneOperation(ps2, pr, sc, bodyFields=[BodyField(torch.tensor(1.0, dtype=TD, device=device), None)])
    np.testing.assert_allclose(got.cpu().numpy(), (pq[:, None] * unit).cpu().numpy(), atol=1e-13)


@pytest.mark.parametrize("device", DEVICES)
def test_volume_moments_mode_equals_nodal_mode_and_surface(device):
    """volume representation as exact-moment pair sets (default) = nodal P1 path = surface loops; the torque is now exact for volumes too."""
    sa, sb = body_pair(device)
    sm = Scene([Body(bodyId=0, center=(0.3, -0.2), angle=0.7, linearVelocity=(0.4, -0.1), angularVelocity=0.9, reps=[VolumeRep(LV, LE)])], device)
    ps = particles(device)
    fld = [BodyField.rigid(sa.bodies[0], rho=1.3)]
    for op in [WarpOperation.Density, WarpOperation.Interpolate, WarpOperation.Gradient, WarpOperation.Divergence, WarpOperation.Curl]:
        oa = sceneOperation(ps, props(op), sa, bodyFields=fld)
        om = sceneOperation(ps, props(op), sm, bodyFields=fld)
        np.testing.assert_allclose(om.cpu().numpy(), oa.cpu().numpy(), atol=2e-10 * max(1, float(oa.abs().max())), rtol=0)
    f = BodyField(torch.tensor(1.7, dtype=TD, device=device), None)
    _, ra = sceneOperation(ps, props(WarpOperation.Gradient), sa, bodyFields=[f], returnReaction=True)
    _, rm = sceneOperation(ps, props(WarpOperation.Gradient), sm, bodyFields=[f], returnReaction=True)
    np.testing.assert_allclose(rm.force.cpu().numpy(), ra.force.cpu().numpy(), atol=1e-10)
    np.testing.assert_allclose(rm.torque.cpu().numpy(), ra.torque.cpu().numpy(), atol=1e-9)
    assert rm.torqueExact == [True]


@pytest.mark.parametrize("device", DEVICES)
def test_per_query_gradient_field(device):
    """perQuery with a per-query gradient a1_i equals the body linear field when a1_i is the same for all queries."""
    ps = particles(device, n=40, seed=12)
    sc = Scene([Body(center=(0.2, 0.1), angle=0.4, reps=[SurfaceRep.polygon(LSHAPE)])], device)
    body = sc.bodies[0]
    a0 = torch.tensor(0.7, dtype=TD, device=device)
    a1 = torch.tensor([0.3, -0.5], dtype=TD, device=device)
    ref = sceneOperation(ps, props(WarpOperation.Gradient), sc, bodyFields=[BodyField(a0, a1)])
    Ax = a0 + (ps.positions - body.center) @ a1
    got = sceneOperation(ps, props(WarpOperation.Gradient), sc, bodyFields=[BodyField(Ax, a1[None].expand(40, 2).contiguous(), perQuery=True)])
    np.testing.assert_allclose(got.cpu().numpy(), ref.cpu().numpy(), atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_half_plane_first_moments_and_linear_fields(device):
    """tier-3 planar first moments: HalfPlane / flat-wall SDF with a linear boundary field equals the exact surface (box loop) result."""
    box = SurfaceRep.box((0, 0), (1, 1), solid="outside")
    h = 0.1
    rng = np.random.default_rng(3)
    P = np.concatenate([rng.uniform(0.01, 0.99, (200, 2))])
    # keep particles at least one support from the corners so the single wall model is exact (no second wall within h)
    cx = np.minimum(P[:, 0], 1 - P[:, 0]) > 1.01 * h
    P = P[cx & (np.minimum(P[:, 1], 1 - P[:, 1]) < h)]
    ps = state(P, device, sup=h, rho=np.ones(len(P)))
    a0 = torch.as_tensor(np.random.default_rng(1).normal(size=len(P)), dtype=TD, device=device)
    a1 = torch.tensor([0.0, -9.81], dtype=TD, device=device)
    fld = BodyField(a0, a1, perQuery=True)
    ref = sceneOperation(ps, props(WarpOperation.Gradient, mode=GradientScheme.Symmetric), Scene([Body(reps=[box])], device), queryValues=a0, bodyFields=[fld])
    # the wall seen as half planes / SDF of the box (only the bottom/top walls matter for the selected particles, side walls are > h away)
    hp = Scene([Body(bodyId=i, reps=[ImplicitRep(HalfPlaneBody(p_, n_))]) for i, (p_, n_) in enumerate(
        [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, -1.0)), ((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (-1.0, 0.0))])], device)
    got = sceneOperation(ps, props(WarpOperation.Gradient, mode=GradientScheme.Symmetric), hp, queryValues=a0, bodyFields=[fld] * 4)
    np.testing.assert_allclose(got.cpu().numpy(), ref.cpu().numpy(), atol=1e-6 * max(1, float(ref.abs().max())))
    def boxsdf(p):
        q = (p - 0.5).abs() - 0.5
        return -(q.clamp(min=0).norm(dim=1) + q.max(dim=1).values.clamp(max=0))
    sdf = SdfRep.fromFunction(boxsdf, (-0.3, -0.3), (1.3, 1.3), h / 16, fallback=box)
    got = sceneOperation(ps, props(WarpOperation.Gradient, mode=GradientScheme.Symmetric), Scene([Body(reps=[sdf])], device), queryValues=a0, bodyFields=[fld])
    np.testing.assert_allclose(got.cpu().numpy(), ref.cpu().numpy(), atol=2e-3 * max(1, float(ref.abs().max())))


@pytest.mark.parametrize("device", DEVICES)
def test_covariance_operation(device):
    """Covariance = int y (x) grad_x W (the renormalisation matrix of the wall): surface = volume = half-plane model; a particle deep inside a body sees the full-plane value I."""
    sa, sm = body_pair(device)
    sm = Scene([Body(bodyId=0, center=(0.3, -0.2), angle=0.7, reps=[VolumeRep(LV, LE)])], device)
    ps = particles(device)
    pr = props(WarpOperation.Covariance)
    ca = sceneOperation(ps, pr, sa)
    cm = sceneOperation(ps, pr, sm)
    assert ca.shape == (len(ps.positions), 2, 2)
    np.testing.assert_allclose(cm.cpu().numpy(), ca.cpu().numpy(), atol=2e-10)
    # deep inside the body (support entirely in the solid): C = I
    inside = Scene([Body(reps=[SurfaceRep.box((-1, -1), (1, 1))])], device)
    ps2 = state(np.array([[0.0, 0.0], [0.2, -0.1]]), device, sup=0.3)
    c = sceneOperation(ps2, pr, inside)
    np.testing.assert_allclose(c.cpu().numpy(), np.stack([np.eye(2)] * 2), atol=1e-12)
    # flat wall at distance d: C = lambda t(x)t + (lambda - q lambda') n(x)n  against the half-plane model
    hp = Scene([Body(reps=[ImplicitRep(HalfPlaneBody((0.0, 0.0), (0.0, 1.0)))])], device)
    sf = Scene([Body(reps=[SurfaceRep.box((-3, -3), (3, 0.0))])], device)
    ps3 = state(np.array([[0.0, 0.03], [0.1, 0.07], [-0.2, 0.12]]), device, sup=0.2)
    np.testing.assert_allclose(sceneOperation(ps3, pr, hp).cpu().numpy(), sceneOperation(ps3, pr, sf).cpu().numpy(), atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_scene_inside_agrees_across_representations(device):
    """point-in-solid: a tank wall as surface loop (background 1), as the omniSPH-style slab volume, as SDF, and a rotated hexagon body."""
    from warpSPHBoundaries.sim.dfsph2d import domain_scene
    lo, hi, h = (0.0, 0.0), (1.0, 0.5), 0.05
    rng = np.random.default_rng(1)
    pts = torch.as_tensor(rng.uniform([-0.2, -0.2], [1.2, 0.7], (4000, 2)), dtype=torch.float64, device=device)
    truth = ~((pts[:, 0] > 0) & (pts[:, 0] < 1) & (pts[:, 1] > 0) & (pts[:, 1] < 0.5))
    near = (((pts[:, 0] - 0).abs() < 0.01) | ((pts[:, 0] - 1).abs() < 0.01) | ((pts[:, 1] - 0).abs() < 0.01) | ((pts[:, 1] - 0.5).abs() < 0.01)) 
    inner = ~((pts[:, 0] < -0.12) | (pts[:, 0] > 1.12) | (pts[:, 1] < -0.12) | (pts[:, 1] > 0.62))      # the volume slabs have thickness 2.5 h = 0.125
    for kind in ("surface", "volume", "sdf"):
        sc = domain_scene(kind, lo, hi, h, device)
        got = sc.inside(pts)
        ok = ~near & (inner if kind != "surface" else torch.ones_like(inner))
        assert bool((got[ok] == truth[ok]).all()), kind
    hexa = Body(bodyId=1, center=(0.5, 0.25), angle=0.3, reps=[SurfaceRep.regularPolygon((0, 0), 0.1, 6, areaPreserving=False)])
    sc = Scene([hexa], device)
    c = torch.tensor([[0.5, 0.25], [0.5 + 0.08, 0.25], [0.5 + 0.2, 0.25]], dtype=torch.float64, device=device)
    assert sc.inside(c).tolist() == [True, True, False]


@pytest.mark.parametrize("device", DEVICES)
def test_signed_distance_and_normal(device):
    """tank (surface loop, solid outside), a rotated hexagon and the SDF box agree with the exact distances; the normal points from the wall into the fluid."""
    from warpSPHBoundaries.sim.dfsph2d import domain_scene
    lo, hi = (0.0, 0.0), (1.0, 0.5)
    pts = torch.tensor([[0.5, 0.05], [0.02, 0.3], [0.97, 0.45], [0.5, 0.49], [1.05, 0.2], [0.5, 0.25]], dtype=torch.float64, device=device)
    d_true = torch.tensor([0.05, 0.02, 0.03, 0.01, -0.05, 0.25], dtype=torch.float64, device=device)
    n_true = torch.tensor([[0, 1], [1, 0], [-1, 0], [0, -1], [-1, 0], [0, -1]], dtype=torch.float64, device=device)
    for kind in ("surface", "sdf"):
        sc = domain_scene(kind, lo, hi, 0.05, device)
        d, n, hit = sc.signed_distance(pts)
        assert bool(hit.all())
        tol = 1e-12 if kind == "surface" else 2e-3
        assert float((d - d_true).abs().max()) < tol, kind
        assert float((n[:5] - n_true[:5]).abs().max()) < (1e-12 if kind == "surface" else 0.05), kind
    hexa = Body(bodyId=1, center=(0.5, 0.25), angle=0.4, reps=[SurfaceRep.regularPolygon((0, 0), 0.1, 6, areaPreserving=False)])
    sc = Scene([hexa], device)
    q = torch.tensor([[0.5 + 0.2, 0.25], [0.5, 0.25 + 0.3], [0.5, 0.25]], dtype=torch.float64, device=device)
    d, n, hit = sc.signed_distance(q)
    assert d[2] < 0 and d[0] > 0.1 and d[1] > 0.19
    assert float((n[0] - torch.tensor([1.0, 0.0], dtype=torch.float64, device=device)).norm()) < 0.7       # roughly +x (rotated hexagon)


@pytest.mark.parametrize("device", DEVICES)
def test_adjacency_kernel_guard(device):
    """an adjacency holds the moments of its OWN kernel: sceneOperation must reject an adjacency built for a different kernel (before
    WORK-006 T6.2 the kernel in the operation's props was silently ignored when an adjacency was passed).  On a unit-square SurfaceRep
    scene with a few particles: an adjacency built for kernel 'cone' passed to a Density of kernel 'lw2' (registered via
    viscosity.lap_factor(1.0, 'w2')) raises ValueError (match 'built for kernel'); the same adjacency with kernel 'cone' works and equals
    the result without an adjacency (max|diff| <= 1e-12, same pairs, same arithmetic)."""
    from warpSPHBoundaries.scene.viscosity import lap_factor
    lap_factor(1.0, "w2")                                                    # registers the 'lw2' kernel (lazy registration)
    unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    sc = Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(unit)])], device)
    pts = np.array([[0.5, 0.3], [0.2, 0.8], [1.3, 0.4], [0.5, 1.2], [0.1, 0.1]])
    ps = state(pts, device, sup=0.4)
    adj = sc.pairMoments(ps, props(WarpOperation.Density, kernel="cone"))
    with pytest.raises(ValueError, match="moments are those of kernel"):
        sceneOperation(ps, props(WarpOperation.Density, kernel="lw2"), sc, moments=adj)
    got = sceneOperation(ps, props(WarpOperation.Density, kernel="cone"), sc, moments=adj)
    ref = sceneOperation(ps, props(WarpOperation.Density, kernel="cone"), sc)
    assert float((got - ref).abs().max()) <= 1e-12
