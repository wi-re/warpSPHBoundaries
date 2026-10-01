"""Scene layer: bodies with pose + OBB, surface / volume / implicit / SDF representations, per-type adjacency and operations."""
import numpy as np
import pytest
import torch
import warp as wp

from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, ParticleState, WarpOperation
from edgebound import scene as S
from edgebound import warpbc
from edgebound.implicitBodies import DiskBody, HalfPlaneBody
from edgebound.scene import Body, BodyField, ImplicitRep, Scene, SdfRep, SurfaceRep, VolumeRep, sceneOperation

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
    return Scene([a], dev), Scene([b], dev)


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
    from edgebound.kernels import kernel as K
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
    adj = Scene(bodies, device).buildAdjacency(ps, pr)
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
