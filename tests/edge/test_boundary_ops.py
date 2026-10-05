"""warpSPH-style boundary operations (boundaryOps): all operation classes vs the numpy P1 weights, exact reproduction on covering meshes,
reaction conservation, directions, kernels, devices."""
from fractions import Fraction as F

import numpy as np
import pytest
import torch
import warp as wp

from warpSPHCore import GradientScheme, KernelFunctions, OperationDirection, OperationProperties, ParticleState, WarpOperation
from edgebound.scene import boundaryOps as B
from edgebound.edge import np_fem

DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
TD = torch.float64


def make_state(pos, dev, supports=None, kinds=None, rho=None, mass=None):
    n = len(pos)
    t = lambda a, dt=TD: torch.as_tensor(a, dtype=dt, device=dev)
    return ParticleState(positions=t(pos), supports=t(np.ones(n) if supports is None else supports), masses=t(np.ones(n) if mass is None else mass),
                         kinds=torch.as_tensor(np.zeros(n, dtype=np.int32) if kinds is None else kinds, dtype=torch.int32, device=dev),
                         densities=t(np.ones(n) if rho is None else rho))


def random_mesh(dev, ntri=6, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.uniform(-1, 1, (3 * ntri, 2))
    E = np.arange(3 * ntri).reshape(ntri, 3)
    return B.BoundaryMesh(torch.as_tensor(V, dtype=TD, device=dev), torch.as_tensor(E, dtype=torch.int32, device=dev)), V, E


def brute(V, E, pos, sup, kernel, fnodal=None):
    """independent reference: np_fem P1 weights for every (particle, element) pair, assembled by hand."""
    N, nE = len(pos), len(E)
    W = np.zeros((N, nE, 3))
    G = np.zeros((N, nE, 3, 2))
    for e in range(nE):
        tri = np.repeat(V[E[e]][None], N, axis=0)
        w, g = np_fem.weights(tri, pos, kernel, 1, sup, grad=True)
        W[:, e], G[:, e] = w, g
    return W, G


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kname", ["w2", "w4", "cubic", "quartic", "b8"])
def test_all_operations_match_the_numpy_weights(device, kname):
    kern = {"w2": KernelFunctions.Wendland2, "w4": KernelFunctions.Wendland4, "cubic": KernelFunctions.CubicSpline,
            "quartic": KernelFunctions.QuarticSpline, "b8": KernelFunctions.B8}[kname]
    mesh, V, E = random_mesh(device)
    rng = np.random.default_rng(1)
    N = 40
    pos = rng.uniform(-1.2, 1.2, (N, 2))
    sup = rng.uniform(0.6, 1.4, N)
    rho_i = rng.uniform(0.8, 1.2, N)
    ps = make_state(pos, device, sup, rho=rho_i)
    W, G = brute(V, E, pos, sup, kname)
    fnode = rng.normal(size=len(V))
    vnode = rng.normal(size=(len(V), 2))
    rho_b = rng.uniform(0.9, 1.1, len(V))
    fq = rng.normal(size=N)
    vq = rng.normal(size=(N, 2))
    Ae = fnode[E]                                    # [nE,3]
    adj = None
    for op in (WarpOperation.Density, WarpOperation.Interpolate, WarpOperation.Gradient, WarpOperation.Divergence, WarpOperation.Curl):
        for mode in ([GradientScheme.Naive, GradientScheme.Difference, GradientScheme.Summation, GradientScheme.Symmetric] if op in (WarpOperation.Gradient, WarpOperation.Divergence, WarpOperation.Curl) else [GradientScheme.Naive]):
            props = OperationProperties(kernel=kern, operation=op, gradientMode=mode, operationMode=OperationDirection.BoundaryToFluid)
            adj = adj or B.buildBoundaryAdjacency(ps, props, mesh)
            tq = lambda a: torch.as_tensor(a, dtype=TD, device=device)
            if op == WarpOperation.Density:
                out = B.boundaryOperation(ps, props, mesh, referenceDensities=tq(rho_b), adjacency=adj)
                ref = (W * rho_b[E][None]).sum((1, 2))
            elif op == WarpOperation.Interpolate:
                out = B.boundaryOperation(ps, props, mesh, referenceValues=tq(vnode), adjacency=adj)
                ref = np.einsum("nek,ekc->nc", W, vnode[E])
            else:
                A = fnode if op == WarpOperation.Gradient else vnode
                Aq = fq if op == WarpOperation.Gradient else vq
                out = B.boundaryOperation(ps, props, mesh, queryValues=tq(Aq), referenceValues=tq(A), referenceDensities=tq(rho_b), adjacency=adj)
                An = A[E]                                                         # [nE,3,...]
                fi = Aq[:, None, None] if op == WarpOperation.Gradient else Aq[:, None, None, :]
                An = An[None]
                if mode == GradientScheme.Naive:
                    a = An + 0 * fi
                elif mode == GradientScheme.Difference:
                    a = An - fi
                elif mode == GradientScheme.Summation:
                    a = An + fi
                else:
                    rb = rho_b[E][None]
                    ri = rho_i[:, None, None]
                    if op != WarpOperation.Gradient:
                        rb, ri = rb[..., None], ri[..., None]
                    a = rb * ri * (fi / ri ** 2 + An / rb ** 2)
                if op == WarpOperation.Gradient:
                    ref = np.einsum("nek,nekd->nd", a, G)
                elif op == WarpOperation.Divergence:
                    ref = np.einsum("nekc,nekc->n", a, G)
                else:
                    ref = (a[..., 1] * G[..., 0] - a[..., 0] * G[..., 1]).sum((1, 2))
            assert np.abs(out.cpu().numpy() - ref).max() < 1e-10 * (1 + np.abs(ref).max()), (op, mode, kname)


@pytest.mark.parametrize("device", DEVICES)
def test_covering_mesh_exactness_and_directions(device):
    from .conftest import tiling
    tris = tiling(F(2), 2)
    V = np.array(sorted({(float(a), float(b)) for T in tris for a, b in T}))
    idx = {tuple(v): i for i, v in enumerate(V)}
    E = np.array([[idx[(float(a), float(b))] for a, b in T] for T in tris])
    mesh = B.BoundaryMesh(torch.as_tensor(V, dtype=TD, device=device), torch.as_tensor(E, dtype=torch.int32, device=device))
    pos = np.array([[0.1, -0.2], [0.0, 0.0], [0.75, 0.0], [-0.3, 0.4]])
    ps = make_state(pos, device)
    for op in (WarpOperation.Density,):
        props = OperationProperties(kernel=KernelFunctions.Wendland4, operation=op, operationMode=OperationDirection.BoundaryToFluid)
        out = B.boundaryOperation(ps, props, mesh)
        assert np.abs(out.cpu().numpy() - 1).max() < 1e-12                         # support covered by the mesh: integral 1
    # linear field A = 2 + x - 3y: interpolation = A(x_i) + 0 ... exact: int A W = A(x_i) (kernel moments), gradient = grad A
    A = torch.as_tensor(2 + V[:, 0] - 3 * V[:, 1], dtype=TD, device=device)
    props = OperationProperties(kernel=KernelFunctions.Wendland4, operation=WarpOperation.Interpolate, operationMode=OperationDirection.BoundaryToFluid)
    out = B.boundaryOperation(ps, props, mesh, referenceValues=A)
    ref = 2 + pos[:, 0] - 3 * pos[:, 1]
    assert np.abs(out.cpu().numpy() - ref).max() < 1e-12
    props = OperationProperties(kernel=KernelFunctions.Wendland4, operation=WarpOperation.Gradient, gradientMode=GradientScheme.Naive,
                                operationMode=OperationDirection.BoundaryToFluid)
    g = B.boundaryOperation(ps, props, mesh, queryValues=torch.zeros(len(pos), dtype=TD, device=device), referenceValues=A)
    assert np.abs(g.cpu().numpy() - np.array([1.0, -3.0])).max() < 1e-11
    # directions: fluid sources never see the boundary elements; boundary-kind queries are filtered
    for mode_, expect_zero in ((OperationDirection.FluidToFluid, True), (OperationDirection.BoundaryToBoundary, True),
                               (OperationDirection.AllToAll, False), (OperationDirection.BoundaryToFluid, False)):
        pr = OperationProperties(kernel=KernelFunctions.Wendland4, operation=WarpOperation.Density, operationMode=mode_)
        o = B.boundaryOperation(ps, pr, mesh)
        assert (np.abs(o.cpu().numpy()).max() == 0) == expect_zero, mode_


@pytest.mark.parametrize("device", DEVICES)
def test_reaction_conserves_momentum(device):
    mesh, V, E = random_mesh(device, seed=3)
    rng = np.random.default_rng(2)
    N = 30
    pos = rng.uniform(-1.2, 1.2, (N, 2))
    ps = make_state(pos, device, rho=rng.uniform(.9, 1.1, N), mass=rng.uniform(.5, 2, N))
    props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Gradient, gradientMode=GradientScheme.Symmetric,
                                operationMode=OperationDirection.BoundaryToFluid)
    p = torch.as_tensor(rng.uniform(0, 2, N), dtype=TD, device=device)
    pb = torch.as_tensor(rng.uniform(0, 2, len(V)), dtype=TD, device=device)
    out, R = B.boundaryOperation(ps, props, mesh, queryValues=p, referenceValues=pb, referenceDensities=torch.as_tensor(rng.uniform(.9, 1.1, len(V)), dtype=TD, device=device), returnReaction=True)
    total_fluid = (ps.masses.to(TD)[:, None] * out).sum(0)
    assert (R.sum(0) + total_fluid).abs().max() < 1e-12 * (1 + total_fluid.abs().max())


def test_unsupported_kernels_and_dimensions_are_loud():
    with pytest.raises(NotImplementedError):
        B.kernelName(KernelFunctions.Gaussian)
    with pytest.raises(NotImplementedError):
        B.kernelName(KernelFunctions.HOCT4)
    dev = DEVICES[-1]
    mesh, V, E = random_mesh(dev)
    ps = make_state(np.zeros((2, 2)), dev)
    props = OperationProperties(kernel=KernelFunctions.Wendland4, operation=WarpOperation.Laplacian)
    with pytest.raises(NotImplementedError):
        B.boundaryOperation(ps, props, mesh, queryValues=torch.zeros(2, dtype=TD, device=dev), referenceValues=torch.zeros(len(V), dtype=TD, device=dev))
