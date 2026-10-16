"""Cross-product rounding and support normalization of edge/indicator jumps."""
import math

import numpy as np
import pytest
import torch
import warp as wp

from warpSPHBoundaries.edge.predicates import cross2_f64, cross2_f32
from warpSPHBoundaries.edge import warpbc, warpfused
from warpSPHBoundaries.edge.warpfused import FusedGroup
from warpSPHBoundaries.scene.fixedadj import fixed_topology
from warpSPHBoundaries.scene.scene import SurfaceRep
from warpSPHBoundaries.validation.adversarial import polygon_cases
from warpSPHBoundaries.validation.atlas import polygon_channels

DEVICES = ['cpu']+(['cuda:0'] if wp.is_cuda_available() else [])


@wp.kernel
def cross64(a: wp.array2d(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    out[i] = cross2_f64(a[i, 0], a[i, 1], a[i, 2], a[i, 3])


@wp.kernel
def cross32(a: wp.array2d(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = cross2_f32(a[i, 0], a[i, 1], a[i, 2], a[i, 3])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,kernel,wtype', [(np.float64, cross64, wp.float64), (np.float32, cross32, wp.float32)])
def test_rounded_cross_matches_indicator_even_when_fma_changes_collinearity(device, dtype, kernel, wtype):
    p = np.array([.3998629299902229, -.010470779323149262], dtype=dtype)
    data = np.array([[*p, *(-p)], [*p, p[0], p[1]],
                     [*p, -p[0], np.nextafter(-p[1], dtype(np.inf))],
                     [*p, -p[0], np.nextafter(-p[1], dtype(-np.inf))]], dtype=dtype)
    expected = data[:, 0]*data[:, 3]-data[:, 1]*data[:, 2]
    # Demonstrate the failure mechanism independently of GPU availability.
    if dtype == np.float64 and hasattr(math, 'fma'):
        assert expected[0] == 0
        assert math.fma(float(p[0]), float(-p[1]), -float(p[1])*float(-p[0])) != 0
    a = wp.array(data, dtype=wtype, device=device)
    out = wp.zeros(len(data), dtype=wtype, device=device)
    wp.launch(kernel, dim=len(data), inputs=[a, out], device=device)
    np.testing.assert_array_equal(out.numpy(), expected)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('solid', ['inside', 'outside'])
@pytest.mark.parametrize('fused', [False, True])
def test_boundary_integrals_share_normalized_predicates_with_both_adjacencies(device, solid, fused):
    case = next(c for c in polygon_cases() if c.name == 'triangle-angle-3')
    # Exact vertices/midpoints, then tiny displacements to both sides.
    middle = (case.vertices+np.roll(case.vertices, -1, axis=0))/2
    points = np.vstack((case.vertices, middle, middle+[0., 1e-12], middle-[0., 1e-12]))
    h = np.resize(np.array([.37, .7, 1., 1.3]), len(points))
    rep = SurfaceRep.polygon(case.vertices, solid=solid).to(device)
    pos = torch.tensor(points, dtype=torch.float64, device=device)
    sup = torch.tensor(h, dtype=torch.float64, device=device)
    topo = rep.topology(pos, sup)
    fixed = fixed_topology(rep, pos, sup, torch.ones(len(pos), dtype=torch.bool, device=device), float(h.max()))
    indicator = topo.indicator(rep, pos)
    torch.testing.assert_close(fixed.ind, indicator, atol=2e-15, rtol=0)
    # Eager and fixed topology need the same support normalization; restricting
    # a topology must retain it even when its indicator has not been cached.
    fresh = rep.topology(pos, sup)
    rows = torch.arange(0, len(pos), 2, device=device)
    rowmap = torch.full((len(pos),), -1, device=device)
    rowmap[rows] = torch.arange(len(rows), device=device)
    restricted = fresh.restrict(rowmap, rows)
    torch.testing.assert_close(restricted.indicator(rep, pos[rows]), indicator[rows], atol=2e-15, rtol=0)
    qi, e = topo.qi, topo.e
    if fused:
        values = warpfused.fused_channels(qi, e, pos, sup, rep.vertices, rep.edges,
                                         (FusedGroup('w4', route='mono'),), device=device, as_float64=True)
    else:
        values = warpbc.edge_channels(qi, e, pos, sup, rep.vertices, rep.edges, 'w4', device=device, stable=False)
    aggregate = torch.zeros((len(pos), 9), dtype=torch.float64, device=device)
    aggregate.index_add_(0, qi.long(), values)
    aggregate[:, 0] += indicator
    aggregate[:, 5] += indicator
    aggregate[:, 8] += indicator
    target = []
    for point, support in zip(points, h):
        a = polygon_channels(case.vertices/support, point[None]/support, 'w4', dtype=np.longdouble)[0]
        if solid == 'outside':
            full = np.zeros(12); full[[0, 8, 11]] = 1
            a = full-a
        target.append(a[[0, 1, 2, 6, 7, 8, 9, 10, 11]])
    np.testing.assert_allclose(aggregate.cpu().numpy(), target, rtol=0, atol=3e-11)
