"""Reusable adversarial geometry, independent references and measured budgets.

All quantities use atlas.CHANNELS and h=1. Missing channels are NaN; failures
and nonfinite values of implemented channels are counted, never filtered out of
the report as if they were accurate. Signal scales are maxima over a case's
queries, not pointwise relative errors at symmetry zeros. Estimates are not
certified bounds. A method without a same-model reference reports TOTAL only.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .atlas import CHANNELS, ALPHAS, _reference_rule, polygon_channels

SUPPORTED = (0, 1, 2, 6, 7, 8, 9, 10, 11)


@dataclass
class PolygonCase:
    name: str
    family: str
    parameter: float
    vertices: np.ndarray

    def distance(self, points):
        """Exact segment distance with a winding sign, including concave polygons."""
        points = np.asarray(points, dtype=float)
        a = self.vertices[None]-points[:, None]
        b = np.roll(a, -1, axis=1)
        edge = b-a
        t = np.clip(-(a*edge).sum(-1)/(edge*edge).sum(-1), 0, 1)
        d = np.linalg.norm(a+t[..., None]*edge, axis=-1).min(1)
        cross = a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
        winding = (((a[..., 1] <= 0) & (b[..., 1] > 0) & (cross > 0)).sum(1)
                   - ((b[..., 1] <= 0) & (a[..., 1] > 0) & (cross < 0)).sum(1))
        return np.where(winding != 0, -d, d)

    def probes(self):
        """Deterministic edge/vertex/medial-axis/grazing probes plus seeded queries."""
        v = self.vertices
        edge = np.roll(v, -1, axis=0)-v
        normals = np.stack((edge[:, 1], -edge[:, 0]), axis=1)/np.linalg.norm(edge, axis=1)[:, None]
        middle = (v+np.roll(v, -1, axis=0))/2
        points = [v, middle, v.mean(0, keepdims=True)]
        for delta in (1e-14, 1e-10, 1e-5, .03):
            points += [middle+delta*normals, middle-delta*normals]
        # Local probes resolve tiny elements; fixed support offsets expose
        # cancellation for far / tiny elements and nearly tangent support cuts.
        span = np.ptp(v, axis=0).max()
        rng = np.random.default_rng(7301)
        points += [v.mean(0)+span*rng.uniform(-.5, .5, (12, 2)),
                   v.mean(0)+np.array([[.2, .3], [-.3, .2], [.7, .1], [.98, .02]])]
        right = v[np.argmax(v[:, 0])]
        points += [right+np.array([[1-1e-8, 0], [1., 0], [1+1e-8, 0]])]
        return np.concatenate(points)


def polygon_cases():
    base = np.array([[-.4, -.3], [.5, -.2], [-.1, .5]])
    cases = [PolygonCase(f'triangle-size-{s:g}', 'size', s, base*s+[.2, .1])
             for s in (1., .1, .01, .001, 1e-5, 1e-7, 1e-9)]
    for degrees in (.1, 1., 3., 10., 30., 60., 150., 179.):
        # Tip at origin, prescribed interior angle; all vertices within h.
        a = np.deg2rad(degrees/2)
        v = np.array([[0., 0.], [-.8*np.cos(a), .8*np.sin(a)],
                      [-.8*np.cos(a), -.8*np.sin(a)]])
        cases.append(PolygonCase(f'triangle-angle-{degrees:g}', 'angle', degrees, v))
    for ratio in (.1, .001, 1e-5, 1e-7):
        cases.append(PolygonCase(f'sliver-{ratio:g}', 'aspect', ratio,
                     np.array([[-.8, -.4*ratio], [.8, 0.], [-.8, .4*ratio]])))
    cases.append(PolygonCase('reflex-notch', 'reflex', 270.,
                 np.array([[-.8, -.8], [.8, -.8], [.8, 0.], [0., 0.], [0., .8], [-.8, .8]])))
    # A sharp reentrant notch with walls separated by much less than h.
    cases.append(PolygonCase('acute-notch', 'notch', 2.,
                 np.array([[-.8, -.8], [.8, -.8], [.8, .8], [.015, .8],
                           [0., -.4], [-.015, .8], [-.8, .8]])))
    return cases


def triangle_area_reference(vertices, points, kernel, nodes=48):
    """Independent 2D Duffy rule, stable factored kernel, extended precision.

    Useful for tiny triangles: no indicator cancellation or inverse barycentric
    matrix. Refinement is required at the kernel origin, knots and support rim.
    """
    from ..edge.np_fem import kernel_ubasis
    v, points = np.asarray(vertices, dtype=np.longdouble), np.asarray(points, dtype=np.longdouble)
    roots, weights = _reference_rule(nodes)
    u, w = (roots+1)/2, weights/2
    l1, l2 = np.meshgrid(u, u, indexing='ij')
    l2 = l2*(1-l1)
    location = l1[..., None]*(v[1]-v[0])+l2[..., None]*(v[2]-v[0])
    e0, e1 = v[1]-v[0], v[2]-v[0]
    jac = abs(e0[0]*e1[1]-e0[1]*e1[0])
    area = (w[:, None]*w[None]*(1-l1)*jac).ravel()
    location = location.reshape(-1, 2)
    pi = np.arccos(np.longdouble(-1))
    out = []
    # Float64 rule nodes or pi would otherwise set the reference's eps floor.
    for chunk in np.array_split(points, max(1, (len(points)+7)//8)):
        y = (v[0]-chunk[:, None])+location[None]
        r = np.sqrt((y*y).sum(-1))
        W, dw = np.zeros_like(r), np.zeros_like(r)
        for R, coeffs in kernel_ubasis(kernel).items():
            R = np.longdouble(R.numerator)/R.denominator
            t = np.maximum(R-r, 0)
            for power, c in coeffs.items():
                c = np.longdouble(c.numerator)/c.denominator
                W += np.where(r < R, c*t**power, 0)
                if power:
                    dw -= np.where(r < R, power*c*t**(power-1), 0)
        W /= pi
        radial = -dw/pi/np.maximum(r, np.finfo(np.longdouble).tiny)
        from ..edge.kernels import kernel as get_kernel
        _, hi, origin = get_kernel(kernel).pieces[0]
        if origin[1] == 0:
            # Independent direct powers of the exact origin-piece derivative.
            origin_radial = sum(-k*(np.longdouble(c.numerator)/c.denominator)*r**(k-2)
                                for k, c in enumerate(origin) if k >= 2)
            radial = np.where(r < min(.125, float(hi)), origin_radial/pi, radial)
        g = radial[..., None]*y
        values = [((y[..., 0]**a*y[..., 1]**b*W)*area).sum(1) for a, b in ALPHAS]
        values += [(g[..., j]*area).sum(1) for j in range(2)]
        values += [(y[..., d]*g[..., j]*area).sum(1) for d in range(2) for j in range(2)]
        out.append(np.stack(values, axis=1))
    return np.concatenate(out)


def precision_polygon_reference(vertices, points, kernel, dps=70):
    """Analytic identities at arbitrary precision for exact float-input geometry.

    Shares the edge formulas, so precision refinement is not independent
    validation. Pair with polar / area references. Returns longdouble storage.
    """
    import mpmath as mp
    from ..edge import core, geometry
    out = []
    with mp.workdps(dps+20):
        for x in points:
            prepared = geometry.prepare(vertices, x, 1)
            m = [core.moment(vertices, x, kernel, al, dps=dps, prepared=prepared) for al in ALPHAS]
            g = list(core.gradient(vertices, x, kernel, dps=dps, prepared=prepared))
            g1 = [list(core.moment_gradient(vertices, x, kernel, al, dps=dps, prepared=prepared))
                  for al in [(1, 0), (0, 1)]]
            g1[0][0] += m[0]
            g1[1][1] += m[0]
            out.append([np.longdouble(str(v)) for v in m+g+g1[0]+g1[1]])
    return np.asarray(out, dtype=np.longdouble)


def reference_precision_estimate(fine, coarse):
    """Precision-refinement difference including a longdouble storage floor."""
    return np.maximum(abs(fine-coarse), 4*np.finfo(np.longdouble).eps*abs(fine))


def target_reference(case, points, kernel):
    if case.family == 'aspect' and case.parameter <= .001:
        fine = precision_polygon_reference(case.vertices, points, kernel, 70)
        coarse = precision_polygon_reference(case.vertices, points, kernel, 45)
        return fine, reference_precision_estimate(fine, coarse), '70/45-digit analytic identities; independent polar spot checks'
    if case.family == 'size' and case.parameter <= .01:
        fine = triangle_area_reference(case.vertices, points, kernel, 64)
        coarse = triangle_area_reference(case.vertices, points, kernel, 32)
        return fine, np.abs(fine-coarse), 'independent extended-precision 2D Duffy quadrature'
    target = polygon_channels(case.vertices, points, kernel, dtype=np.longdouble)
    return target, np.full_like(target, np.nan), 'extended-precision polygon identity; independent oracle spot checks'


@dataclass
class Measurement:
    name: str
    values: np.ndarray
    model_reference: np.ndarray | None = None
    estimate: np.ndarray | None = None
    supported: tuple = tuple(range(12))
    accepted: np.ndarray | None = None
    note: str = ''
    domain: np.ndarray | None = None
    status: np.ndarray | None = None


def measured_budget(measurement, target):
    total = measurement.values-target
    numerical = np.full_like(total, np.nan)
    model = np.full_like(total, np.nan)
    if measurement.model_reference is not None:
        numerical = measurement.values-measurement.model_reference
        model = measurement.model_reference-target
    return numerical, model, total


def report(measurement, target, reference_estimate=None):
    available = np.zeros(target.shape, bool)
    available[:, measurement.supported] = True
    if measurement.domain is not None:
        available &= measurement.domain[:, None]
    finite = np.isfinite(measurement.values) & np.isfinite(target) & available
    error = np.where(finite, np.abs(measurement.values-target), np.nan)
    scale = np.maximum(np.max(np.abs(target), axis=0), np.finfo(float).tiny)
    maximum = lambda a: [float(np.nanmax(a[:, j])) if np.isfinite(a[:, j]).any() else None for j in range(12)]
    result = dict(name=measurement.name, note=measurement.note, supported_channels=[CHANNELS[j] for j in measurement.supported],
                  nonfinite_implemented_values=int((available & ~np.isfinite(measurement.values)).sum()),
                  max_absolute_error=maximum(error), max_error_over_signal_scale=maximum(error/scale),
                  signal_scale=[float(v) for v in scale],
                  error_split_available=measurement.model_reference is not None)
    if measurement.status is not None:
        result['quadrature_queries'] = int((measurement.status != 0).sum())
        result['quadrature_failed_queries'] = int((measurement.status < 0).sum())
    if reference_estimate is not None:
        result['max_reference_estimate'] = maximum(reference_estimate)
    result['in_domain_queries'] = len(target) if measurement.domain is None else int(measurement.domain.sum())
    result['status'] = ('unrepresentable_input' if 'not representable' in measurement.note else
                        'outside_domain' if result['in_domain_queries'] == 0 else
                        'nonfinite_result' if result['nonfinite_implemented_values'] else 'evaluated')
    if measurement.estimate is not None:
        est = np.abs(measurement.estimate)
        result['max_refinement_estimate'] = maximum(est)
        uncertainty = np.zeros_like(error) if reference_estimate is None else np.nan_to_num(reference_estimate)
        # Count ONLY underestimates exceeding a floating arithmetic / reference
        # uncertainty floor. Differences of two rules are not error bounds.
        result['refinement_underestimates_above_roundoff_floor'] = int((
            finite & (error > 2*est+uncertainty+64*np.finfo(measurement.values.dtype).eps*scale)).sum())
    if measurement.accepted is not None:
        accepted = measurement.accepted[:, None] & available
        result['accepted_queries'] = int(measurement.accepted.sum())
        result['accepted_m00_errors_over_0_01'] = int((measurement.accepted & (error[:, 0] > .01)).sum())
        result['max_accepted_absolute_error'] = maximum(np.where(accepted, error, np.nan))
    return result


def fem_channels(vertices, points, kernel, hybrid=False, nodes=8):
    """Actual P2 nodal weights contracted to degree-2 moments and g0/g1."""
    from ..edge import fem, np_fem
    v = np.broadcast_to(vertices, (len(points), 3, 2))
    fn = np_fem.weights_hybrid if hybrid else np_fem.weights
    kwargs = {'nodes': nodes} if hybrid else {}
    weights, gradient = fn(v, points, kernel, 2, grad=True, **kwargs)
    relative = (vertices[0]-points[:, None])+(np.asarray(fem.basis(2)[0], dtype=float) @ (vertices-vertices[0]))[None]
    m = [(weights*relative[..., 0]**a*relative[..., 1]**b).sum(1) for a, b in ALPHAS]
    m += [gradient[..., j].sum(1) for j in range(2)]
    m += [(relative[..., d]*gradient[..., j]).sum(1) for d in range(2) for j in range(2)]
    return np.stack(m, axis=1)


def make_sdf(case, points, spacing):
    import torch
    from ..scene.scene import SdfRep, SurfaceRep
    lo = np.minimum(case.vertices.min(0), points.min(0))-1.1
    hi = np.maximum(case.vertices.max(0), points.max(0))+1.1
    return SdfRep.fromFunction(lambda p: torch.as_tensor(case.distance(p.numpy()), dtype=torch.float64),
                               lo, hi, spacing, fallback=SurfaceRep.polygon(case.vertices))


def scene_channels(rep, points, kernel='w4', device='cuda:0', *, fused=False, stable=False, second=False, return_diagnostics=False):
    """Execute Scene or the fused provider; never substitute a reference result."""
    import torch
    from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties, ParticleState, WarpOperation)
    from ..scene.scene import Scene, Body
    from ..scene.fused import FusedWall, WallOutput
    from ..scene.fixedadj import fixed_adjacency
    from ..edge.warpfused import FusedGroup
    scene = Scene([Body(bodyId=0, reps=[rep])], device)
    x = torch.as_tensor(np.asarray(points, dtype=float), dtype=torch.float64, device=device)
    one = torch.ones(len(x), dtype=torch.float64, device=device)
    ps = ParticleState(positions=x, supports=one, masses=one, densities=one,
                       kinds=torch.zeros(len(x), dtype=torch.int32, device=device))
    props = OperationProperties(kernel=kernel, operation=WarpOperation.Density,
                                gradientMode=GradientScheme.Naive, operationMode=OperationDirection.BoundaryToFluid)
    estimates = torch.zeros((len(x),12),dtype=torch.float64,device=device)
    statuses = torch.zeros(len(x),dtype=torch.int32,device=device)
    if fused:
        nodes,panels = stable if isinstance(stable,tuple) else (8,6)
        group = FusedGroup(kernel, route='cheb' if stable else 'mono', nodes=nodes, panels=panels)
        wall = FusedWall(scene, fixed_adjacency(scene, ps, props, 1.), (group,))
        outputs = (WallOutput('lam', 0, 'lam'), WallOutput('m1', 0, 'm1'),
                   WallOutput('G', 0, 'g0'), WallOutput('Cov', 0, 'cov'))
        a = wall.evaluate(outputs+((WallOutput('m2',0,'m2'),) if second else ()))
        columns = [a['lam'][0, :, None], a['m1'][0], a['G'][0], a['Cov'][0].reshape(-1, 4)]
        if second:
            columns.append(a['m2'][0].reshape(-1,4)[:,[0,1,3]])
        values = torch.cat(columns, dim=1)
        for item in wall.items:
            if 0 in item['estimates']:
                estimates.index_add_(0,item['ba'].cand,item['estimates'][0])
                statuses[item['ba'].cand] = item['qstatus'][:,0]

    else:
        values = torch.zeros((len(x), 12 if second else 9), dtype=torch.float64, device=device)
        missing = torch.zeros_like(values,dtype=torch.bool)
        for entry in scene.pairMoments(ps, props).entries:
            for m in entry['surface']+entry['implicit']:
                first = torch.zeros_like(m.g0) if m.m1 is None else m.m1
                cov = torch.zeros((len(m.q), 4), dtype=torch.float64, device=device) if m.g1 is None else m.g1.reshape(-1, 4)
                columns = [m.lam[:,None],first,m.g0,cov]
                if second:
                    columns.append(torch.zeros((len(m.q),3),dtype=torch.float64,device=device) if m.m2 is None else m.m2.reshape(-1,4)[:,[0,1,3]])
                    if m.m2 is None:
                        missing[m.q.long(),9:] = True
                values.index_add_(0, m.q.long(), torch.cat(columns, dim=1))
                if m.quadrature_estimate is not None:
                    estimates.index_add_(0,m.q.long(),m.quadrature_estimate)
                    statuses[m.q.long()] = m.quadrature_status
                if m.m1 is None:
                    missing[m.q.long(), 1:3] = True
                if m.g1 is None:
                    missing[m.q.long(), 5:9] = True
        values[missing] = torch.nan
    out = np.full((len(x), 12), np.nan)
    order = (*SUPPORTED,3,4,5) if second else SUPPORTED
    out[:, order] = values.detach().cpu().numpy()
    if return_diagnostics:
        est = estimates.detach().cpu().numpy()[:,[0,1,2,9,10,11,3,4,5,6,7,8]]
        status = statuses.detach().cpu().numpy()
        est[status==0] = np.nan
        return out,est,status
    return out


def particle_channels(scene, points, kernel, spacing):
    """Actual ParticleBoundary aggregate; m1 and scalar second moments unsupported."""
    import torch
    from warpSPHCore import ParticleState, KernelFunctions
    from ..scene.particles import ParticleBoundary
    x = torch.as_tensor(np.asarray(points, dtype=float), dtype=torch.float64, device=scene.device)
    one = torch.ones(len(x), dtype=torch.float64, device=scene.device)
    ps = ParticleState(positions=x, supports=one, masses=one, densities=one,
                       kinds=torch.zeros(len(x), dtype=torch.int32, device=scene.device))
    family = {'w2': KernelFunctions.Wendland2, 'w4': KernelFunctions.Wendland4}[kernel]
    a = ParticleBoundary(scene, spacing).aggregate(ps, 1., kernel=family).out
    out = np.full((len(x), 12), np.nan)
    out[:, 0] = a['lam'][0].cpu().numpy()
    out[:, 6:8] = a['G'][0].cpu().numpy()
    out[:, 8:] = a['Cov'][0].reshape(-1, 4).cpu().numpy()
    return out


def contour_reference(rep, points, kernel):
    """Signed triangle chain of the actual oriented marching-squares segments.

    Marching squares stores separate endpoint indices per segment. A signed fan
    avoids assuming shared indices or convexity; it also preserves hole signs.
    """
    surf = rep.contour()
    v, edges = surf.vertices.cpu().numpy(), surf.edges.cpu().numpy()
    out = np.zeros((len(points), 12), dtype=np.longdouble)
    centre = v.mean(0)
    triangles = np.stack((np.broadcast_to(centre, (len(edges), 2)), v[edges[:, 0]], v[edges[:, 1]]), axis=1)
    a, b = triangles[:, 1]-centre, triangles[:, 2]-centre
    signs = np.sign(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])
    # Accumulate in small edge batches to bound memory even for refined grids.
    from ..edge import np2d
    for start in range(0, len(triangles), 16):
        tri, sign = triangles[start:start+16], signs[start:start+16]
        good = sign != 0
        tri, sign = tri[good], sign[good]
        if not len(tri):
            continue
        verts = np.tile(tri, (len(points), 1, 1))
        query = np.repeat(points, len(tri), axis=0)
        columns = [np2d.moment(verts, query, kernel, al, dtype=np.longdouble) for al in ALPHAS]
        for al in ((0, 0), (1, 0), (0, 1)):
            g = np2d.grad_moment(verts, query, kernel, al, dtype=np.longdouble)
            columns.extend((g[:, 0], g[:, 1]))
        out += (np.stack(columns, axis=1).reshape(len(points), len(tri), 12)*sign[None, :, None]).sum(1)
    # These fixtures enclose finite solids with fluid border; background=0.
    if surf.background:
        raise ValueError('finite-solid fixtures require a fluid sampling border')
    return out
