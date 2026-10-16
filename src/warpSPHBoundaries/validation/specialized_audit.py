"""Specialized approximation sweeps, representation checks and independent oracles."""
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import json

import mpmath as mp
import numpy as np

from .atlas import (CHANNELS, LABELS, _pyplot, _disk_lookup, disk_reference,
                    polygon_channels, _reference_rule)
from .adversarial import Measurement, SUPPORTED, scene_channels, report, target_reference


def register_kernels():
    from ..scene import tensile
    from ..scene.viscosity import lap_factor
    for name in ('w2', 'w4'):
        lap_factor(1., name)
        tensile._register(name)


@lru_cache(maxsize=None)
def _factors(kernel, derivative):
    """Factor exact high-degree profiles before evaluating reference integrands."""
    import sympy as sp
    from ..edge.kernels import kernel as get_kernel
    r = sp.Symbol('r')
    result = []
    for lo, hi, co in get_kernel(kernel).pieces:
        poly = sum(sp.Rational(c.numerator, c.denominator)*r**i for i, c in enumerate(co))
        if derivative:
            poly = -sp.diff(poly, r)
        c, factors = sp.factor_list(poly, r)
        def rational(x):
            return np.longdouble(int(x.p))/int(x.q)
        result.append((float(lo), float(hi), rational(c),
                       [(np.array([rational(x) for x in sp.Poly(f, r).all_coeffs()]), n) for f, n in factors]))
    return result


def factored_radial(lo, hi, kernel, power, derivative=False):
    """Gaussian integration exact in polynomial degree, with factored profiles.

    Used for high-degree tensile kernels whose expanded antiderivatives suffer
    cancellation even in longdouble. Independent of the fitted disk tables.
    """
    roots, weights = _reference_rule(24)
    out = np.zeros_like(lo)
    for a, b, c, factors in _factors(kernel, derivative):
        left, right = np.maximum(lo, a), np.minimum(hi, b)
        half = np.maximum(right-left, 0)/2
        mid = left+half
        # Chunking the extra radial dimension keeps memory bounded.
        for root, weight in zip(roots, weights):
            r = mid+half*root
            value = np.full_like(r, c)
            for co, exponent in factors:
                value *= np.polyval(co, r)**exponent
            out += weight*half*value*r**power/np.arccos(np.longdouble(-1))
    return out


def table_reference(points, radius, kernel, nodes):
    if kernel not in ('w2p5', 'w4p5'):
        return disk_reference(points, radius, kernel, nodes)
    # Select the stable radial evaluator explicitly, without modifying global
    # functions or the production implementation under test.
    return disk_reference(points, radius, kernel, nodes, radial_evaluator=factored_radial)


def arc_measurements(points, radius, kernel, target):
    """Actual circle-arc evaluator, including its boundary/centre conventions."""
    from ..edge.disk import disk_channels
    points = np.asarray(points, dtype=float)
    distance = np.linalg.norm(points, axis=1)
    e = -points/np.maximum(distance[:, None], 1e-300)
    e[distance == 0] = (1, 0)
    ex, ey = e.T
    values = []
    for nodes in (24, 48):
        data = disk_channels(kernel, distance, radius, nodes=nodes)
        out = np.full_like(target, np.nan, dtype=float)
        out[:, 0] = data['lam']
        out[:, 1:3] = data['m1x'][:, None]*e
        out[:, 6:8] = data['g0x'][:, None]*e
        delta = data['g1xx']-data['g1yy']
        out[:, 8:] = np.column_stack((data['g1yy']+delta*ex*ex, delta*ex*ey,
                                       delta*ex*ey, data['g1yy']+delta*ey*ey))
        values.append(out)
    return [Measurement('arc-24', values[0], target, values[0]-values[1], supported=SUPPORTED,
                         note='actual circle quadrature; D=R and D=0 conventions included'),
            Measurement('arc-48', values[1], target, supported=SUPPORTED)]


def rectangle_reference(lo, hi, points, kernel, nodes=64):
    """Independent area quadrature for high-degree table reference checks.

    Panels split at the query's x/y coordinates, so the radial origin is a
    panel corner. Factored rational profiles avoid the high-degree monomial
    cancellation of the production formula or its longdouble counterpart.
    """
    roots, weights = _reference_rule(nodes)
    pi = np.arccos(np.longdouble(-1))
    results = []
    for point in np.asarray(points, dtype=np.longdouble):
        breaks = [sorted(set([np.longdouble(lo[d]), np.longdouble(hi[d]),
                              np.clip(point[d], lo[d], hi[d])])) for d in range(2)]
        total = np.zeros(12, dtype=np.longdouble)
        for x0, x1 in zip(breaks[0][:-1], breaks[0][1:]):
            for y0, y1 in zip(breaks[1][:-1], breaks[1][1:]):
                x = (x0+x1)/2+(x1-x0)/2*roots-point[0]
                y = (y0+y1)/2+(y1-y0)/2*roots-point[1]
                X, Y = np.meshgrid(x, y, indexing='ij')
                r = np.sqrt(X*X+Y*Y)
                profiles = []
                for derivative in (False, True):
                    value = np.zeros_like(r)
                    for lower, upper, c, factors in _factors(kernel, derivative):
                        f = np.full_like(r, c)
                        for co, exponent in factors:
                            f *= np.polyval(co, r)**exponent
                        value += np.where((r >= lower) & (r < upper), f/pi, 0)
                    profiles.append(value)
                W, negative_dw = profiles
                gx, gy = negative_dw*X/r, negative_dw*Y/r
                w = weights[:, None]*weights[None]*(x1-x0)*(y1-y0)/4
                fields = [W, X*W, Y*W, X*X*W, X*Y*W, Y*Y*W,
                          gx, gy, X*gx, X*gy, Y*gx, Y*gy]
                total += np.array([(w*f).sum() for f in fields])
        results.append(total)
    return np.array(results)


def plot_parameter(folder, rows, title, xlabel, estimates=False):
    plt = _pyplot()
    names = list(dict.fromkeys(m['name'] for row in rows for m in row['methods']))
    palette = {name: plt.get_cmap('tab20')(i % 20) for i, name in enumerate(names)}
    if estimates:
        names = [n for n in names if n.endswith('order-2') or n == 'arc-24']
    fig, axes = plt.subplots(2, 6, figsize=(18, 8), constrained_layout=True)
    for channel, ax in enumerate(axes.flat):
        implemented = False
        for name in names:
            x, y = [], []
            for row in rows:
                item = next((m for m in row['methods'] if m['name'] == name), None)
                value = None if item is None else item['max_absolute_error'][channel]
                if value is not None:
                    x.append(row['parameter']); y.append(max(value, 1e-18))
            if x:
                implemented = True
                line, = ax.loglog(x, y, '.-', color=palette[name], lw=1, label=name)
                if estimates:
                    ex, ey = [], []
                    for row in rows:
                        item = next((m for m in row['methods'] if m['name'] == name), None)
                        value = None if item is None else item.get('max_refinement_estimate', [None]*12)[channel]
                        if value is not None:
                            ex.append(row['parameter']); ey.append(max(value, 1e-18))
                    ax.loglog(ex, ey, '--', color=line.get_color(), label=name+' order difference')
        rx, ry = [], []
        for row in rows:
            if row['methods']:
                value = row['methods'][0].get('max_reference_estimate', [None]*12)[channel]
                if value is not None:
                    rx.append(row['parameter']); ry.append(max(value, 1e-18))
        if rx:
            ax.loglog(rx, ry, ':', color='#333333', label='reference refinement estimate')
        if not implemented:
            ax.set_facecolor('#eeeeee')
            ax.text(.5, .55, 'method channel\nnot implemented', ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.axhline(8*np.finfo(float).eps, color='#999999', ls=':')
        ax.set(title=LABELS[channel], xlabel=xlabel, ylabel='maximum absolute discrepancy')
        ax.grid(alpha=.15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=5, fontsize=8)
    regime = ('T3 below R/h=2 is forced extrapolation; T4 disk smooth-piece domain is enforced. ' if folder.name.startswith('disk-') else
              'Strip curves include forced interior/large-width extrapolation; d>a acceptance is recorded. ' if folder.name.startswith('strip-') else
              'Forced SDF extrapolation; ellipse reference includes chord-refinement uncertainty. ' if folder.name == 'variable-curvature-ellipse' else '')
    fig.suptitle(title+'\n'+regime+'\nGrey: unsupported method channels. Refinement/order differences are empirical estimates, not bounds.', fontsize=11)
    fig.savefig(folder/('estimates.png' if estimates else 'errors.png'))
    plt.close(fig)


def save_row(folder, parameter, target, uncertainty, methods, **arrays):
    from .robustness import save_measurements
    folder.mkdir(parents=True, exist_ok=True)
    return dict(parameter=float(parameter), methods=save_measurements(folder, target, uncertainty, methods, **arrays))


def _curvature(points, radius, kernel, cavity=False, order=2):
    import torch
    from ..scene.implicitBodies import Tier3
    distance = np.linalg.norm(points, axis=1)
    q, normal, k = distance-radius, points/np.maximum(distance[:, None], 1e-300), 1/radius
    if cavity:
        q, normal, k = -q, -normal, -k
    table = Tier3(kernel, 'cpu')
    qt = torch.tensor(q, dtype=torch.float64)
    lam, dl = table.lam(qt, torch.full_like(qt, k))
    if order == 0:
        lam, dl = table.lam(qt, torch.zeros_like(qt))
    elif order == 1:
        f2, df2 = table._hermite(2, qt.abs())
        sign = torch.where(qt >= 0, 1., -1.)
        lam, dl = lam-sign*k*k*f2, dl-k*k*df2
    _, _, first, nn, tt = [v.numpy() for v in table.planar_moments(qt)]
    out = np.full((len(points), 12), np.nan)
    out[:, 0], out[:, 1:3], out[:, 6:8] = lam.numpy(), -first[:, None]*normal, dl.numpy()[:, None]*normal
    tensor = tt[:, None, None]*np.eye(2)+(nn-tt)[:, None, None]*normal[:, :, None]*normal[:, None, :]
    out[:, 8:] = tensor.reshape(-1, 4)
    return out


def plane_total(kernel):
    from .atlas import _radial_integral
    full = np.zeros(12, dtype=np.longdouble)
    full[0], full[8], full[11] = 1, 1, 1
    full[3] = full[5] = np.arccos(np.longdouble(-1))*_radial_integral(np.array([0.], dtype=np.longdouble), np.array([1.], dtype=np.longdouble), kernel, 3)[0]
    return full


def disk_sweeps(output, kernels, device):
    import torch
    from ..scene.implicitBodies import Tier4
    from ..scene.scene import ImplicitRep
    from ..scene.implicitBodies import DiskBody
    register_kernels()
    table_kernels = ('w2', 'w4', 'cone', 'lw2', 'lw4', 'w2p5', 'w4p5')
    radii = sorted(set([1e-6, 1e-3, .01, .05, .1, .19999999, .2, .20000001,
                        .49999999, .5, .50000001, .7, .99999999, 1., 1.00000001,
                        1.99999999, 2., 2.00000001, 4., 16., 100.]))
    result = []
    for kernel in dict.fromkeys(tuple(kernels)+table_kernels):
        folder = output/('disk-'+kernel)
        rows = []
        series = [Tier4(kernel, 'cpu', K=k) for k in range(3)] if kernel in kernels else []
        for radius in radii:
            distances = np.unique(np.maximum(0., np.r_[0., radius+np.array([-1., -.7, -.2, -1e-9, 0., 1e-9, .1, .3, .7, 1-1e-9, 1., 1+1e-9]),
                                                               1-radius+np.array([-1e-9, 0., 1e-9])]))
            angle = .23
            points = distances[:, None]*np.array([np.cos(angle), np.sin(angle)])
            target = table_reference(points, radius, kernel, 96)
            estimate = np.abs(target-table_reference(points, radius, kernel, 48))
            methods = []
            methods += arc_measurements(points, radius, kernel, target)
            if kernel in table_kernels:
                methods.append(Measurement('disk-lookup', _disk_lookup(points, radius, kernel), target, supported=SUPPORTED))
            if kernel in kernels:
                for cavity in (False, True):
                    previous = None
                    for order in range(3):
                        value = _curvature(points, radius, kernel, cavity, order)
                        # Complement cavity fields to compare signed errors
                        # with the same solid-disk target and common scales.
                        if cavity:
                            value = plane_total(kernel)-value
                        diff = None if previous is None else np.abs(value-previous)
                        methods.append(Measurement(f'T3-{"cavity" if cavity else "circle"}-order-{order}', value,
                                                   estimate=diff, supported=SUPPORTED,
                                                   accepted=np.full(len(points), radius >= 2) if order == 2 else None,
                                                   note='forced curvature model; actual radius threshold recorded; m1/g1 remain planar at EVERY order'))
                        previous = value
                Dt, at = torch.tensor(distances), torch.full((len(points),), radius, dtype=torch.float64)
                previous = None
                for k, table in enumerate(series):
                    val, derivative, valid = table.lam(Dt, at)
                    a = np.full_like(target, np.nan, dtype=float)
                    a[:, 0] = np.where(valid.numpy(), val.numpy(), np.nan)
                    # Library rejects invalid intervals; NaNs there are outside
                    # the domain, not a failed numerical result.
                    order_estimate = None if previous is None else np.abs(a-previous)
                    methods.append(Measurement(f'T4-disk-order-{k}', a, estimate=order_estimate,
                                               supported=(0,), accepted=(valid.numpy() & (radius <= .2)) if k == 2 else None,
                                               domain=valid.numpy(),
                                               note='kernel-piece/rim-invalid queries explicitly masked; order difference is a truncation indicator'))
                    previous = a
                for edges in (12, 24, 48):
                    t = 2*np.pi*np.arange(edges)/edges
                    r = radius*np.sqrt(2*np.pi/(edges*np.sin(2*np.pi/edges)))
                    poly = r*np.column_stack((np.cos(t), np.sin(t)))
                    value = polygon_channels(poly, points, kernel)
                    model = polygon_channels(poly, points, kernel, dtype=np.longdouble)
                    methods.append(Measurement(f'polygon-{edges}', value, model))
                if device is not None and radius in (.19999999, .2, .20000001, 1.99999999, 2., 2.00000001):
                    for cavity in (False, True):
                        rep = ImplicitRep(DiskBody(center=(0., 0.), radius=radius, solid='outside' if cavity else 'inside'))
                        a = scene_channels(rep, points, kernel, device)
                        if cavity:
                            a = plane_total(kernel)-a
                        # Mixed tier rows support lam/G throughout; implicit
                        # disk first moments are unavailable on the oracle path.
                        a[:, [1, 2, 8, 9, 10, 11]] = np.nan
                        methods.append(Measurement('actual-tier-switch-cavity' if cavity else 'actual-tier-switch', a,
                                                   supported=(0, 6, 7), note='actual Scene: T3 / T4 / area-preserving polygon fallback'))
            row = save_row(folder/f'R-{radius:.12g}', radius, target, estimate, methods, points=points)
            rows.append(row)
        folder.mkdir(parents=True, exist_ok=True)
        plot_parameter(folder, rows, kernel+' disk: tables, models and policy transitions', '$R/h$')
        if kernel in kernels:
            plot_parameter(folder, rows, kernel+' disk: measured truncation versus order-difference estimates', '$R/h$', estimates=True)
        result.append(dict(study='disk', kernel=kernel, rows=rows))
    return result


def box_sweeps(output, device):
    if device is None:
        return []
    from ..scene.scene import BoxRep
    rows_all = []
    register_kernels()
    for kernel in ('w2', 'w4', 'lw2', 'lw4', 'w2p5', 'w4p5', 'cone'):
        folder = output/('box-'+kernel)
        rows = []
        for width in (1e-7, 1e-5, .001, .03, .3, 1., 2.):
            lo, hi = (-.7, -width/2), (.7, width/2)
            vertices = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
            from .adversarial import PolygonCase
            case = PolygonCase('box', 'box', width, vertices)
            points = np.vstack((case.probes(), [[x, y] for x in (0., .0625, .25, .5, 1.) for y in (0., .0625, .25, .5, 1.)]))
            if kernel in ('w2p5', 'w4p5'):
                target = rectangle_reference(lo, hi, points, kernel, 96)
                uncertainty = np.abs(target-rectangle_reference(lo, hi, points, kernel, 48))
            else:
                target = polygon_channels(vertices, points, kernel, dtype=np.longdouble)
                uncertainty = np.full_like(target, np.nan)
            value = scene_channels(BoxRep(lo, hi), points, kernel, device)
            methods = [Measurement('box-lookup', value, target, supported=SUPPORTED)]
            rows.append(save_row(folder/f'width-{width:g}', width, target, uncertainty, methods, points=points))
        plot_parameter(folder, rows, kernel+' box table: thin geometry / exact edges / graded panel joins', 'thickness / h')
        rows_all.append(dict(study='box', kernel=kernel, rows=rows))
    return rows_all


def strip_sweeps(output, kernels):
    from ..edge.tier4 import strip_series
    result = []
    # Coefficients depend on offset, not width: derive each once, then rescale
    # each even transverse derivative term with width^(2k+1).
    distances = np.array([.08, .2, .4, .65, .9, .99])
    for kernel in kernels:
        folder = output/('strip-'+kernel)
        terms = np.array([[float(strip_series(kernel, Fraction(1, 16), Fraction(float(d)), k, dps=35)) for d in distances] for k in range(3)])
        delta = np.diff(np.vstack((np.zeros((1, len(distances))), terms)), axis=0)
        rows = []
        for width in (.001, .01, .03, .1, .3, .7):
            a = width/2
            vertices = np.array([[-2., -a], [2., -a], [2., a], [-2., a]])
            points = np.column_stack((np.zeros(len(distances)), distances))
            target = polygon_channels(vertices, points, kernel, dtype=np.longdouble)
            estimates = np.cumsum(np.array([(a*16)**(2*k+1)*delta[k] for k in range(3)]), axis=0)
            methods = []
            for k in range(3):
                values = np.full_like(target, np.nan, dtype=float)
                values[:, 0] = estimates[k]
                diff = None
                if k:
                    diff = np.full_like(values, np.nan)
                    diff[:, 0] = np.abs(estimates[k]-estimates[k-1])
                methods.append(Measurement(f'T4-strip-order-{k}', values, estimate=diff, supported=(0,),
                                           accepted=distances > a,
                                           note='forced width expansion; intended exterior domain d>a recorded, finite box equals infinite strip on these supports'))
            rows.append(save_row(folder/f'width-{width:g}', width, target, np.full_like(target, np.nan), methods, points=points))
        plot_parameter(folder, rows, kernel+' strip series: transverse thickness / distance to centreline', 'strip thickness / h')
        plot_parameter(folder, rows, kernel+' strip: measured truncation versus order-difference estimates', 'strip thickness / h', estimates=True)
        result.append(dict(study='strip', kernel=kernel, rows=rows))
    return result


def representation_audit(output):
    from .adversarial import PolygonCase
    rows = []
    for offset in (0., 1., 1e4, 1e8):
        for size in (1e-3, 1e-6, 1e-9):
            v = np.array([[-.4, -.3], [.5, -.2], [-.1, .5]], dtype=np.longdouble)*size+np.longdouble(offset)
            x = v.mean(0)[None]
            local = polygon_channels(v-x, np.zeros((1, 2)), 'w4', dtype=np.longdouble)
            for dtype in (np.float32, np.float64):
                vr, xr = v.astype(dtype), x.astype(dtype)
                edge = np.roll(vr, -1, axis=0)-vr
                collapsed = bool((np.linalg.norm(edge, axis=1) == 0).any())
                a, b = (vr[1]-vr[0]).astype(float), (vr[2]-vr[0]).astype(float)
                area = abs(a[0]*b[1]-a[1]*b[0])
                collapsed = bool(collapsed or area == 0)
                represented = None if collapsed else polygon_channels(vr.astype(np.longdouble), xr.astype(np.longdouble), 'w4', dtype=np.longdouble)
                rows.append(dict(offset=offset, size=size, dtype=np.dtype(dtype).name, collapsed=collapsed,
                                 input_geometry_m00_error=None if represented is None else float(abs(represented[0, 0]-local[0, 0])),
                                 local_reference_m00=float(local[0, 0])))
    (output/'representation.json').write_text(json.dumps(rows, indent=2)+'\n')
    return rows


def ellipse_distance(points, a, b):
    """Closest-point ellipse SDF by monotone Lagrange-multiplier bisection.

    Axis points need their own interior branch (the nearest point may be off
    the axis). This is a geometry construction, not the curvature approximation
    under test. a>=b>0; the circle has an exact separate expression.
    """
    p = np.abs(np.asarray(points, dtype=float))
    if a == b:
        return np.linalg.norm(p, axis=1)-a
    if not a > b > 0:
        raise ValueError('ellipse axes require a >= b > 0')
    outside = (p[:, 0]/a)**2+(p[:, 1]/b)**2 >= 1
    low = np.where(outside, 0., -b*b+np.finfo(float).eps*b*b)
    high = np.maximum(a*p[:, 0]+b*p[:, 1], a*a)
    for _ in range(64):
        t = (low+high)/2
        f = (a*p[:, 0]/(t+a*a))**2+(b*p[:, 1]/(t+b*b))**2-1
        low, high = np.where(f > 0, t, low), np.where(f > 0, high, t)
    t = (low+high)/2
    q = np.column_stack((a*a*p[:, 0]/(t+a*a), b*b*p[:, 1]/(t+b*b)))
    axis = (p[:, 1] == 0) & (p[:, 0] < (a*a-b*b)/a)
    q[axis, 0] = a*a*p[axis, 0]/(a*a-b*b)
    q[axis, 1] = b*np.sqrt(np.maximum(1-(q[axis, 0]/a)**2, 0))
    return np.where(outside, 1., -1.)*np.linalg.norm(q-p, axis=1)


def ellipse_sweeps(output, device):
    from .adversarial import PolygonCase, make_sdf
    from .sdf_stress import sampled_rep_model
    from ..scene.scene import SdfRep, SurfaceRep
    folder = output/'variable-curvature-ellipse'
    rows = []
    for ratio in (.2, .5, .75, 1.):
        a, b = 2., 2.*ratio
        t = np.arange(16)*2*np.pi/16
        boundary = np.column_stack((a*np.cos(t), b*np.sin(t)))
        normal = np.column_stack((np.cos(t)/a, np.sin(t)/b))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        points = np.concatenate([boundary+d*normal for d in (-.3, -1e-9, 0., 1e-9, .3)])
        polys = []
        for n in (256, 512, 1024):
            theta = np.arange(n)*2*np.pi/n
            polys.append(np.column_stack((a*np.cos(theta), b*np.sin(theta))))
        case = PolygonCase('ellipse', 'ellipse', ratio, polys[0])
        case.distance = lambda p: ellipse_distance(p, a, b)
        rep = make_sdf(case, points, 1/64)
        target = polygon_channels(polys[2], points, 'w4', dtype=np.longdouble)
        uncertainty = np.abs(target-polygon_channels(polys[1], points, 'w4', dtype=np.longdouble))
        pred = sampled_rep_model(rep, points, 'w4')
        methods = [Measurement('ellipse-SDF-forced', pred['values'], supported=tuple(range(12)),
                                accepted=pred['accepted'], note='true ellipse SDF; varying curvature over support; polygon-reference refinement uncertainty included')]
        if device is not None:
            sdf = SdfRep(rep.d.clone(), rep.origin.clone(), rep.spacing, fallback=SurfaceRep.polygon(polys[0]))
            value = scene_channels(sdf, points, 'w4', device, second=True)
            methods.append(Measurement('ellipse-guarded', value, supported=tuple(range(12)),
                                       note='actual Scene, fallback is a 256-edge ellipse polygon; total includes polygon geometry error'))
            polygon = scene_channels(SurfaceRep.polygon(polys[0]), points, 'w4', device, fused=True)
            model = polygon_channels(polys[0], points, 'w4', dtype=np.longdouble)
            methods.append(Measurement('ellipse-fused-polygon', polygon, model, supported=SUPPORTED))
        rows.append(save_row(folder/f'ratio-{ratio:g}', ratio, target, uncertainty, methods,
                             points=points, ellipse_axes=np.array([a, b]), fallback=pred['fallback']))
    plot_parameter(folder, rows, 'Noncircular smooth curvature: true ellipse SDF versus refined polygon references', 'minor / major semiaxis')
    return [dict(study='ellipse', kernel='w4', reference='1024-edge polygon; 512/1024 chord-refinement estimate, not an analytic ellipse oracle', rows=rows)]


def oracle_spot_checks(output, cases, kernels):
    from ..edge import oracle
    from ..edge.kernels import kernel as get_kernel
    rows = []
    for name in ('triangle-size-1e-05', 'triangle-angle-0.1', 'sliver-1e-07'):
        case = next(c for c in cases if c.name == name)
        point = case.vertices.mean(0)+np.array([.2, .3])
        for kernel in kernels:
            with mp.workdps(45):
                pieces = [(lo, hi, {n-2: -n*c for n, c in enumerate(co) if n and c}) for lo, hi, co in get_kernel(kernel).pieces]
                target = [oracle.polar_moment(case.vertices, point, kernel, al, dps=40) for al in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))]
                for al in ((1, 0), (0, 1), (2, 0), (1, 1), (1, 1), (0, 2)):
                    target.append(oracle.polar_moment_profile(case.vertices, point, al, pieces, dps=40)/mp.pi)
                reference, _, _ = target_reference(case, point[None], kernel)
                # Subtract using mp values, not by rounding both to float64.
                error = [float(abs(mp.mpf(str(a))-b)) for a, b in zip(reference[0], target)]
                rows.append(dict(case=name, kernel=kernel, point=point.tolist(),
                                 max_absolute_reference_error=max(error),
                                 reference_m00_relative_error=error[0]/max(abs(float(target[0])), 1e-300)))
    (output/'independent-oracle-checks.json').write_text(json.dumps(rows, indent=2)+'\n')
    return rows


def run_specialized(output, kernels, device):
    output = Path(output)
    print('Specialized disk tables / curvature / series / tier switches', flush=True)
    result = disk_sweeps(output, kernels, device)
    print('Specialized box tables', flush=True)
    result += box_sweeps(output, device)
    print('Specialized thin-strip expansions', flush=True)
    result += strip_sweeps(output, kernels)
    print('Noncircular smooth curvature variation', flush=True)
    result += ellipse_sweeps(output, device)
    return result
