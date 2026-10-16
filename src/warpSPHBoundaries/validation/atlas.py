"""Offline 2D element atlas: shared channel order, geometry and error budgets.

Run ``python scripts/studies/boundary_atlas.py --help``.  All example coordinates
are in support units (h=1), y=x'-x. Scalar moments are ordered by total degree,
with the x exponent descending, then the kernel-gradient integrals follow:
m00, m10, m01, m20, m11, m02, Gx, Gy, g1xx, g1xy, g1yx, g1yy.
Here g1[d,j]=integral(y_d partial_{x_j} W), NOT partial_{x_j} m_d.

New element adapters should return Evaluation objects in CHANNELS order, and
NaN for unsupported/invalid channels. Preserve the following error split:
numerical = implementation - same_model_reference;
model = same_model_reference - target_geometry_reference;
total = numerical + model (SIGNED, before taking absolute values).
Quadrature refinement differences and reference refinement differences are
empirical estimates, never certified upper bounds or evidence by themselves
that an implementation is within a few eps. Error graphs use a fixed per-channel
scale max(1, max(abs(target))) in dimensionless support units, not pointwise
relative errors at zeros. Raw signed errors and scales are saved in NPZ.

The initial adapters compare a disk's polygon boundary (T1), the SAME polygon
triangulated into a fan (T2), the existing curvature/planar-moment model (T3),
and the shipped disk lookup (T4). T1/T2 polygonization is model error relative
to the disk. T3 is shown only for R/h>=2 and shares curvature corrections across all twelve
moments. Missing second moments of T4 remain visibly unavailable. T4 has no
geometry approximation; its lookup error belongs in numerical evaluation.
"""
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from warpSPHBoundaries.validation.paths import evidence_path
import argparse
import json

import mpmath as mp
import numpy as np

from ..edge import np2d
from ..edge.kernels import kernel as get_kernel


ALPHAS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
CHANNELS = ("m00", "m10", "m01", "m20", "m11", "m02",
            "Gx", "Gy", "g1xx", "g1xy", "g1yx", "g1yy")
LABELS = (r"$m_{00}$", r"$m_{10}/h$", r"$m_{01}/h$", r"$m_{20}/h^2$",
          r"$m_{11}/h^2$", r"$m_{02}/h^2$", r"$hG_x$", r"$hG_y$",
          r"$g_{1,xx}$", r"$g_{1,xy}$", r"$g_{1,yx}$", r"$g_{1,yy}$")
TIERS = ("T1 surface", "T2 volume", "T3 SDF model", "T4 disk lookup")
COLORS = ("#1764ab", "#b55b08", "#228557", "#8d4aa8")
CUT_ANGLE = .17                    # oblique: mixed / odd components are not identically zero
RASTER_BATCH = 512                 # bound memory independently of image sampling


@dataclass
class Evaluation:
    """One evaluator plus a same-model reference, in the shared channel order."""
    name: str
    values: np.ndarray
    model_reference: np.ndarray
    quadrature_estimate: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


def error_budget(evaluation, target):
    """Return signed numerical, model and total errors; keep unavailable NaNs."""
    numerical = evaluation.values - evaluation.model_reference
    model = evaluation.model_reference - target
    return numerical, model, evaluation.values - target


def polygon_channels(vertices, points, kernel="w4", *, triangulated=False,
                     dtype=np.float64, quad=None, stable=None, unsplit=True):
    """All shared channels for a polygon or its fan (convex examples only)."""
    vertices, points = np.asarray(vertices, dtype=dtype), np.asarray(points, dtype=dtype)
    if len(points) > RASTER_BATCH:
        return np.concatenate([polygon_channels(vertices, points[i:i+RASTER_BATCH], kernel,
                               triangulated=triangulated, dtype=dtype, quad=quad, stable=stable, unsplit=unsplit)
                               for i in range(0, len(points), RASTER_BATCH)])
    if triangulated:
        centre = np.broadcast_to(vertices.mean(axis=0), vertices.shape)
        triangles = np.stack([centre, vertices, np.roll(vertices, -1, axis=0)], axis=1)
        verts = np.tile(triangles, (len(points), 1, 1))
        queries = np.repeat(points, len(triangles), axis=0)
    else:
        verts = np.broadcast_to(vertices, (len(points), *vertices.shape))
        queries = points
    columns = [np2d.moment(verts, queries, kernel, alpha, dtype=dtype, quad=quad, stable=stable, unsplit=unsplit) for alpha in ALPHAS]
    for alpha in ((0, 0), (1, 0), (0, 1)):
        # grad_moment's unit-support convention matches this offline atlas.
        g = np2d.grad_moment(verts, queries, kernel, alpha, dtype=dtype, quad=quad, stable=stable, unsplit=unsplit)
        columns.extend((g[:, 0], g[:, 1]))
    out = np.stack(columns, axis=1)
    if triangulated:
        out = out.reshape(len(points), len(triangles), len(CHANNELS)).sum(axis=1)
    return out


def _radial_integral(lo, hi, kernel, power, derivative=False):
    """Exact radial polynomial antiderivatives, evaluated in extended precision."""
    out = np.zeros_like(lo)
    pi = np.arccos(np.longdouble(-1))
    for a, b, coefficients in get_kernel(kernel).pieces:
        left, right = np.maximum(lo, float(a)), np.minimum(hi, float(b))
        valid = right > left
        left, right = np.where(valid, left, 0), np.where(valid, right, 0)
        for n, c in enumerate(coefficients):
            if not c or (derivative and n == 0):
                continue
            exponent = n + power + 1 - int(derivative)
            coefficient = np.longdouble(c.numerator) / np.longdouble(c.denominator)
            if derivative:
                coefficient *= -n                         # -W'/r times polynomial moments
            out += coefficient * (right**exponent-left**exponent) / exponent / pi
    return out


@lru_cache(maxsize=16)
def _reference_rule(nodes):
    """Refine Gauss roots/weights in longdouble; float64 rules limit eps plots."""
    roots = np.polynomial.legendre.leggauss(nodes)[0].astype(np.longdouble)
    for _ in range(4):
        previous, value = np.ones_like(roots), roots.copy()
        for k in range(2, nodes+1):
            previous, value = value, ((2*k-1)*roots*value-(k-1)*previous)/k
        derivative = nodes*(roots*value-previous)/(roots*roots-1)
        roots = roots-value/derivative
    previous, value = np.ones_like(roots), roots.copy()
    for k in range(2, nodes+1):
        previous, value = value, ((2*k-1)*roots*value-(k-1)*previous)/k
    derivative = nodes*(roots*value-previous)/(roots*roots-1)
    weights = 2/((1-roots*roots)*derivative*derivative)
    return roots, weights*(2/weights.sum())


def disk_reference(points, radius, kernel="w4", nodes=64, *, radial_evaluator=None):
    """Independent ray/disk quadrature of all channels (no boundary primitives).

    Polynomial radial integrals are analytic. Angular panels split at every
    kernel-radius intersection. theta=limit*sin(u) removes the tangent square
    root at exterior-ray endpoints. Pair this with nodes*2 as an uncertainty
    estimate, and high-precision oracle spot checks for independent validation.
    """
    if not np.isfinite(radius) or radius <= 0 or nodes < 2:
        raise ValueError("positive radius and at least two quadrature nodes required")
    points = np.asarray(points, dtype=np.longdouble)
    if len(points) > RASTER_BATCH:
        return np.concatenate([disk_reference(points[i:i+RASTER_BATCH], radius, kernel, nodes, radial_evaluator=radial_evaluator)
                               for i in range(0, len(points), RASTER_BATCH)])
    radius = np.longdouble(radius)
    distance = np.sqrt((points*points).sum(axis=1))
    # At the centre every ray is equivalent; avoid overflowing intermediate
    # ratios in the unused exterior/tangency expressions.
    safe = np.where(distance == 0, np.longdouble(1), distance)
    exterior = distance >= radius
    limit = np.where(exterior, np.arcsin(np.minimum(radius/safe, 1)), np.arccos(np.longdouble(-1)))
    breaks = [np.zeros_like(distance), limit,
              np.minimum(np.arccos(np.longdouble(-1))/2, limit)]
    for _, upper, _ in get_kernel(kernel).pieces:
        r = np.longdouble(upper.numerator) / np.longdouble(upper.denominator)
        cosine = np.where(distance == 0, 1, (distance**2+r*r-radius*radius)/(2*safe*r))
        angle = np.arccos(np.clip(cosine, -1, 1))
        breaks.append(np.minimum(angle, limit))
    breaks = np.sort(np.stack(breaks, axis=1), axis=1)
    # A sine map is used only on exterior rays; interior rays have no tangency.
    mapped = np.where(exterior[:, None], np.arcsin(np.clip(breaks/limit[:, None], 0, 1)), breaks)
    gx, gw = _reference_rule(nodes)
    mid, half = (mapped[:, 1:]+mapped[:, :-1])/2, np.diff(mapped, axis=1)/2
    u = mid[..., None]+half[..., None]*gx.astype(np.longdouble)
    theta = np.where(exterior[:, None, None], limit[:, None, None]*np.sin(u), u)
    weights = half[..., None]*gw.astype(np.longdouble)
    weights = weights*np.where(exterior[:, None, None], limit[:, None, None]*np.cos(u), 1)
    cosine, sine = np.cos(theta), np.sin(theta)
    b = distance[:, None, None]*cosine
    disc = np.maximum(radius*radius-distance[:, None, None]**2*sine*sine, 0)
    root = np.sqrt(disc)
    lo, hi = np.maximum(b-root, 0), np.minimum(b+root, 1)
    hi = np.maximum(lo, hi)
    radial_fn = _radial_integral if radial_evaluator is None else radial_evaluator
    radial = [radial_fn(lo, hi, kernel, k) for k in (1, 2, 3)]
    gradient = [radial_fn(lo, hi, kernel, k, True) for k in (1, 2)]
    integrate = lambda f: 2*(weights*f).sum(axis=(1, 2))
    m0 = integrate(radial[0])
    m1 = integrate(cosine*radial[1])
    mnn, mtt = integrate(cosine*cosine*radial[2]), integrate(sine*sine*radial[2])
    g0 = integrate(cosine*gradient[0])
    gnn, gtt = integrate(cosine*cosine*gradient[1]), integrate(sine*sine*gradient[1])
    e = -points/safe[:, None]
    e[distance == 0] = (1, 0)
    ex, ey = e.T
    return np.stack([m0, m1*ex, m1*ey, mtt+(mnn-mtt)*ex*ex,
                     (mnn-mtt)*ex*ey, mtt+(mnn-mtt)*ey*ey,
                     g0*ex, g0*ey, gtt+(gnn-gtt)*ex*ex,
                     (gnn-gtt)*ex*ey, (gnn-gtt)*ex*ey, gtt+(gnn-gtt)*ey*ey], axis=1)


def _disk_lookup(points, radius, kernel):
    from ..edge.disktables import CH, load_disk_tables, PanelTable, DiskTables
    small, large, config = load_disk_tables(kernel)
    table = DiskTables.__new__(DiskTables)
    pu, pr, nu, nr, lpu, lps, lnu, lns = config
    def panel(coef, pa, pb, na, nb):
        p = PanelTable.__new__(PanelTable)
        p.pa, p.pb, p.na, p.nb, p.coef = pa, pb, na, nb, coef
        return p
    table.S = {(b, k): panel(small[3*b+k], pu, pr, nu, nr) for b in (0, 1) for k in range(3)}
    table.L = panel(large, lpu, lps, lnu, lns)
    distance = np.linalg.norm(points, axis=1)
    data = dict(zip(CH, table.channels(distance, np.full(len(points), radius)).T))
    e = -points/np.maximum(distance[:, None], 1e-300)
    e[distance == 0] = (1, 0)
    ex, ey = e.T
    out = np.full((len(points), len(CHANNELS)), np.nan)
    out[:, :3] = np.stack([data['lam'], data['m1x']*ex, data['m1x']*ey], axis=1)
    out[:, 6:8] = data['g0x'][:, None]*e
    delta = data['g1xx']-data['g1yy']
    out[:, 8:] = np.stack([data['g1yy']+delta*ex*ex, delta*ex*ey,
                           delta*ex*ey, data['g1yy']+delta*ey*ey], axis=1)
    return out


def _tier3(points, radius, kernel, exact=False):
    """Shared order-two curvature family; analytic circle jets, all moments."""
    if radius < 2:
        return np.full((len(points), len(CHANNELS)), np.nan)
    import torch
    from ..scene.implicitBodies import Tier3
    from .curvature import model_reference
    distance = np.linalg.norm(points, axis=1)
    q = distance-radius
    normal = points/np.maximum(distance[:, None], 1e-300)
    normal[distance == 0] = (1, 0)
    curvature = np.full(len(points), 1/radius)
    if exact:
        return model_reference(q, normal, curvature, kernel=kernel, order=160)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64)
    return Tier3(kernel, 'cpu').curvature_moments(tensor(q), tensor(normal), tensor(curvature)).numpy()


def disk_study(points, radius, kernel="w4", edges=24, *, references=False):
    """Example adapters. references=True enables expensive same-model checks."""
    angle = 2*np.pi*np.arange(edges)/edges
    polygon = radius*np.stack([np.cos(angle), np.sin(angle)], axis=1)
    # Extended precision keeps float64 errors visible before float conversion.
    target = disk_reference(points, radius, kernel, 96)
    coarse = disk_reference(points, radius, kernel, 48)
    poly_ref = polygon_channels(polygon, points, kernel, dtype=np.longdouble)
    result = []
    for triangulated, name in ((False, TIERS[0]), (True, TIERS[1])):
        values = polygon_channels(polygon, points, kernel, triangulated=triangulated)
        estimate = None
        if references:
            # This diagnostic measures quadrature on the SAME represented solid.
            q8 = polygon_channels(polygon, points, kernel, triangulated=triangulated, quad=8)
            q16 = polygon_channels(polygon, points, kernel, triangulated=triangulated, quad=16)
            estimate = np.abs(q8-q16)
        result.append(Evaluation(name, values, poly_ref, estimate,
                      [f"Inscribed {edges}-gon; numerical reference: extended-precision closed form."]))
    t3 = _tier3(points, radius, kernel)
    t3ref = _tier3(points, radius, kernel, exact=True) if references else np.full_like(t3, np.nan)
    result.append(Evaluation(TIERS[2], t3, t3ref, notes=[
        "Valid here only for R/h >= 2; scalar value has curvature terms through order 2.",
        "Circle distance and curvature are analytic inputs; sampled-SDF reconstruction error is excluded.",
        "All twelve moments share the geometric order-two expansion; curvature variation is zero for these circles.",
        "Same-model reference integrates the shape-Taylor integrands independently of the production tables.",
        "Raster same-model references are omitted; numerical/model budgets use the cut."]))
    lookup = _disk_lookup(points, radius, kernel)
    disk_model = np.array(target, copy=True)
    disk_model[:, 3:6] = np.nan
    result.append(Evaluation(TIERS[3], lookup, disk_model, notes=[
        "Any-radius disk lookup; no geometry/model approximation; lookup error is numerical.",
        "Second scalar moments are not implemented in the disk lookup."]))
    return polygon, target, np.abs(target-coarse), result


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def plot_fields(output, radius, kernel, grid, polygon, evaluations, channels):
    """Same axes, component order and per-channel colour scales across tiers."""
    plt = _pyplot()
    fig, axes = plt.subplots(4, len(channels), figsize=(18, 11), constrained_layout=True)
    theta = np.linspace(0, 2*np.pi, 300)
    for j, channel in enumerate(channels):
        finite = [e.values[:, channel][np.isfinite(e.values[:, channel])] for e in evaluations]
        limit = max([float(np.max(np.abs(a))) for a in finite if len(a)]+[1e-12])
        for i, evaluation in enumerate(evaluations):
            ax = axes[i, j]
            values = evaluation.values[:, channel].reshape(len(grid), len(grid))
            ax.set_facecolor('#ededed')
            im = ax.pcolormesh(grid, grid, values, cmap='RdBu_r', vmin=-limit, vmax=limit, shading='nearest', rasterized=True)
            if not np.isfinite(values).any():
                reason = 'outside validity' if i == 2 and radius < 2 else 'not implemented'
                ax.text(.5, .5, reason, transform=ax.transAxes, ha='center', va='center', color='#555555')
            ax.plot(radius*np.cos(theta)-radius, radius*np.sin(theta), color='#333333', lw=.7)
            cut_r = radius+np.linspace(max(-.9, -.95*radius), .95, 80)
            ax.plot(cut_r*np.cos(CUT_ANGLE)-radius, cut_r*np.sin(CUT_ANGLE),
                    color='#777777', lw=.5, ls=':')
            if i < 2:
                closed = np.vstack([polygon, polygon[:1]])
                ax.plot(closed[:, 0]-radius, closed[:, 1], color='#f0a53a', lw=.8)
                if i == 1:
                    for vertex in polygon:
                        ax.plot([-radius, vertex[0]-radius], [0, vertex[1]],
                                color='#f0a53a', lw=.35, alpha=.5)
            ax.set(xlim=(grid[0], grid[-1]), ylim=(grid[0], grid[-1]), aspect='equal')
            if i == 0:
                ax.set_title(LABELS[channel])
            if j == 0:
                ax.set_ylabel(evaluation.name+'\n$y/h$')
            if i == 3:
                ax.set_xlabel('$(x-R)/h$')
            if i < 3:
                ax.tick_params(labelbottom=False)
            if j:
                ax.tick_params(labelleft=False)
        fig.colorbar(im, ax=axes[:, j], shrink=.55, pad=.01)
    fig.suptitle(f'Disk boundary integrals | {kernel} | R/h = {radius:g}\n'
                 'Black: disk boundary; orange: polygon / volume fan. T3 first moments are planar; grey: unavailable.', fontsize=12)
    fig.savefig(output)
    plt.close(fig)


def plot_errors(output, radius, kernel, cut, target, uncertainty, evaluations, channels, kind):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    scales = np.maximum(1, np.max(np.abs(target), axis=0))
    eps = np.finfo(float).eps
    floor = .01
    for ax, channel in zip(axes.flat, channels):
        any_valid = False
        for i, evaluation in enumerate(evaluations):
            budget = error_budget(evaluation, target)
            error = budget[0 if kind == 'numerical' else 1][:, channel]
            if np.isfinite(error).any():
                any_valid = True
                ax.semilogy(cut, np.maximum(np.abs(error)/(scales[channel]*eps), floor), color=COLORS[i], label=evaluation.name)
            if kind == 'numerical' and evaluation.quadrature_estimate is not None:
                estimate = evaluation.quadrature_estimate[:, channel]/(scales[channel]*eps)
                ax.semilogy(cut, np.maximum(estimate, floor), '--', color=COLORS[i], alpha=.55,
                            label=evaluation.name+' Gauss 8/16 estimate')
        ax.semilogy(cut, np.maximum(uncertainty[:, channel]/(scales[channel]*eps), floor), ':', color='#555555', label='target 48/96-node difference')
        ax.axhline(1, color='#aaaaaa', lw=.7)
        ax.axhline(8, color='#aaaaaa', lw=.7, ls='--')
        ax.set(title=LABELS[channel], xlabel=f'$d/h$; radial cut at {CUT_ANGLE:g} rad', ylabel='absolute error / (scale × float64 eps)')
        ax.grid(alpha=.18)
        if not any_valid:
            ax.text(.5, .5, 'no implemented evaluator', transform=ax.transAxes, ha='center')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=8)
    detail = ('same-model reference; dashed curves estimate separate edge quadrature'
              if kind == 'numerical' else 'same-model reference minus exact disk geometry reference')
    fig.suptitle(f'{kind.capitalize()} error | {kernel} | R/h = {radius:g}\n{detail}\n'
                 'Scale = max(1, max |target channel|); plotted floor 0.01 eps. Refinement differences are not bounds.', fontsize=11)
    fig.savefig(output)
    plt.close(fig)


def plot_quadrature(output, radius, kernel, points, polygon, target, uncertainty):
    """Measured error against same-model closed form AND refinement estimates."""
    plt = _pyplot()
    fig, axes = plt.subplots(4, 3, figsize=(13, 13), constrained_layout=True)
    reference = polygon_channels(polygon, points, kernel, dtype=np.longdouble)
    nodes = (4, 8, 16, 32, 64)
    values = [polygon_channels(polygon, points, kernel, quad=n) for n in nodes]
    stable = [polygon_channels(polygon, points, kernel, stable=(n, 8)) for n in nodes]
    scale = np.maximum(1, np.max(np.abs(reference), axis=0))*np.finfo(float).eps
    for ax, channel in zip(axes.flat, range(len(CHANNELS))):
        measured = [np.max(np.abs(v[:, channel]-reference[:, channel]))/scale[channel] for v in values]
        estimated = [np.max(np.abs(b[:, channel]-a[:, channel]))/scale[channel] for a, b in zip(values[:-1], values[1:])]
        stable_error = [np.max(np.abs(v[:, channel]-reference[:, channel]))/scale[channel] for v in stable]
        stable_estimate = [np.max(np.abs(b[:, channel]-a[:, channel]))/scale[channel] for a, b in zip(stable[:-1], stable[1:])]
        ax.loglog(nodes, np.maximum(measured, .01), 'o-', color=COLORS[0], label='measured vs closed form')
        ax.loglog(nodes[:-1], np.maximum(estimated, .01), 's--', color='#d07810', label='|Q(n) − Q(2n)| estimate')
        ax.loglog(nodes, np.maximum(stable_error, .01), 'o-', color=COLORS[2], label='Chebyshev quadrature, 8 panels')
        ax.loglog(nodes[:-1], np.maximum(stable_estimate, .01), 's--', color=COLORS[3], label='Chebyshev refinement estimate')
        ax.set_xticks(nodes, [str(n) for n in nodes])
        from matplotlib.ticker import NullLocator
        ax.xaxis.set_minor_locator(NullLocator())
        ax.axhline(8, color='#aaaaaa', ls=':')
        ax.set(title=LABELS[channel], xlabel='Gauss nodes per edge panel', ylabel='max error / (scale × float64 eps)')
        ax.grid(alpha=.2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='outside lower center', ncol=2)
    fig.suptitle(f'Quadrature convergence on the same {len(polygon)}-gon | {kernel} | R/h={radius:g}\n'
                 '8 eps guide; floor 0.01 eps. Refinement estimates are not certified error bounds.', fontsize=12)
    fig.savefig(output)
    plt.close(fig)
    return {'nodes': list(nodes), 'max_measured_error': [np.max(np.abs(v-reference), axis=0).astype(float).tolist() for v in values],
            'max_refinement_estimate': [np.max(np.abs(b-a), axis=0).astype(float).tolist() for a, b in zip(values[:-1], values[1:])],
            'chebyshev_panels': 8,
            'max_chebyshev_measured_error': [np.max(np.abs(v-reference), axis=0).astype(float).tolist() for v in stable],
            'max_chebyshev_refinement_estimate': [np.max(np.abs(b-a), axis=0).astype(float).tolist() for a, b in zip(stable[:-1], stable[1:])]}


def model_convergence(output, kernel, edges):
    """Radius sweep of existing value models; higher moments remain in the atlas.

    Successive-order differences are model-error indicators, not remainder
    bounds. Only applicable regimes of the existing policies are plotted.
    """
    from ..edge import tier3, tier4
    plt = _pyplot()
    radii = np.geomspace(.05, 16, 31)
    points = np.column_stack([radii+.3, np.zeros_like(radii)])
    # disk_reference accepts one radius; each sample has its own geometry.
    target = np.array([disk_reference([p], r, kernel, 96)[0, 0] for p, r in zip(points, radii)])
    uncertainty = np.array([abs(disk_reference([p], r, kernel, 48)[0, 0]-v)
                            for p, r, v in zip(points, radii, target)])
    t3 = np.full((3, len(radii)), np.nan)
    t4 = np.full_like(t3, np.nan)
    with mp.workdps(45):
        fk = [tier3.F_k(kernel, Fraction(3, 10), k, dps=40) for k in range(3)]
        for j, r in enumerate(radii):
            if r >= 2:
                for order in range(3):
                    t3[order, j] = float(sum(fk[k]/mp.mpf(float(r))**k for k in range(order+1)))
            if r <= .2:
                for order in range(3):
                    try:
                        t4[order, j] = float(tier4.disk_series(kernel, Fraction(float(r)),
                                                            Fraction(float(r))+Fraction(3, 10), order, dps=40))
                    except ValueError:                       # a kernel knot/rim crosses the disk
                        pass
    theta = 2*np.pi*np.arange(edges)/edges
    vertices = radii[:, None, None]*np.stack([np.cos(theta), np.sin(theta)], axis=1)[None]
    polygon = np2d.value(vertices, points, kernel, dtype=np.longdouble)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, values, name, color in zip(axes, (t4, t3), ('T4 small-disk series', 'T3 curvature expansion'), (COLORS[3], COLORS[2])):
        for order, style in enumerate((':', '--', '-')):
            ax.loglog(radii, np.maximum(np.abs(values[order]-target), 1e-18), style, color=color,
                      label=f'{name}, order {order}')
        ax.loglog(radii, np.maximum(np.abs(values[2]-values[1]), 1e-18), '-.', color='#b76a12',
                  label='|order 2 − order 1| estimate')
        ax.loglog(radii, np.maximum(np.abs(polygon-target), 1e-18), color='#3974a4', alpha=.65,
                  label=f'inscribed {edges}-gon model error')
        ax.loglog(radii, np.maximum(uncertainty, 1e-18), color='#777777', ls=':', label='reference refinement estimate')
        ax.axhline(8*np.finfo(float).eps, color='#aaaaaa', ls='--', label='8 float64 eps')
        ax.axvspan(.2, 2, color='#eeeeee')
        ax.set(xlabel='$R/h$', ylabel='absolute error of $m_{00}$', ylim=(1e-18, 1), xlim=(radii[0], radii[-1]))
        ax.grid(alpha=.15)
        ax.legend(fontsize=7, loc='lower right')
    fig.suptitle(f'Model truncation at d/h=0.3 | {kernel}\n'
                 'Shown in existing policy regimes; grey band uses fallback geometry. Estimates are not certified bounds.', fontsize=12)
    fig.savefig(Path(output)/'model-convergence.png')
    plt.close(fig)
    np.savez_compressed(Path(output)/'model-convergence.npz', radius=radii, target=target,
                        reference_estimate=uncertainty, T3=t3, T4=t4, polygon=polygon)


def run(output, radii=(.2, .7, 4.), kernel='w4', grid_size=129, cut_size=129, edges=24,
        dpi=300, include_stress=True, sdf_spacing=1/64):
    if grid_size < 3 or cut_size < 3 or edges < 3 or dpi < 72 or any(not np.isfinite(r) or r <= 0 for r in radii):
        raise ValueError('grid/cut sizes and edge count must be >=3, dpi >=72; radii must be positive')
    if not np.isfinite(sdf_spacing) or sdf_spacing <= 0:
        raise ValueError('SDF spacing must be positive and finite')
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        raise RuntimeError('the numerical budget requires extended precision beyond float64')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    plt.rcParams['savefig.dpi'] = dpi
    grid = np.linspace(-1.1, 1.1, grid_size)
    gx, gy = np.meshgrid(grid, grid)
    manifest = {'schema_version': 1, 'kernel': kernel, 'support': 1., 'channels': CHANNELS,
                'moment_exponents': ALPHAS, 'tiers': TIERS, 'float64_eps': np.finfo(float).eps,
                'reference_dtype': str(np.dtype(np.longdouble)), 'reference_eps': float(np.finfo(np.longdouble).eps),
                'cut_angle_radians': CUT_ANGLE,
                'error_estimates_are_certified_bounds': False,
                'grid_size': grid_size, 'cut_size': cut_size, 'export_dpi': dpi,
                'cut_error_scale': 'max(1, max(abs(target channel))) in support units; not pointwise relative error',
                'model_convergence': {'figure': 'model-convergence.png', 'data': 'model-convergence.npz',
                                      'quantity': 'm00 only: the existing T4 series has no higher-moment implementation'},
                'cases': []}
    for radius in radii:
        print(f'Building atlas R/h={radius:g}', flush=True)
        folder = output/f'radius-{radius:g}'
        folder.mkdir(exist_ok=True)
        points = np.column_stack([gx.ravel()+radius, gy.ravel()])
        polygon, target, uncertainty, evaluations = disk_study(points, radius, kernel, edges)
        # The oblique cut exercises odd and mixed channels as well as diagonals;
        # points lie on both sides of the surface. Our polar reference is valid
        # at D=R too (the boundary-quadrature reference is not used).
        cut = np.linspace(max(-.9, -.95*radius), .95, cut_size)
        cut_points = (radius+cut)[:, None]*np.array([np.cos(CUT_ANGLE), np.sin(CUT_ANGLE)])
        _, cut_target, cut_uncertainty, cut_evaluations = disk_study(cut_points, radius, kernel, edges, references=True)
        arrays = {'grid': grid, 'points': points, 'target': target, 'reference_estimate': uncertainty,
                  'cut': cut, 'cut_points': cut_points, 'cut_target': cut_target, 'cut_reference_estimate': cut_uncertainty,
                  'channel_scale': np.maximum(1, np.max(np.abs(cut_target), axis=0))}
        for i, (evaluation, cut_eval) in enumerate(zip(evaluations, cut_evaluations), 1):
            arrays[f'T{i}_values'] = evaluation.values
            arrays[f'T{i}_cut_values'] = cut_eval.values
            arrays[f'T{i}_cut_model_reference'] = cut_eval.model_reference
            for label, error in zip(('numerical', 'model', 'total'), error_budget(cut_eval, cut_target)):
                arrays[f'T{i}_{label}_error'] = error
            if cut_eval.quadrature_estimate is not None:
                arrays[f'T{i}_quadrature_estimate'] = cut_eval.quadrature_estimate
        np.savez_compressed(folder/'data.npz', **arrays)
        for group, channels in [('moments', range(6)), ('gradient-moments', range(6, 12))]:
            plot_fields(folder/f'{group}.png', radius, kernel, grid, polygon, evaluations, channels)
            for kind in ('numerical', 'model'):
                plot_errors(folder/f'{group}-{kind}-error.png', radius, kernel, cut, cut_target,
                            cut_uncertainty, cut_evaluations, channels, kind)
        convergence = plot_quadrature(folder/'quadrature-convergence.png', radius, kernel, cut_points,
                                     polygon, cut_target, cut_uncertainty)
        case = {'radius_over_h': radius, 'polygon_edges': edges, 'directory': folder.name,
                'evaluators': [{'name': e.name, 'notes': e.notes,
                                'available_channels': [CHANNELS[j] for j in range(12) if np.isfinite(e.values[:, j]).any()]}
                               for e in cut_evaluations], 'quadrature': convergence}
        manifest['cases'].append(case)
    model_convergence(output, kernel, edges)
    if include_stress:
        from .sdf_stress import run_stress
        manifest['sdf_stress'] = run_stress(output/'sdf-stress', kernel, grid_size, cut_size, sdf_spacing)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', default=evidence_path('boundary-atlas'))
    parser.add_argument('--radii', type=float, nargs='+', default=[.2, .7, 4.])
    parser.add_argument('--kernel', choices=['cubic', 'w2', 'w4', 'w6'], default='w4')
    parser.add_argument('--grid', type=int, default=129, help='spatial samples per axis (not image DPI)')
    parser.add_argument('--cut', type=int, default=129, help='samples along each diagnostic cut')
    parser.add_argument('--dpi', type=int, default=300, help='PNG export resolution')
    parser.add_argument('--sdf-spacing', type=float, default=1/64, help='sampled SDF spacing in support units')
    parser.add_argument('--skip-stress', action='store_true', help='omit deliberate SDF breakdown cases')
    parser.add_argument('--edges', type=int, default=24)
    args = parser.parse_args()
    run(args.output, args.radii, args.kernel, args.grid, args.cut, args.edges,
        args.dpi, not args.skip_stress, args.sdf_spacing)


if __name__ == '__main__':
    main()
