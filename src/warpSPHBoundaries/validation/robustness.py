"""Adversarial audit of 2D boundary approximations and their executed fallbacks.

python -m warpSPHBoundaries.validation.robustness --device cuda:0

The default run covers all current approximation families, four base kernels,
and the shipped specialized-kernel tables. Raw failures are retained; error
colours classify all values >1 as unusable. Nonfinite results and fallbacks
are reported explicitly. 2D coverage is finite and explicit in coverage.json.
"""
import argparse
import json
from pathlib import Path

from warpSPHBoundaries.validation.paths import evidence_path

import numpy as np

from .atlas import CHANNELS, LABELS, _pyplot, polygon_channels
from .adversarial import (Measurement, SUPPORTED, polygon_cases, target_reference,
                          measured_budget, report, fem_channels, make_sdf,
                          contour_reference, scene_channels)
from .sdf_stress import sampled_rep_model


FAMILIES = {
    'edge arithmetic': 'closed-f64, diagnostic split-f64, closed-f32',
    'edge quadrature': 'Gauss 4/8/16; Chebyshev 8/16 nodes in float64 and 8 nodes in float32',
    'volume / FEM': 'P2 edge weights and actual far-field hybrid at 8/16 nodes',
    'SDF curvature': 'forced sampled T3 and actual guarded Scene with explicit polygon fallback',
    'SDF distance-only': 'curvature-free half-plane closed-form reference, twelve moments with explicit unresolved normals',
    'SDF contour geometry': 'actual marching-squares contour versus original polygon, including lost features',
    'production edge evaluation': 'actual fused monomial and Chebyshev paths',
    'boundary particles': 'actual ParticleBoundary lattices at h/16 and h/32',
    'disk lookup': 'all shipped disk tables: small/large radii, panel joins, origin and tangencies',
    'box lookup': 'box tables: tiny/thin boxes, boundary points, corners and panel joins',
    'T3 circle / cavity model': 'curvature expansion and planar moments on both sides of solid and cavity',
    'T4 disk series': 'orders 0/1/2, kernel knots and support rim, actual validity decisions',
    'T4 strip series': 'orders 0/1/2, thickness/offset sweep against finite-box closed form',
    'polygon geometry approximation': 'area-preserving circle/cavity polygons at 12/24/48 edges',
    'implicit tier switches': 'actual Scene around R/h=0.2 and 2, both sides and cavity orientation',
    'input representation': 'float32 triangle collapse and translation stress; represented geometry distinguished',
    'noncircular curvature': 'true ellipse closest-point SDF; 256/512/1024-edge reference refinement, with stated geometry uncertainty',
    'circle arc quadrature': 'actual disk_channels at 24/48 nodes, including query exactly on the circle',
}


def save_measurements(folder, target, uncertainty, measurements, **arrays):
    payload = dict(target=target, reference_estimate=uncertainty, **arrays)
    summaries = []
    for m in measurements:
        payload[m.name] = m.values
        for key, values in zip(('numerical', 'model', 'total'), measured_budget(m, target)):
            payload[m.name+'_'+key+'_error'] = values
        if m.status is not None:
            payload[m.name+'_status'] = m.status
        if m.estimate is not None:
            payload[m.name+'_estimate'] = m.estimate
        if m.accepted is not None:
            payload[m.name+'_accepted'] = m.accepted
        if m.domain is not None:
            payload[m.name+'_domain'] = m.domain
        summaries.append(report(m, target, uncertainty))
    np.savez_compressed(folder/'data.npz', **payload)
    return summaries


def polygon_measurements(case, points, target, rep, kernel, device):
    from ..scene.scene import SurfaceRep, SdfRep, Body, Scene
    v = case.vertices
    methods = []
    for name, options in (
        ('closed-f64', {}), ('split-f64', {'unsplit': False}),
        ('closed-f32', {'dtype': np.float32}),
        ('gauss-4', {'quad': 4}), ('gauss-8', {'quad': 8}),
        ('cheb-f64', {'stable': (8, 6)}),
        ('cheb-f32', {'dtype': np.float32, 'stable': (8, 6)})):
        vr = v.astype(options.get('dtype', float))
        edge = np.roll(vr, -1, axis=0)-vr
        if np.any(np.linalg.norm(edge, axis=1) == 0):
            value = np.full_like(target, np.nan, dtype=float)
            note = 'input vertices collapse in the selected dtype; method not representable'
        else:
            value = polygon_channels(v, points, kernel, **options)
            note = 'diagnostic legacy split form' if name == 'split-f64' else ''
        estimate = None
        if name == 'gauss-8':
            estimate = value-polygon_channels(v, points, kernel, quad=16)
        elif name == 'cheb-f64':
            estimate = value-polygon_channels(v, points, kernel, stable=(16, 6))
        methods.append(Measurement(name, value, target, estimate, note=note))
    if len(v) == 3:
        for name, hybrid, n in (('P2-edge', False, 8), ('P2-hybrid', True, 8)):
            value = fem_channels(v, points, kernel, hybrid=hybrid, nodes=n)
            estimate = value-fem_channels(v, points, kernel, hybrid=True, nodes=16) if hybrid else None
            methods.append(Measurement(name, value, target, estimate))
    prediction = sampled_rep_model(rep, points, kernel)
    from .distance_only import sampled_distance_only
    distance_only = sampled_distance_only(rep, points, kernel)
    methods.append(Measurement('SDF-distance-only', distance_only['values'], distance_only['model_reference'],
                               note='Distance/normal-only half-plane closed form, all twelve moments; no curvature, local-model selection or fallback. Undefined sampled normals leave oriented channels unavailable. Model budget includes SDF reconstruction and half-plane geometry bias.'))
    methods.append(Measurement('SDF-forced', prediction['values'], supported=tuple(range(12)),
                               accepted=prediction['accepted'],
                               note='TOTAL: sampling + shared order-two curvature/variation model + tables; all twelve moments, invalid extrapolations included'))
    # A missing contour is an explicit geometric failure. Keep its zero field
    # and error against the original feature, rather than dropping the case.
    lost = not bool((rep.d < 0).any())
    contour = np.zeros_like(target) if lost else contour_reference(rep, points, kernel)
    methods.append(Measurement('SDF-contour', contour, contour,
                               note='geometry lost: no solid grid nodes' if lost else 'exact integral of the sampled contour; model error versus original polygon'))
    if device is not None:
        for name, stable in (('fused-mono', False), ('fused-cheb', True)):
            value,estimate,status = scene_channels(SurfaceRep.polygon(v),points,kernel,device,
                fused=True,stable=stable,second=True,return_diagnostics=True)
            methods.append(Measurement(name,value,target,estimate=estimate,status=status,
                note='Current executed surface backend; all twelve channels, conditioned device triangle route'))
        sdf = SdfRep(rep.d.clone(), rep.origin.clone(), rep.spacing, fallback=SurfaceRep.polygon(v))
        value = scene_channels(sdf, points, kernel, device, second=True)
        methods.append(Measurement('SDF-guarded', value, supported=tuple(range(12)),
                                   note='actual Scene precompute with explicit polygon fallback; TOTAL discrepancy'))
        if kernel in ('w2', 'w4'):
            from .adversarial import particle_channels
            for spacing in (1/16, 1/32):
                value = particle_channels(Scene([Body(bodyId=0, reps=[SurfaceRep.polygon(v)])], device), points, kernel, spacing)
                methods.append(Measurement(f'particles-{int(1/spacing)}', value,
                                           supported=(0, 6, 7, 8, 9, 10, 11),
                                           note='actual lattice quadrature; unsupported moments remain NaN'))
    return methods, prediction, lost


def plot_overview(output, records):
    from .error_colors import error_style, error_colorbar, BAND_LEGEND
    from .audit_report import scheme_info
    plt = _pyplot()
    names = list(dict.fromkeys(m['name'] for r in records for m in r['methods']))
    cases = list(dict.fromkeys(r['case'] for r in records))
    matrix = np.full((2, len(names), len(cases)), np.nan)
    failures = np.zeros((len(names), len(cases)), bool)
    for r in records:
        for m in r['methods']:
            i, j = names.index(m['name']), cases.index(r['case'])
            failures[i, j] |= m['nonfinite_implemented_values'] > 0
            for k, key in enumerate(('max_absolute_error', 'max_error_over_signal_scale')):
                values = [v for v in m[key] if v is not None]
                if values:
                    old = matrix[k, i, j]
                    matrix[k, i, j] = max(0 if np.isnan(old) else old, max(values))
    for k, suffix in enumerate(('absolute', 'relative')):
        fig, ax = plt.subplots(figsize=(18, 7.5), constrained_layout=True)
        a = matrix[k]
        im = ax.imshow(np.ma.masked_invalid(a), aspect='auto', **error_style())
        ax.set_facecolor('#dddddd')
        ax.set_yticks(range(len(names)), [n+' ['+scheme_info(n)[0]+']' for n in names], fontsize=8)
        ax.set_xticks(range(len(cases)), cases, rotation=40, ha='right', fontsize=7)
        yy, xx = np.where(failures)
        ax.scatter(xx, yy, marker='x', c='#00b9d1', s=18)
        ax.set_title(('Maximum absolute error across channels/kernels',
                      'Maximum error / per-case channel signal scale (tiny contributions stay visible)')[k])
        error_colorbar(fig, im, ax=ax, shrink=.8, detailed=True)
        fig.suptitle('Adversarial 2D approximation audit | cyan ×: nonfinite / unrepresentable result\n'
                     'Bracketed labels: expected regime, not an accuracy verdict. Grey: unavailable / not run.\n'+BAND_LEGEND, fontsize=11)
        fig.savefig(output/f'overview-{suffix}.png')
        plt.close(fig)


def plot_family(output, records, family, kernel, estimates=False):
    plt = _pyplot()
    selected = sorted([r for r in records if r['family'] == family and r['kernel'] == kernel], key=lambda r: r['parameter'])
    names = list(dict.fromkeys(m['name'] for r in selected for m in r['methods']))
    palette = {name: plt.get_cmap('tab20')(i % 20) for i, name in enumerate(names)}
    if estimates:
        names = [n for n in names if n in ('gauss-8', 'cheb-f64', 'P2-hybrid')]
    fig, axes = plt.subplots(2, 6, figsize=(18, 8), constrained_layout=True)
    x = [r['parameter'] for r in selected]
    for i, ax in enumerate(axes.flat):
        for name in names:
            y = [next((m['max_error_over_signal_scale'][i] for m in r['methods'] if m['name'] == name), None) for r in selected]
            if any(v is not None for v in y):
                line, = ax.plot(x, [np.nan if v is None else max(v, 1e-18) for v in y], '.-', color=palette[name], label=name, lw=.9, ms=3)
                if estimates:
                    refined = []
                    for r in selected:
                        m = next((m for m in r['methods'] if m['name'] == name), None)
                        e = None if m is None else m.get('max_refinement_estimate', [None]*12)[i]
                        refined.append(np.nan if e is None else max(e/m['signal_scale'][i], 1e-18))
                    ax.plot(x, refined, '--', color=line.get_color(), label=name+' refinement estimate', lw=.9)
        if estimates:
            reference = []
            for r in selected:
                m = r['methods'][0]
                e = m.get('max_reference_estimate', [None]*12)[i]
                reference.append(np.nan if e is None else max(e/m['signal_scale'][i], 1e-18))
            ax.plot(x, reference, ':', color='#333333', label='reference refinement estimate', lw=.9)
        ax.set(xscale='log', yscale='log', title=LABELS[i], xlabel={'size': 'triangle size / h', 'angle': 'tip angle (degrees)', 'aspect': 'thickness / length'}[family])
        ax.axhline(8*np.finfo(float).eps, color='#777777', ls=':', lw=.7)
        ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=6, fontsize=8)
    fig.suptitle(f'{kernel} | {family}: error relative to channel signal, maximized over targeted probes\n'
                 'Dashed refinement differences are NOT bounds. Dotted horizontal: 8 float64 eps; this is a target, not a claimed reference accuracy.', fontsize=12)
    fig.savefig(output/f'{family}-{kernel}-{"estimates" if estimates else "errors"}.png')
    plt.close(fig)


def plot_spatial(folder, case, kernel, grid_size, spacing, device):
    from ..scene.scene import SurfaceRep, SdfRep
    plt = _pyplot()
    grid = np.linspace(-1.2, 1.2, grid_size)
    x, y = np.meshgrid(grid, grid)
    points = np.column_stack((x.ravel(), y.ravel()))
    target = polygon_channels(case.vertices, points, kernel, dtype=np.longdouble)
    rep = make_sdf(case, points, spacing)
    raw = sampled_rep_model(rep, points, kernel)
    data = [('Forced SDF TOTAL error', raw['values']),
            ('Gauss 8 numerical error', polygon_channels(case.vertices, points, kernel, quad=8))]
    if device is not None:
        sdf = SdfRep(rep.d.clone(), rep.origin.clone(), spacing, fallback=SurfaceRep.polygon(case.vertices))
        data += [('Actual guarded SDF TOTAL error', scene_channels(sdf, points, kernel, device)),
                 ('Actual fused polygon numerical error', scene_channels(SurfaceRep.polygon(case.vertices), points, kernel, device, fused=True,second=True))]
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'maps.npz', grid=grid, points=points, target=target,
                        forced=raw['values'], fallback=raw['fallback'],
                        **{f'method_{i}': a for i, (_, a) in enumerate(data)})
    render_spatial(folder, case, kernel, grid, target, data)


def render_spatial(folder, case, kernel, grid, target, data):
    """Render cached fields without rerunning geometry or production evaluators."""
    from .error_colors import error_style, error_colorbar, BAND_LEGEND
    plt = _pyplot()
    grid_size = len(grid)
    for group in (range(6), range(6, 12)):
        fig, axes = plt.subplots(1+len(data), 6, figsize=(18, 3*(1+len(data))), constrained_layout=True)
        outline = np.vstack((case.vertices, case.vertices[0]))
        for col, channel in enumerate(group):
            for row, (name, values) in enumerate([('Target geometry reference', target)]+data):
                ax = axes[row, col]
                a = values[:, channel] if row == 0 else np.abs(values[:, channel]-target[:, channel])
                ax.set_facecolor('#dddddd')
                if np.isfinite(a).any():
                    if row == 0:
                        lim = max(float(np.max(np.abs(a))), 1e-30)
                        im = ax.pcolormesh(grid, grid, a.reshape(grid_size, grid_size), shading='nearest', cmap='RdBu_r', vmin=-lim, vmax=lim)
                    else:
                        im = ax.pcolormesh(grid, grid, np.ma.masked_invalid(a).reshape(grid_size, grid_size),
                                           shading='nearest', **error_style())
                    if row:
                        error_colorbar(fig, im, ax=ax, shrink=.65)
                    else:
                        fig.colorbar(im, ax=ax, shrink=.65)
                else:
                    ax.text(.5, .5, 'unavailable', ha='center', transform=ax.transAxes)
                ax.plot(*outline.T, color='#333333', lw=.6)
                ax.set(xlim=(grid[0], grid[-1]), ylim=(grid[0], grid[-1]), aspect='equal')
                if row == 0:
                    ax.set_title(LABELS[channel])
                if col == 0:
                    ax.set_ylabel(name+'\n$y/h$', fontsize=8)
                if row == len(data):
                    ax.set_xlabel('$x/h$')
        fig.suptitle(f'{case.name} | {kernel} | Grey: unsupported / undefined. Error scales shared across panels.\n'+BAND_LEGEND, fontsize=11)
        fig.savefig(folder/('moments.png' if group.start == 0 else 'gradient-moments.png'))
        plt.close(fig)


def replot(output=evidence_path('robustness'), dpi=None):
    """Refresh error heatmaps from saved data; keep arrays and metrics intact."""
    output = Path(output)
    summary = json.loads((output/'summary.json').read_text())
    _pyplot().rcParams['savefig.dpi'] = summary['dpi'] if dpi is None else dpi
    for case in polygon_cases():
        folder = output/'maps'/case.name
        if not (folder/'maps.npz').exists():
            continue
        with np.load(folder/'maps.npz') as arrays:
            names = ['Forced SDF TOTAL error', 'Gauss 8 numerical error',
                     'Actual guarded SDF TOTAL error', 'Actual fused polygon numerical error']
            data = [(name, arrays[f'method_{i}']) for i, name in enumerate(names) if f'method_{i}' in arrays]
            render_spatial(folder, case, 'w4', arrays['grid'], arrays['target'], data)
    from .audit_report import generate
    generate(output, dpi)


def collect_findings(summary, output):
    """Produce actionable evidence, separately from successful audit execution.

    The 1e-10 marker only identifies gross numerical defects in exact methods;
    it does not replace the user's few-eps target. All errors and signal scales
    remain available so a weaker marker cannot be mistaken for certification.
    """
    output = Path(output)
    failures, lost, false_acceptance, arc_failures = [], [], [], []
    exact = {'closed-f64', 'fused-mono', 'fused-cheb', 'P2-edge', 'P2-hybrid', 'scene-device', 'fused-device-mono', 'fused-device-cheb'}
    for row in summary['records']:
        if row.get('lost_sdf_feature', False):
            lost.append(dict(case=row['case'], kernel=row['kernel']))
        for m in row['methods']:
            if m.get('accepted_m00_errors_over_0_01', 0):
                false_acceptance.append(dict(case=row['case'], kernel=row['kernel'], method=m['name'],
                                             queries=m['accepted_m00_errors_over_0_01']))
            maximum = max((x for x in m['max_absolute_error'] if x is not None), default=0)
            if m['name'] in exact and (maximum > 1e-10 or m['nonfinite_implemented_values']):
                data = np.load(output/row['kernel']/row['case']/'data.npz')
                error = np.abs(data[m['name']]-data['target'])
                if not np.isfinite(error).any():
                    failures.append(dict(case=row['case'], kernel=row['kernel'], method=m['name'],
                                         max_absolute_error=None, nonfinite_implemented_values=m['nonfinite_implemented_values']))
                    data.close()
                    continue
                i, j = np.unravel_index(np.nanargmax(error), error.shape)
                failures.append(dict(case=row['case'], kernel=row['kernel'], method=m['name'],
                                     max_absolute_error=float(error[i, j]), channel=CHANNELS[j],
                                     query=data['points'][i].tolist(), query_index=int(i),
                                     actual=float(data[m['name']][i, j]), target=float(data['target'][i, j]),
                                     data=str(Path(row['kernel'])/row['case']/'data.npz')))
                data.close()
    for study in summary['specialized']:
        for row in study['rows']:
            for m in row['methods']:
                if m['name'] == 'arc-24':
                    maximum = max((x for x in m['max_absolute_error'] if x is not None), default=0)
                    if maximum > 1e-10:
                        arc_failures.append(dict(kernel=study['kernel'], radius=row['parameter'],
                                                 max_absolute_error=maximum,
                                                 data=str(Path('specialized')/('disk-'+study['kernel'])/f"R-{row['parameter']:.12g}"/'data.npz')))
                if m['name'] == 'ellipse-SDF-forced' and m.get('accepted_m00_errors_over_0_01', 0):
                    false_acceptance.append(dict(study=study['study'], parameter=row['parameter'], kernel=study['kernel'],
                                                 method=m['name'], queries=m['accepted_m00_errors_over_0_01']))
    result = dict(audit_execution_complete=True,
                  approximations_universally_validated=False,
                  gross_numerical_failure_marker=1e-10,
                  few_eps_certification=False,
                  exact_path_accuracy_failures=sorted(failures, key=lambda a: a['max_absolute_error'] or 0, reverse=True),
                  lost_sdf_geometry=lost, accepted_local_models_with_m00_error_over_0_01=false_acceptance,
                  circle_arc_quadrature_accuracy_failures=arc_failures,
                  accuracy_regressions='tests/validation/test_robustness.py and tests/scene/test_boundary_predicates.py: ordinary accuracy gates for sloped triangle edges and circle-boundary quadrature',
                  reference_checks=summary['oracle'])
    (output/'findings.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def coverage_status(summary):
    """Execution coverage is distinct from accuracy validation."""
    names = {m['name'] for r in summary['records'] for m in r['methods']}
    names |= {m['name'] for s in summary['specialized'] for r in s['rows'] for m in r['methods']}
    prefixes = {'edge arithmetic': 'closed-', 'edge quadrature': 'gauss-',
                'volume / FEM': 'P2-', 'SDF curvature': 'SDF-forced',
                'SDF distance-only': 'SDF-distance-only',
                'SDF contour geometry': 'SDF-contour', 'production edge evaluation': 'fused-',
                'boundary particles': 'particles-', 'disk lookup': 'disk-lookup',
                'box lookup': 'box-lookup', 'T3 circle / cavity model': 'T3-circle-',
                'T4 disk series': 'T4-disk-', 'T4 strip series': 'T4-strip-',
                'polygon geometry approximation': 'polygon-', 'implicit tier switches': 'actual-tier-switch',
                'input representation': '', 'noncircular curvature': 'ellipse-SDF-'}
    prefixes['circle arc quadrature'] = 'arc-'
    return {family: ('executed; inspect accuracy findings' if
                     (bool(summary['representation']) if family == 'input representation' else
                      any(n.startswith(prefixes[family]) for n in names)) else 'not run')
            for family in FAMILIES}


def run(output=evidence_path('robustness'), kernels=('cubic', 'w2', 'w4', 'w6'),
        device='cuda:0', grid_size=129, dpi=300, spacing=1/64, spatial=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _pyplot().rcParams['savefig.dpi'] = dpi
    records = []
    cases = polygon_cases()
    for case in cases:
        print(f'Adversarial geometry: {case.name}', flush=True)
        points = case.probes()
        rep = make_sdf(case, points, spacing)
        for kernel in kernels:
            folder = output/kernel/case.name
            folder.mkdir(parents=True, exist_ok=True)
            target, uncertainty, reference = target_reference(case, points, kernel)
            measurements, prediction, lost = polygon_measurements(case, points, target, rep, kernel, device)
            methods = save_measurements(folder, target, uncertainty, measurements,
                                        points=points, vertices=case.vertices,
                                        fallback=prediction['fallback'], accepted=prediction['accepted'])
            records.append(dict(case=case.name, family=case.family, parameter=case.parameter,
                                kernel=kernel, queries=len(points), reference=reference,
                                lost_sdf_feature=lost, methods=methods))
    for family in ('size', 'angle', 'aspect'):
        for kernel in kernels:
            plot_family(output, records, family, kernel)
            plot_family(output, records, family, kernel, estimates=True)
    if spatial:
        for name in ('triangle-size-0.001', 'triangle-angle-3', 'sliver-0.001', 'acute-notch'):
            case = next(c for c in cases if c.name == name)
            print(f'Spatial audit: {name}', flush=True)
            plot_spatial(output/'maps'/name, case, 'w4', grid_size, spacing, device)
    from .specialized_audit import run_specialized, representation_audit, oracle_spot_checks
    specialized = run_specialized(output/'specialized', kernels, device)
    representation = representation_audit(output)
    oracle = oracle_spot_checks(output, cases, kernels)
    summary = dict(schema_version=1, channels=CHANNELS, support=1., kernels=kernels,
                   grid_size=grid_size, dpi=dpi, sdf_spacing=spacing, device=device,
                   estimates_are_certified_bounds=False, production_policy_changed=False,
                   records=records, specialized=specialized, representation=representation, oracle=oracle)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    collect_findings(summary, output)
    coverage = dict(approximation_families=FAMILIES, base_kernels=kernels,
                    execution_by_family=coverage_status(summary),
                    production_paths_executed=device is not None,
                    geometry_families=['tiny triangles', 'acute/obtuse corners', 'slivers', 'reflex and acute notches',
                                       'tiny/thin boxes', 'circles/cavities', 'thin strips', 'smooth ellipses with varying curvature',
                                       'existing SDF thin-strip/narrow-gap stress cases'],
                    explicit_limits=['finite deterministic 2D coverage, not a proof for every geometry',
                                     '3D surface/tetrahedral tiers and general T4 tube solvers are not implemented',
                                     'noncircular smooth curvature uses refined polygon references with geometry uncertainty, not a fully independent analytic ellipse oracle',
                                     'no certified a priori bounds; reference/refinement uncertainty is empirical'],
                    existing_stress_command='python scripts/studies/boundary_atlas.py',
                    repeat_command='python -m warpSPHBoundaries.validation.robustness --device cuda:0')
    (output/'coverage.json').write_text(json.dumps(coverage, indent=2)+'\n')
    from .audit_report import generate
    generate(output, dpi)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=evidence_path('robustness'))
    p.add_argument('--kernels', nargs='+', choices=['cubic', 'w2', 'w4', 'w6'], default=['cubic', 'w2', 'w4', 'w6'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--skip-production', action='store_true', help='CPU mathematical study only; coverage explicitly reports production paths not run')
    p.add_argument('--skip-maps', action='store_true')
    p.add_argument('--replot', action='store_true', help='refresh error heatmaps from saved arrays; no numerical evaluations')
    p.add_argument('--grid', type=int, default=129)
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--sdf-spacing', type=float, default=1/64)
    a = p.parse_args()
    if a.replot:
        replot(a.output, a.dpi)
    else:
        run(a.output, a.kernels, None if a.skip_production else a.device, a.grid, a.dpi, a.sdf_spacing, not a.skip_maps)


if __name__ == '__main__':
    main()
