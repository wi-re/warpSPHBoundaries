"""Curvature-free semi-analytic SDF reference, original geometry and error budgets."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from warpSPHBoundaries.validation.paths import evidence_path

import numpy as np
import torch

from warpSPHBoundaries.scene.scene import SdfRep
from warpSPHBoundaries.validation.adversarial import Measurement, SUPPORTED, make_sdf, polygon_cases
from warpSPHBoundaries.validation.atlas import CHANNELS, LABELS, _pyplot
from warpSPHBoundaries.validation.audit_report import generate
from warpSPHBoundaries.validation.distance_only import sampled_distance_only, exact_distance_normal, halfplane_channels
from warpSPHBoundaries.validation.error_colors import error_style, error_colorbar, BAND_LEGEND
from warpSPHBoundaries.validation.report_view import accessible_report
from warpSPHBoundaries.validation.robustness import save_measurements
from warpSPHBoundaries.validation.sdf_stress import CASES, sampled_rep_model, _axes


def evaluate(rep, case, points, target, kernel):
    result = sampled_distance_only(rep, points, kernel)
    d, n = exact_distance_normal(case, points)
    exact_plane = halfplane_channels(d, n, kernel, np.longdouble)
    return dict(result, exact_distance=d, exact_normal=n, exact_plane=exact_plane,
                numerical_error=result['values']-result['model_reference'],
                sampling_error=result['model_reference']-exact_plane,
                geometry_error=exact_plane-target, total_error=result['values']-target)


def plot_fields(path, case, grid, target, result, channels):
    plt = _pyplot()
    fig, axes = plt.subplots(3, 6, figsize=(18, 8.5), constrained_layout=True)
    for col, channel in enumerate(channels):
        ref = target[:, channel].reshape(len(grid), len(grid))
        value = result['values'][:, channel].reshape(ref.shape)
        limit = max(float(np.max(np.abs(ref))), 1e-12)
        for row, values in enumerate((ref, value, np.abs(value-ref))):
            ax = axes[row, col]
            ax.set_facecolor('#e9e9e9')
            if row < 2:
                im = ax.pcolormesh(grid, grid, np.ma.masked_invalid(values), shading='nearest',
                                   cmap='RdBu_r', vmin=-limit, vmax=limit, rasterized=True)
            else:
                im = ax.pcolormesh(grid, grid, np.ma.masked_invalid(values), shading='nearest', rasterized=True, **error_style())
            _axes(ax, case, grid)
            if row == 0:
                ax.set_title(LABELS[channel])
            if col == 0:
                ax.set_ylabel(('Original geometry', 'Distance-only half-plane\n(no curvature / fallback)', 'Total absolute error')[row]+'\n$y/h$')
            else:
                ax.tick_params(labelleft=False)
            if row == 2:
                ax.set_xlabel('$x/h$')
                error_colorbar(fig, im, ax=ax, shrink=.65, pad=.01)
            else:
                ax.tick_params(labelbottom=False)
                if row == 1:
                    fig.colorbar(im, ax=axes[:2, col], shrink=.65, pad=.01, extend='both')
    fig.suptitle(f'{case.name}: DISTANCE-ONLY SEMI-ANALYTIC REFERENCE — local half-plane, no curvature recovery\n'
                 'Mass uses distance only; oriented moments also use the sampled unit normal. Grey: undefined normal / outside grid.\n'
                 'Value colours use reference scale; error colours retain the actual discrepancy.\n'+BAND_LEGEND, fontsize=10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_cut_budget(path, case, cut, target, result, reference_estimate):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 6, figsize=(18, 8.5), constrained_layout=True)
    for channel, ax in enumerate(axes.flat):
        for key, label, color in (
            ('total_error', 'total discrepancy', '#222222'),
            ('geometry_error', 'half-plane geometry bias', '#b35806'),
            ('sampling_error', 'SDF distance/normal reconstruction', '#009e73'),
            ('numerical_error', 'evaluation precision difference', '#2166ac')):
            ax.semilogy(cut, np.maximum(np.abs(result[key][:, channel]), 1e-18), color=color, label=label)
        ax.semilogy(cut, np.maximum(reference_estimate[:, channel], 1e-18), ':', color='#777777', label='target refinement estimate')
        ax.axhline(np.finfo(np.float32).eps, ls='--', color='#c5410d', lw=.7)
        ax.axhline(np.finfo(np.float64).eps, ls=':', color='#112b52', lw=.7)
        ax.set(title=LABELS[channel], xlabel=f'{case.cut_axis} cut / h')
        ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=8)
    fig.suptitle(f'{case.name}: distance-only signed error budget (magnitudes shown)\n'
                 'Shared twelve-channel order, absolute support units. Contributions add before taking magnitudes; missing exact normals leave gaps.\n'
                 'Float64 versus extended-precision differences and reference refinements are empirical, not certified bounds.', fontsize=11)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_mass(path, case, grid, target, result):
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for i, (ax, values) in enumerate(zip(axes, (target[:, 0], result['values'][:, 0], np.abs(result['total_error'][:, 0])))):
        if i < 2:
            im = ax.pcolormesh(grid, grid, values.reshape(len(grid), len(grid)), shading='nearest', cmap='viridis', vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, shrink=.7)
        else:
            im = ax.pcolormesh(grid, grid, values.reshape(len(grid), len(grid)), shading='nearest', **error_style())
            error_colorbar(fig, im, ax=ax, shrink=.7, detailed=True)
        _axes(ax, case, grid)
        ax.set(xlabel='$x/h$', ylabel='$y/h$', title=('Original geometry $m_{00}$', 'Distance-only half-plane $m_{00}$', 'Total absolute mass error')[i])
    fig.suptitle(f'{case.name}: distance-only SDF reference — no curvature, no guard, no fallback\n'
                 'Continuous mass model; geometric bias remains. '+BAND_LEGEND.split('\n')[0], fontsize=10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run(baseline, guard, output, grid_size=129):
    baseline, guard, output = Path(baseline), Path(guard), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((baseline/'summary.json').read_text())
    cases = {c.name: c for c in polygon_cases()}
    records, counts = [], []
    for row in source['records']:
        with np.load(baseline/row['kernel']/row['case']/'data.npz') as a:
            points, target, uncertainty = a['points'], a['target'], a['reference_estimate']
        case = cases[row['case']]
        rep = make_sdf(case, points, 1/64)
        result = evaluate(rep, case, points, target, row['kernel'])
        forced = sampled_rep_model(rep, points, row['kernel'])
        methods = [Measurement('SDF-distance-only', result['values'], result['model_reference'],
                               note='All twelve closed-form half-plane moments. No curvature or selection. Mass finite with undefined directions shown separately; float64/extended precision comparison is empirical.'),
                   Measurement('SDF-forced', forced['values'], supported=SUPPORTED,
                               note='Diagnostic curvature series, guard bypassed; nine channels.')]
        with np.load(guard/row['kernel']/row['case']/'data.npz') as a:
            np.testing.assert_array_equal(a['points'], points)
            np.testing.assert_array_equal(a['target'], target)
            methods.append(Measurement('SDF-cell-guarded', a['SDF-cell-guarded'], supported=SUPPORTED,
                                       note='Preserved executed CPU guard checkpoint; all targeted queries used original-polygon fallback.'))
        folder = output/row['kernel']/row['case']
        folder.mkdir(parents=True, exist_ok=True)
        summaries = save_measurements(folder, target, uncertainty, methods, points=points, vertices=case.vertices,
            **{k: v for k, v in result.items() if k not in ('values', 'model_reference')},
            **{'SDF-distance-only_sampling_error': result['sampling_error'],
               'SDF-distance-only_geometry_error': result['geometry_error']})
        records.append(dict(row, methods=summaries))
        counts.append(dict(case=row['case'], kernel=row['kernel'], normal_undefined=int((~result['normal_defined']).sum()),
                           mass_nonfinite=int((~np.isfinite(result['values'][:, 0])).sum()),
                           mass_max_absolute_error=float(np.max(np.abs(result['total_error'][:, 0])))))
        print(row['kernel'], row['case'], flush=True)
    grid = np.linspace(-1.2, 1.2, grid_size)
    x, y = np.meshgrid(grid, grid)
    points = np.column_stack((x.ravel(), y.ravel()))
    cut = np.linspace(-1.2, 1.2, 401)
    stress_rows = []
    stress_cache = output.parent/'boundary-atlas'/'sdf-stress'
    cached_summary = json.loads((stress_cache/'summary.json').read_text()) if (stress_cache/'summary.json').exists() else {}
    for case in CASES:
        print('stress', case.name, flush=True)
        folder = output/'stress'/case.name
        folder.mkdir(parents=True, exist_ok=True)
        rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy())), (-2.5, -2.5), (2.5, 2.5), 1/64)
        cached = stress_cache/case.name/'data.npz'
        previous = next((c for c in cached_summary.get('cases', []) if c['name'] == case.name), {})
        matching_geometry = previous.get('geometry') == json.loads(json.dumps(asdict(case)))
        if cached.exists() and matching_geometry and cached_summary.get('kernel') == 'w4':
            with np.load(cached) as a:
                target = a['target'] if np.array_equal(a['points'], points) else case.reference(points, 'w4')
        else:
            target = case.reference(points, 'w4')
        result = evaluate(rep, case, points, target, 'w4')
        cut_points = case.cut_points(cut)
        cut_target = case.reference(cut_points, 'w4')
        cut_result = evaluate(rep, case, cut_points, cut_target, 'w4')
        uncertainty = (np.abs(cut_target-case.reference(cut_points, 'w4', 48)) if case.kind == 'disks'
                       else np.full_like(cut_target, np.nan))
        np.savez_compressed(folder/'data.npz', grid=grid, points=points, target=target,
                            cut=cut, cut_target=cut_target, cut_reference_estimate=uncertainty,
                            **result, **{'cut_'+k: v for k, v in cut_result.items()})
        plot_mass(folder/'mass.png', case, grid, target, result)
        plot_fields(folder/'moments.png', case, grid, target, result, range(6))
        plot_fields(folder/'gradient-moments.png', case, grid, target, result, range(6, 12))
        plot_cut_budget(folder/'error-budget.png', case, cut, cut_target, cut_result, uncertainty)
        stress_rows.append(dict(name=case.name, mass_max_error=float(np.max(np.abs(result['total_error'][:, 0]))),
                                mass_range=[float(result['values'][:, 0].min()), float(result['values'][:, 0].max())],
                                undefined_sampled_normals=int((~result['normal_defined']).sum())))
    summary = dict(source, records=records, specialized=[], representation=[], checkpoint='curvature-free half-plane SDF reference')
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (output/'checkpoint.json').write_text(json.dumps(dict(method='SDF-distance-only', counts=counts, stress=stress_rows), indent=2)+'\n')
    (output/'coverage.json').write_text(json.dumps(dict(explicit_limits=[
        '21 preserved polygon fixtures, four kernels, all twelve half-plane moments where the sampled normal is defined.',
        'Standalone reference sampler computes values and first differences only, without production curvature recovery or selection.',
        'Four Wendland C4 spatial stress cases: corner, tight disk, thin strip and narrow gap; fixed twelve-channel order.',
        'Mass depends only on distance; oriented moments with unresolved normals are NaN, explicitly retained and counted.',
        'Model/geometry, sampled-distance/normal reconstruction, and finite-precision evaluation contributions are separate.',
        'No quadrature in the half-plane implementation; extended-precision comparisons are empirical, not certified bounds.',
        'Original polygon targets retain high-precision reference estimates; disk targets use radial/polar integration and cut refinement estimates.',
        'Production dispatch and the existing guarded checkpoint remain unchanged. Reference CPU timings are not throughput claims.']), indent=2)+'\n')
    checks = baseline/'independent-oracle-checks.json'
    if checks.exists():
        (output/checks.name).write_bytes(checks.read_bytes())
    path = generate(output, 300)
    title, rest = path.read_text().split('\n', 1)
    intro = '''## Distance-only semi-analytic reference

`SDF-distance-only` is a standalone half-plane model. Mass is F0(d/h);
vector/tensor moments additionally use the sampled unit normal. The evaluator
computes first differences only: no Laplacian, curvature recovery, curvature
series, curvature validity selection, or fallback. All twelve moments use
the existing polygon closed forms on an equivalent half-plane, rotated into
the normal frame. This reference study uses h=1 and CPU float64 evaluation;
production dispatch is unchanged. It is a reference model, not a throughput
implementation or a claim to reproduce every detail of the cited paper.

The mass has no curvature pole and remains continuous wherever the sampled
distance is continuous. It can still be geometrically wrong: at a right-angle
vertex the planar limit gives mass 1/2 instead of the actual 1/4. Thin solids,
gaps and multiple sheets produce different nonlocal-geometry biases. At a
medial axis the mass is defined but a unique normal may not be: direction-dependent
channels are grey where the sampled gradient is unresolved. A reconstructed
bisector at a ridge is a chosen normal, not a physically unique closest face.
Finite or unavailable oriented moments therefore have a different interpretation
from a curvature-series blow-up. Exact geometric normal ties are separately
masked in the decomposition; this does not discard a valid scalar mass.

The preserved polygon sweep covers 21 geometries and four kernels. Comparison
methods are the forced curvature series and the executed CPU guard checkpoint.
All 4,480 polygon query/kernel records in that guard checkpoint used fallback;
its accuracy is not evidence for the local model. A nonfinite oriented result
of the distance-only model at an undefined normal is labelled in checkpoint.json;
it must not be interpreted as mass divergence. No accuracy success is inferred
by hiding these missing directions.

Large signal-relative errors on tiny triangles mean geometry/model loss:
the actual mass scales with triangle area while the half-plane predicts about
half a kernel. This is distinct from an unbounded curvature-series prediction;
read the absolute and signal-relative plots together.

For original geometry G and the two half-plane descriptions, the signed split is:

```
numerical = float64 plane(sampled d,n) - extended-precision plane(sampled d,n)
SDF reconstruction = extended-precision plane(sampled d,n) - plane(exact d,n)
half-plane geometry = plane(exact d,n) - I(G)
total = numerical + SDF reconstruction + half-plane geometry
```

These contributions add pointwise where the normal references are defined;
their separately maximized magnitudes need not add. Mass never needs a normal.
The normal sampling contribution includes reconstruction and normalization.
The exact-geometry jets use analytic disk distances or segment projections
evaluated in extended precision, not derivatives of the sampled grid.
No quadrature is used for the half-plane profiles. Float64/extended-precision
differences are empirical evaluation checks, not certified error bounds.
Disk-union targets use independent radial/polar integrals with a 48/96-node
refinement estimate on cuts; polygon targets use extended/high precision
closed forms. Shared formulas and finite reference precision limit few-eps claims.

'''
    for case in CASES:
        intro += f"### {case.name}: distance-only reference\n\n{case.assumption}.\n\n"
        for file, label in [('mass.png', 'Mass and total error'), ('moments.png', 'Scalar moments'),
                            ('gradient-moments.png', 'Gradient moments'), ('error-budget.png', 'Separate error contributions')]:
            intro += f'![{case.name}: {label}](stress/{case.name}/{file})\n\n'
    path.write_text(title+'\n\n'+intro+rest)
    accessible_report(path)
    print('Wrote', path, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default=evidence_path('triangle-conditioning'))
    p.add_argument('--guarded', default=evidence_path('sdf-guard-cpu'))
    p.add_argument('--output', default=evidence_path('sdf-distance-only'))
    p.add_argument('--grid', type=int, default=129)
    a = p.parse_args()
    run(a.baseline, a.guarded, a.output, a.grid)
