"""Executed all-moment curvature corrections against independent curved domains."""
import argparse
from dataclasses import asdict
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from sdf_accuracy import references, signal_scales, KERNELS, DISTANCES, RESOLUTIONS
from warpSPHBoundaries.scene.scene import SdfRep, _sdfCurvature, tier3Table
from warpSPHBoundaries.validation.adversarial import scene_channels
from warpSPHBoundaries.validation.atlas import CHANNELS, LABELS, _pyplot
from warpSPHBoundaries.validation.curvature import model_reference, exact_variation, RippleWall
from warpSPHBoundaries.validation.error_colors import EPS32, error_style, error_colorbar
from warpSPHBoundaries.validation.paths import evidence_path
from warpSPHBoundaries.validation.report_view import accessible_report
from warpSPHBoundaries.validation.sdf_accuracy import accepted_channel_gate, smooth_cases, area_reference

VERSION = 'shape-order2-warp-v3'


def verify_backend(baseline, output, device):
    """Re-execute every cached query without recomputing unchanged references."""
    if baseline.resolve() == output.resolve():
        raise ValueError('Preserve the pre-fusion evidence')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(baseline/'references', output/'references', dirs_exist_ok=True)
    maximum, changed, queries = 0., 0, 0
    for case in smooth_cases():
        for resolution in RESOLUTIONS:
            first_path = baseline/'data'/KERNELS[0]/f'{case.name}-h{resolution}.npz'
            with np.load(first_path) as a:
                points = a['points']
            rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy())),
                                     points.min(0)-1.35, points.max(0)+1.35, 1/resolution).to(device)
            pos = torch.tensor(points, dtype=torch.float64, device=device)
            h = torch.ones(len(points), dtype=torch.float64, device=device)
            for kernel in KERNELS:
                relative = Path('data')/kernel/f'{case.name}-h{resolution}.npz'
                with np.load(baseline/relative) as source:
                    a = dict(source)
                if str(a['version']) not in ('shape-order2-v2', VERSION):
                    raise ValueError('Backend verification requires the shared order-two checkpoint')
                _, low = _sdfCurvature(rep, pos, h, kernel, device)
                accepted = ~low.cpu().numpy()
                actual = np.full_like(a['values'], np.nan)
                if accepted.any():
                    actual[accepted] = scene_channels(rep, points[accepted], kernel, device, second=True)
                common = accepted & a['accepted']
                if common.any():
                    difference = np.abs(actual[common]-a['values'][common])
                    maximum = max(maximum, float(difference.max()))
                    np.testing.assert_allclose(actual[common], a['values'][common], atol=3e-13, rtol=3e-12)
                changed += int(np.count_nonzero(accepted != a['accepted']))
                queries += len(points)
                a.update(version=VERSION, device=device, accepted=accepted, values=actual,
                         numerical=actual.astype(np.longdouble)-a['same_model'],
                         total=actual.astype(np.longdouble)-a['target'])
                np.testing.assert_allclose((a['numerical']+a['reconstruction']+a['model'])[accepted],
                                           a['total'][accepted], atol=1e-17, rtol=1e-12)
                path = output/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **a)
            print(f'verified {case.name} h/{resolution}', flush=True)
    result = dict(device=device, baseline=str(baseline), queries=queries,
                  maximum_output_change=maximum, selection_changes=changed)
    (output/'backend-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


def measure(output, device, reference_cache):
    output.mkdir(parents=True, exist_ok=True)
    for case in smooth_cases():
        refs = {kernel: references(case, kernel, output, reference_cache) for kernel in KERNELS}
        points = case.points(DISTANCES)
        _, normal, _, curvature, _ = case.frame
        normals = np.broadcast_to(normal@case.matrix.T, (len(points), 2)).copy()
        exact_beta = exact_variation(case, len(points))
        exact_models = {kernel: model_reference(DISTANCES, normals, np.full(len(points), curvature), exact_beta, kernel)
                        for kernel in KERNELS}
        for resolution in RESOLUTIONS:
            paths = [output/'data'/kernel/f'{case.name}-h{resolution}.npz' for kernel in KERNELS]
            if all(p.exists() for p in paths):
                compatible = True
                for path in paths:
                    with np.load(path) as a:
                        assert str(a['device']) == device
                        compatible &= str(a['version']) == VERSION
                if compatible:
                    print(f'cached {case.name} h/{resolution}', flush=True)
                    continue
            rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy())),
                                     points.min(0)-1.35, points.max(0)+1.35, 1/resolution).to(device)
            pos = torch.tensor(points, device=device)
            h = torch.ones(len(points), dtype=torch.float64, device=device)
            d, n, k, norm = rep.signed(pos)
            beta, covered = rep.curvatureVariation(pos)
            cpu = lambda x: x.detach().cpu().numpy()
            for kernel, path in zip(KERNELS, paths):
                ref = refs[kernel]
                table = tier3Table(kernel, device)
                raw = cpu(table.curvature_moments(d, n, k, beta))
                constant = cpu(table.curvature_moments(d, n, k, torch.zeros_like(beta)))
                _, low = _sdfCurvature(rep, pos, h, kernel, device)
                accepted = ~cpu(low)
                actual = np.full_like(raw, np.nan)
                if accepted.any():
                    actual[accepted] = scene_channels(rep, points[accepted], kernel, device, second=True)
                    np.testing.assert_allclose(actual[accepted], raw[accepted], atol=2e-14, rtol=2e-13)
                # The historical mixed model is a comparator, not a production
                # result: reconstruct its former mass/gradient and planar m1/g1.
                old_mass, old_derivative = table.lam(d, k)
                _, _, first, nn, tt = table.planar_moments(d)
                tensor = tt[:, None, None]*torch.eye(2, dtype=torch.float64, device=device)+(nn-tt)[:, None, None]*n[:, :, None]*n[:, None, :]
                legacy = np.full_like(raw, np.nan)
                legacy[:, 0], legacy[:, 1:3] = cpu(old_mass), cpu(-first[:, None]*n)
                legacy[:, 6:8], legacy[:, 8:] = cpu(old_derivative[:, None]*n), cpu(tensor.flatten(1))
                same = model_reference(cpu(d), cpu(n), cpu(k), cpu(beta), kernel, 128)
                coarse = model_reference(cpu(d), cpu(n), cpu(k), cpu(beta), kernel, 64)
                exact = exact_models[kernel]
                numerical = raw.astype(np.longdouble)-same
                reconstruction = same-exact
                model = exact-ref['target']
                total = actual.astype(np.longdouble)-ref['target']
                np.testing.assert_allclose((numerical+reconstruction+model)[accepted], total[accepted], atol=1e-17, rtol=1e-12)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, version=VERSION, device=device, geometry=json.dumps(asdict(case), sort_keys=True),
                    points=points, distances=DISTANCES, channels=CHANNELS, accepted=accepted,
                    values=actual, raw=raw, constant=constant, legacy=legacy, target=ref['target'],
                    sampled_distance=cpu(d), sampled_normal=cpu(n), sampled_curvature=cpu(k),
                    sampled_variation=cpu(beta), exact_variation=exact_beta, variation_covered=cpu(covered),
                    same_model=same, exact_model=exact, numerical=numerical, reconstruction=reconstruction,
                    model=model, total=total, model_reference_refinement=np.abs(same-coarse),
                    reference_uncertainty=ref['reference_uncertainty'], scale=signal_scales(ref['target']))
            print(f'measured {case.name} h/{resolution}: {accepted.sum()}/{len(points)} accepted', flush=True)


def ripple_stress(output, device):
    """New expansion counterexample: bounded curvature, large variation."""
    path = output/'rapid-ripple.npz'
    if path.exists():
        with np.load(path) as a:
            if str(a['version']) == VERSION and str(a['device']) == device:
                return
    from warpSPHBoundaries.scene.sdf_guard import cell_support_limits, square_minima, support_validity
    wall = RippleWall()
    distances = np.linspace(-.6, .6, 13)
    points = wall.points(distances)
    rep = SdfRep.fromFunction(lambda p: torch.tensor(wall.distance(p.numpy())),
                             points.min(0)-1.25, points.max(0)+1.25, 1/128).to(device)
    pos = torch.tensor(points, device=device)
    h = torch.ones(len(points), dtype=torch.float64, device=device)
    d, n, k, _ = rep.signed(pos)
    beta, _ = rep.curvatureVariation(pos)
    table = tier3Table('w4', device)
    cpu = lambda x: x.detach().cpu().numpy()
    forced = cpu(table.curvature_moments(d, n, k, beta))
    _, low = _sdfCurvature(rep, pos, h, 'w4', device)
    old = square_minima(cell_support_limits(rep._node_fields(), rep.spacing, .5, .1, .25))
    old_accepted = cpu(support_validity(old, rep.origin, rep.spacing, pos, h))
    old_mass, old_derivative = table.lam(d, k)
    _, _, first, nn, tt = table.planar_moments(d)
    cov = tt[:, None, None]*torch.eye(2, dtype=torch.float64, device=device)+(nn-tt)[:, None, None]*n[:, :, None]*n[:, None, :]
    legacy = np.full_like(forced, np.nan)
    legacy[:, 0], legacy[:, 1:3] = cpu(old_mass), cpu(-first[:, None]*n)
    legacy[:, 6:8], legacy[:, 8:] = cpu(old_derivative[:, None]*n), cpu(cov.flatten(1))
    target, uncertainty = [], []
    for point in points:
        fine, e = area_reference(wall, point, 'w4', 96, 2e-12)
        coarse, ec = area_reference(wall, point, 'w4', 48, 1e-11)
        target.append(fine)
        uncertainty.append(np.abs(fine-coarse)+e+ec)
    assert old_accepted.any() and low.all()
    np.savez_compressed(path, version=VERSION, device=device, distances=distances, points=points,
        target=np.asarray(target), reference_uncertainty=np.asarray(uncertainty), forced=forced,
        legacy=legacy, accepted=~cpu(low), curvature_only_accepted=old_accepted,
        sampled_variation=cpu(beta), sampled_curvature=cpu(k), channels=CHANNELS)


def report(output, comparison=None):
    cases = smooth_cases()
    arrays, records = {}, []
    parity = []
    for case in cases:
        for kernel in KERNELS:
            for resolution in RESOLUTIONS:
                relative = Path('data')/kernel/f'{case.name}-h{resolution}.npz'
                with np.load(output/relative) as a:
                    a = dict(a)
                assert str(a['version']) in ('shape-order2-v2', VERSION)
                arrays[case.name, kernel, resolution] = a
                mask = a['accepted']
                gate = accepted_channel_gate(a['total'], a['reference_uncertainty'], a['scale'], mask, EPS32)
                row = dict(case=case.name, kernel=kernel, resolution=resolution, accepted=int(mask.sum()),
                    rejected=int((~mask).sum()), channel_pass=gate.tolist(), all_twelve_pass=bool(mask.any() and gate.all()),
                    max_absolute=float(np.nanmax(np.abs(a['total'][mask]))) if mask.any() else None,
                    max_relative=float(np.nanmax(np.abs(a['total'][mask])/a['scale'])) if mask.any() else None)
                records.append(row)
                if comparison:
                    with np.load(comparison/relative) as b:
                        for key in ('points', 'target', 'accepted'):
                            np.testing.assert_array_equal(a[key], b[key])
                        difference = np.abs(a['values'][mask]-b['values'][mask])
                        np.testing.assert_allclose(a['values'][mask], b['values'][mask], atol=3e-12, rtol=3e-12)
                        parity.append(float(difference.max()) if difference.size else 0.)
    plt = _pyplot()
    titles = dict(total='Executed Scene: total error', model='Exact jets: order-two geometry/model error',
                  reconstruction='Sampled d/n/curvature/variation: reconstruction error',
                  numerical='Spline/table evaluation discrepancy',
                  reference_uncertainty='Original-domain reference quadrature uncertainty')
    for component, title in titles.items():
        fig, axes = plt.subplots(2, 6, figsize=(20, 10), constrained_layout=True)
        for j, ax in enumerate(axes.flat):
            data = np.array([np.abs(arrays[c.name, 'w4', 64][component][:, j])/arrays[c.name, 'w4', 64]['scale'][j] for c in cases], float)
            if component != 'reference_uncertainty':
                mask = np.array([arrays[c.name, 'w4', 64]['accepted'] for c in cases])
                data = np.where(mask, data, np.nan)
            im = ax.pcolormesh(DISTANCES, range(len(cases)), np.ma.masked_invalid(data), shading='nearest', **error_style())
            ax.set(title=LABELS[j], ylim=(len(cases)-.5, -.5), xlabel='$d/h$')
            ax.set_yticks(range(len(cases)), [c.name for c in cases] if j % 6 == 0 else [])
        error_colorbar(fig, im, ax=list(axes.flat), detailed=True, shrink=.7)
        fig.suptitle(title+'\nShared geometric order two, all twelve moments. w4, h/64; block-signal-relative; empirical precision guides.')
        fig.savefig(output/f'{component}-w4.png', dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(19, 7), constrained_layout=True)
    from matplotlib.colors import ListedColormap
    for ax, resolution in zip(axes, RESOLUTIONS):
        gate = np.full((len(cases), 12), np.nan)
        for i, case in enumerate(cases):
            rows = [r for r in records if r['case'] == case.name and r['resolution'] == resolution and r['accepted']]
            if rows:
                gate[i] = np.all([r['channel_pass'] for r in rows], 0)
        ax.imshow(np.ma.masked_invalid(gate), aspect='auto', vmin=0, vmax=1,
                  cmap=ListedColormap(['#c5410d', '#112b52']).with_extremes(bad='#d9dde3'))
        ax.set_yticks(range(len(cases)), [c.name for c in cases])
        ax.set_xticks(range(12), CHANNELS, rotation=70)
        ax.set_title(f'h/{resolution}: all four kernels')
    fig.suptitle('All-twelve-moment ε32 matrix: navy = measured pass; orange = fail; grey = no accepted samples\nReference estimates included. No continuous-domain or few-ε64 certificate.')
    fig.savefig(output/'acceptance-matrix.png', dpi=300)
    plt.close(fig)

    groups = [(0,), (1, 2), (6, 7), (8, 9, 10, 11)]
    maxima = {component: [max(float(np.max(np.abs(arrays[c.name, kernel, 64][component][:, group])))
                             for c in cases for kernel in KERNELS) for group in groups]
              for component in ('model', 'reconstruction', 'numerical', 'total')}
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), constrained_layout=True)
    comparison_rows = []
    for ax, group, name in zip(axes, groups, ('m00', 'm1', 'G', 'g1')):
        for scheme, color in zip(('legacy', 'constant', 'values'), ('#b55b08', '#8d4aa8', '#1764ab')):
            values = [max(float(np.max(np.abs(arrays[c.name, kernel, 64][scheme][:, group]-arrays[c.name, kernel, 64]['target'][:, group])))
                          for kernel in KERNELS) for c in cases]
            label = {'legacy': 'legacy mixed model', 'constant': 'curvature only', 'values': 'corrected Scene'}[scheme]
            ax.semilogy(range(len(cases)), np.maximum(values, 1e-18), '-o', label=label, color=color)
            for case, value in zip(cases, values):
                comparison_rows.append(dict(case=case.name, group=name, scheme=scheme, max_absolute=value))
        ax.set_xticks(range(len(cases)), [c.name for c in cases], rotation=70, ha='right', fontsize=7)
        ax.set(title=name, ylabel='Max absolute error, four kernels', ylim=(1e-9, 1))
        ax.axhline(EPS32, ls=':', color='black', label='ε32 absolute guide')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('Same sampled jets, h/64: reconstructed legacy mixed model → curvature moments → curvature + variation\nLegacy is a mathematical comparator; corrected Scene is the actual executed output.')
    fig.savefig(output/'before-after.png', dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for name, color in zip(('inflection-b0.03', 'inflection-b0.12', 'inflection-b0.3'), ('#1764ab', '#228557', '#8d4aa8')):
        a = arrays[name, 'w4', 64]
        axes[0].plot(DISTANCES, a['target'][:, 6], color=color, label=name+' true')
        axes[0].plot(DISTANCES, a['values'][:, 6], '--', color=color, label=name+' Scene')
        for scheme, style in (('legacy', '--'), ('values', '-')):
            label = 'legacy' if scheme == 'legacy' else 'corrected Scene'
            axes[1].semilogy(DISTANCES, np.maximum(abs(a[scheme][:, 6]-a['target'][:, 6]), 1e-18), style, color=color, label=name+' '+label)
        axes[2].plot(DISTANCES, a['sampled_variation']-a['exact_variation'], color=color, label=name)
    for ax, title in zip(axes, ('Inflection tangential gradient restored', 'Absolute gradient error: legacy vs corrected', 'Curvature-variation reconstruction error')):
        ax.set(title=title, xlabel='$d/h$')
        ax.grid(alpha=.2)
        ax.legend(fontsize=6)
    axes[1].axhline(EPS32, ls=':', color='black')
    fig.savefig(output/'inflection-gradient.png', dpi=300)
    plt.close(fig)

    with np.load(output/'rapid-ripple.npz') as a:
        ripple = dict(a)
    fig, axes = plt.subplots(2, 6, figsize=(20, 7), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        target = ripple['target'][:, j]
        data = np.array([np.abs(ripple[key][:, j]-target) for key in ('legacy', 'forced')]+[
            np.where(ripple['accepted'], np.abs(ripple['forced'][:, j]-target), np.nan)], float)
        im = ax.pcolormesh(ripple['distances'], range(3), np.ma.masked_invalid(data), shading='nearest', **error_style())
        ax.set(title=LABELS[j], xlabel='$d/h$', ylim=(2.5, -.5))
        ax.set_yticks(range(3), ['legacy diagnostic', 'new forced diagnostic', 'new policy accepts'] if j % 6 == 0 else [])
    error_colorbar(fig, im, ax=list(axes.flat), detailed=True, shrink=.7)
    fig.suptitle('Model counterexample: z=0.001 sin(20t), |κ|≤0.4 but |κ_s|≈8\nAbsolute moment errors. All new-model rows are rejected; forced errors are diagnostics, never substituted into output.')
    fig.savefig(output/'rapid-ripple.png', dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].plot(ripple['distances'], ripple['sampled_curvature'], label='sampled κh')
    axes[0].plot(ripple['distances'], ripple['sampled_variation'], label='sampled κ_s h²')
    axes[0].axhline(.5, color='black', ls=':')
    axes[0].axhline(-.5, color='black', ls=':')
    axes[0].set(title='Small curvature can coexist with rapid variation', xlabel='$d/h$')
    for channel in (0, 1, 2, 6, 7, 8, 9, 11):
        axes[1].semilogy(ripple['distances'], np.maximum(ripple['reference_uncertainty'][:, channel], 1e-18), label=CHANNELS[channel])
    axes[1].set(title='Actual-domain quadrature uncertainty, 48/96', xlabel='$d/h$', ylabel='Empirical absolute estimate')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(output/'rapid-ripple-estimates.png', dpi=300)
    plt.close(fig)

    checks = []
    for case in cases:
        for kernel in KERNELS:
            with np.load(output/'references'/kernel/f'{case.name}.npz') as a:
                checks.append(float(a['independent_difference'].max()))
    summary = dict(version=str(next(iter(arrays.values()))['version']), device=str(next(iter(arrays.values()))['device']), channels=CHANNELS,
                   accepted=sum(r['accepted'] for r in records), rejected=sum(r['rejected'] for r in records),
                   passed_sweeps=sum(r['all_twelve_pass'] for r in records), sweeps=len(records), records=records,
                   max_independent_discrepancy=max(checks), component_maxima_h64=maxima,
                   comparisons=comparison_rows, max_device_difference=max(parity) if parity else None)
    summary['rapid_ripple'] = dict(curvature_only_accepted=int(ripple['curvature_only_accepted'].sum()),
        new_accepted=int(ripple['accepted'].sum()), queries=len(ripple['distances']),
        max_abs_forced_error=float(np.nanmax(np.abs(ripple['forced']-ripple['target']))),
        max_reference_estimate=float(ripple['reference_uncertainty'].max()))
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# Curvature corrections: shared twelve-moment family', '',
        f"Main smooth sweep: **{summary['accepted']} accepted records**, {summary['rejected']} rejected, {summary['passed_sweeps']}/{summary['sweeps']} full-twelve-moment ε32 sweeps passed. "
        'Corrections improve the model; acceptance is still a differential-field domain check, not an integral-tolerance certificate.', '',
        '13 geometries, four kernels, three grids, 18 signed offsets including the wall. Actual Scene outputs are compared with true-domain Cartesian quadrature. '
        'Independent 45/60-digit boundary checks retained from the validated references cover all twelve moments at 52 probes. '
        f"Maximum independent discrepancy: {max(checks):.3g} absolute.", '',
        '## What changed', '',
        'All twelve moments use the same boundary graph expansion through geometric order two: κh counts as first order, and κ_s h² as second order. '
        'Curvature terms now correct m1, g1 and scalar second moments. The curvature-variation term restores the tangential gradient at inflections. '
        'Maple verifies the formulas and ∂m1=g1−λI through this order. Derivative consistency is asymptotic; it is not an exact identity for arbitrary truncated geometry.', '',
        'Production reuses the shipped F0/F1 coefficient splines and integrates their cubic segments analytically. No new kernel table or fixed boundary quadrature is introduced. '
        'The complete precision-refinable closed-form model is available as `edge.curvature.moments`; independent direct shape-Taylor quadrature validates the coefficient reduction. '
        'Existing F0 derivative interpolation remains a separate numerical accuracy limit.', '',
        '![Before/after moment errors](before-after.png)', '',
        'The legacy curve reconstructs the former curvature-mass/normal-gradient plus planar-m1/g1 formulation using the same sampled jets. '
        'The curvature-only curve disables curvature variation while retaining all curvature moment terms. The corrected Scene curve is executed Scene with the full correction.', '',
        '## Per-moment accuracy', '', '![Twelve-moment matrix](acceptance-matrix.png)', '',
        '![Executed total errors](total-w4.png)', '',
        '## Error decomposition', '',
        'Signed table/evaluation + sampled-jet reconstruction + exact-jet model error = total. '
        'The same-model reference integrates every shape-Taylor integrand directly, without production spline coefficients or reduced moment identities. '
        'All error matrices use the same twelve-channel order and the original scalar/vector/tensor signal scales. Grey means uncomputed/rejected.', '',
        '| Maximum absolute h/64 contribution | m00 | m1 components | G components | g1 components |', '|---|---|---|---|---|',
        *['| '+key+' | '+' | '.join(f'{x:.3g}' for x in values)+' |' for key, values in maxima.items()], '',
        '![Exact-jet model errors](model-w4.png)', '', '![SDF reconstruction errors](reconstruction-w4.png)', '',
        '![Table numerical discrepancies](numerical-w4.png)', '',
        '![Original-domain quadrature uncertainty](reference_uncertainty-w4.png)', '',
        'Reference refinement/adaptive estimates and model-reference 64/128-node differences are empirical, not certified bounds. '
        'Model-reference refinement arrays and all signed components are retained in NPZ.', '',
        '## Inflection and remaining limitations', '', '![Corrected inflection gradients](inflection-gradient.png)', '',
        'Higher geometric orders, curvature second derivatives, sampled differential noise and omitted subgrid geometry remain. '
        'The additional curvature-variation reconstruction needs a two-node distance stencil; incomplete coverage or nonfinite variation requests fallback before evaluating the model. '
        'Mass bounds and all-channel finiteness are checked after evaluation; failed rows are not clamped or replaced by a reference.', '',
        'The domain guard also bounds |κ_s|h² over every cell of the support square, using '
        '||grad Delta d||/(1-d Delta d)³ and a default dimensionless limit of 0.5. This is an asymptotic-domain restriction, not an epsilon32 guarantee.', '',
        '## Rapid-variation counterexample', '',
        f"The wall z=0.001 sin(20t) keeps |κ|≤0.4 but has |κ_s| near 8 at its inflection. "
        f"The curvature-only cell policy accepts {summary['rapid_ripple']['curvature_only_accepted']}/13 cut points. "
        f"The new variation guard accepts {summary['rapid_ripple']['new_accepted']}/13. "
        'This catches a failure introduced by extending the local model to curvature variation. Higher derivatives can still invalidate milder cases; the guard is not a truncation-error estimator.', '',
        '![Rapid-variation moment errors and rejection](rapid-ripple.png)', '',
        '![Rapid-variation jets and quadrature estimates](rapid-ripple-estimates.png)', '',
        'This report measures the Scene SDF branch. FusedWall still lowers an SDF to its explicit polygon geometry; it does not run this curved local model. '
        'Other representations retain their separately documented moment coverage.', '',
        '## Reproduce', '',
        'Run `python scripts/studies/curvature_corrections.py --device cpu`. Current curved-domain references are generated if no compatible cache exists. '
        'Use `--device cuda:0 --output "$EVIDENCE/curvature-corrections-cuda" --reference-cache "$EVIDENCE/sdf-accuracy-cpu" '
        '--compare-to "$EVIDENCE/curvature-corrections-cpu"` for GPU verification. '
        '`--report-only` redraws the saved evidence. Default output is outside the repository.', '']
    if parity:
        lines += ['Device selection agrees with the CPU checkpoint. Maximum actual-output difference: '+f'{max(parity):.3g}.', '']
    (output/'report.md').write_text('\n'.join(lines))
    accessible_report(output/'report.md')
    files = [Path(__file__), *[Path('src/warpSPHBoundaries')/p for p in (
        'scene/implicitBodies.py', 'scene/scene.py', 'scene/sdf_guard.py',
        'scene/sdf_warp.py',
        'edge/curvature.py', 'validation/curvature.py', 'validation/sdf_accuracy.py')]]
    (output/'source-hashes.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2))
    print(json.dumps({k: summary[k] for k in ('device', 'accepted', 'rejected', 'passed_sweeps', 'max_device_difference')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=evidence_path('curvature-corrections-fused-cpu'))
    parser.add_argument('--reference-cache', type=Path, default=evidence_path('sdf-accuracy-cpu'))
    parser.add_argument('--compare-to', type=Path)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--verify-backend-from', type=Path,
                        help='Re-execute saved queries and retain their validated references and model budgets')
    args = parser.parse_args()
    if not args.report_only:
        if args.verify_backend_from:
            verify_backend(args.verify_backend_from, args.output, args.device)
        else:
            measure(args.output, args.device, args.reference_cache)
        ripple_stress(args.output, args.device)
    report(args.output, args.compare_to)
