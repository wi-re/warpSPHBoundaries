"""Audit actual accepted Scene SDF moments against true curved-domain integrals."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from warpSPHBoundaries.validation.paths import evidence_path

import numpy as np
import torch

from warpSPHBoundaries.scene.scene import SdfRep, _sdfTier3
from warpSPHBoundaries.validation.adversarial import SUPPORTED, scene_channels
from warpSPHBoundaries.validation.atlas import CHANNELS, LABELS, _pyplot
from warpSPHBoundaries.validation.error_colors import EPS32, error_style, error_colorbar
from warpSPHBoundaries.validation.distance_only import halfplane_channels
from warpSPHBoundaries.validation.report_view import accessible_report
from warpSPHBoundaries.validation.sdf_accuracy import (
    accepted_channel_gate, area_reference, boundary_reference_mp, local_model_reference, shape_coefficients,
    smooth_cases)


KERNELS = ('cubic', 'w2', 'w4', 'w6')
DISTANCES = np.sort(np.r_[np.linspace(-.75, .9, 17), 0.])
RESOLUTIONS = (16, 32, 64)
SCHEMA = 'accepted-sdf-v1'
MODEL_VERSION = 'planar-lambda-endpoint-v2'


def signal_scales(target):
    """One signal scale per scalar/vector/tensor, avoiding division by odd zeros."""
    scale = np.empty(12)
    for group in ((0,), (1, 2), (3, 4, 5), (6, 7), (8, 9, 10, 11)):
        scale[list(group)] = max(float(np.linalg.norm(target[:, group], axis=1).max()), 1e-30)
    return scale


def references(case, kernel, output, cache=None):
    path = output/'references'/kernel/f'{case.name}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    shared = cache/'references'/kernel/path.name if cache else None
    source = shared if shared and shared.exists() else (path if path.exists() else None)
    spec = json.dumps(asdict(case), sort_keys=True)
    saved = None
    if source and source.exists():
        with np.load(source) as a:
            if str(a['schema']) != SCHEMA or str(a['geometry']) != spec:
                raise ValueError(f'incompatible reference cache: {source}')
            data = dict(a)
        np.testing.assert_allclose(data['independent_area'], data['independent'], atol=3e-12, rtol=3e-11)
        if np.array_equal(data['distances'], DISTANCES):
            if source != path:
                np.savez_compressed(path, **data)
            return data
        if not np.isin(data['distances'], DISTANCES).all():
            raise ValueError(f'incompatible query catalogue: {source}')
        saved = data  # extend a sweep without recomputing identical integrals
    points = case.points(DISTANCES)
    target, refinement, outer = [], [], []
    for distance, point in zip(DISTANCES, points):
        match = np.flatnonzero(saved['distances'] == distance) if saved is not None else []
        if len(match):
            j = match[0]
            target.append(saved['target'][j])
            refinement.append(saved['reference_refinement'][j])
            outer.append(saved['reference_outer_estimate'][j])
            continue
        fine, error = area_reference(case, point, kernel, order=96, tolerance=1e-12)
        coarse, error_coarse = area_reference(case, point, kernel, order=48, tolerance=2e-12)
        target.append(fine)
        refinement.append(np.abs(fine-coarse))
        outer.append(error+error_coarse)
    target, refinement, outer = map(np.asarray, (target, refinement, outer))
    # One independent all-moment spot check per geometry/kernel at a fluid-side
    # offset, including variable curvature, rotation and cavity conventions.
    oracle_point = case.points([.3])[0]
    if saved is None:
        independent = boundary_reference_mp(case, oracle_point, kernel, dps=45)
        oracle_fine = boundary_reference_mp(case, oracle_point, kernel, dps=60)
        direct, direct_error = area_reference(case, oracle_point, kernel, order=96, tolerance=1e-12)
        precision_difference = np.abs(independent-oracle_fine)
    else:
        oracle_fine, direct = saved['independent'], saved['independent_area']
        direct_error = saved['independent_outer_estimate']
        precision_difference = saved['independent_precision_difference']
    np.testing.assert_allclose(direct, oracle_fine, atol=3e-12, rtol=3e-11)
    p, n, _, k, variation = case.frame
    normal = np.broadcast_to(n@case.matrix.T, (len(points), 2))
    exact_model = local_model_reference(DISTANCES, normal, np.full(len(points), k), kernel, 128)
    model_coarse = local_model_reference(DISTANCES, normal, np.full(len(points), k), kernel, 64)
    # Local mass's tangential derivative includes a curvature-variation term;
    # production g0 uses only d-lambda times n. Report this inconsistency.
    tangent_derivative = np.array([(shape_coefficients(d, kernel, 128)[0]+2*k*shape_coefficients(d, kernel, 128)[1])
                                  *variation/(1+d*k) for d in DISTANCES])
    data = dict(schema=SCHEMA, geometry=spec, distances=DISTANCES, points=points, target=target,
                reference_refinement=refinement, reference_outer_estimate=outer,
                reference_uncertainty=refinement+outer[:, None]+32*np.finfo(float).eps,
                exact_model=exact_model, model_reference_refinement=np.abs(exact_model-model_coarse),
                exact_normal=normal, exact_curvature=k, curvature_variation=variation,
                local_mass_tangential_derivative=tangent_derivative,
                independent_point=oracle_point, independent=oracle_fine,
                independent_precision_difference=precision_difference,
                independent_area=direct, independent_difference=np.abs(direct-oracle_fine),
                independent_outer_estimate=direct_error)
    np.savez_compressed(path, **data)
    return data


def measure(output, device='cpu', cache=None):
    output.mkdir(parents=True, exist_ok=True)
    for case in smooth_cases():
        refs = {kernel: references(case, kernel, output, cache) for kernel in KERNELS}
        for resolution in RESOLUTIONS:
            paths = [output/'data'/kernel/f'{case.name}-h{resolution}.npz' for kernel in KERNELS]
            if all(path.exists() for path in paths):
                same_queries = True
                for path in paths:
                    with np.load(path) as a:
                        if str(a['schema']) != SCHEMA or str(a['device']) != device or str(a['geometry']) != json.dumps(asdict(case), sort_keys=True):
                            raise ValueError(f'incompatible measurement cache: {path}')
                        same_queries &= np.array_equal(a['distances'], DISTANCES)
                        same_queries &= str(a.get('model_version', 'before-endpoint-fix')) == MODEL_VERSION
                if same_queries:
                    print(f'cached {case.name}, h/{resolution}', flush=True)
                    continue
            points = case.points(DISTANCES)
            lo, hi = points.min(0)-1.35, points.max(0)+1.35
            rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy()), dtype=torch.float64),
                                     lo, hi, 1/resolution).to(device)
            # Diagnostic helper is CPU-only. The raw model below uses precisely
            # the same table/jet operations on the selected production device.
            position = torch.tensor(points, dtype=torch.float64, device=device)
            supports = torch.ones(len(points), dtype=torch.float64, device=device)
            d, n, k, norm, denominator, covered = rep._signed(position)
            from warpSPHBoundaries.scene.scene import tier3Table, _planar_moments
            for kernel, path in zip(KERNELS, paths):
                ref = refs[kernel]
                table = tier3Table(kernel, device)
                raw = np.full((len(points), 12), np.nan)
                mass, derivative = table.lam(d, k)
                first, cov = _planar_moments(rep, position, supports, kernel, device)
                cpu = lambda x: x.detach().cpu().numpy()
                raw[:, 0], raw[:, 1:3] = cpu(mass), cpu(first)
                raw[:, 6:8], raw[:, 8:] = cpu(derivative[:, None]*n), cpu(cov).reshape(-1, 4)
                fallback = cpu(_sdfTier3(rep, position, supports, kernel, device)[2])
                accepted = ~fallback
                # Evaluate actual Scene only where the local model is selected.
                # No polygon fallback exists in this audit, and no target is
                # substituted into the values array at rejected queries.
                values = np.full_like(raw, np.nan)
                if accepted.any():
                    values[accepted] = scene_channels(rep, points[accepted], kernel, device)
                    np.testing.assert_allclose(values[accepted][:, SUPPORTED], raw[accepted][:, SUPPORTED], atol=5e-14, rtol=5e-14)
                same_model = local_model_reference(cpu(d), cpu(n), cpu(k), kernel, 128)
                coarse = local_model_reference(cpu(d), cpu(n), cpu(k), kernel, 64)
                numerical = raw.astype(np.longdouble)-same_model
                reconstruction = same_model-ref['exact_model']
                model = ref['exact_model']-ref['target']
                total = values.astype(np.longdouble)-ref['target']
                budget = numerical+reconstruction+model
                np.testing.assert_allclose(budget[accepted][:, SUPPORTED], total[accepted][:, SUPPORTED], atol=1e-17, rtol=1e-13)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, schema=SCHEMA, model_version=MODEL_VERSION, geometry=json.dumps(asdict(case), sort_keys=True),
                    device=device, resolution=resolution, points=points, distances=DISTANCES, channels=CHANNELS,
                    target=ref['target'], raw=raw, values=values, accepted=accepted, fallback=fallback,
                    sampled_distance=cpu(d), sampled_normal=cpu(n), sampled_curvature=cpu(k),
                    sampled_gradient_norm=cpu(norm), sampled_denominator=cpu(denominator), covered=cpu(covered),
                    same_model=same_model, exact_model=ref['exact_model'], numerical=numerical,
                    reconstruction=reconstruction, model=model, total=total,
                    model_reference_refinement=np.abs(same_model-coarse),
                    reference_uncertainty=ref['reference_uncertainty'], scale=signal_scales(ref['target']))
            print(f'measured {case.name}, h/{resolution}, {int(accepted.sum())}/{len(points)} accepted (last kernel)', flush=True)


def compare_device_checkpoint(output, other):
    """Compare actual outputs/selection, never use the other device as truth."""
    rows = []
    for case in smooth_cases():
        for kernel in KERNELS:
            for resolution in RESOLUTIONS:
                relative = Path('data')/kernel/f'{case.name}-h{resolution}.npz'
                if not (other/relative).exists():
                    raise ValueError(f'missing comparison measurement: {other/relative}')
                with np.load(output/relative) as a, np.load(other/relative) as b:
                    for key in ('geometry', 'points', 'accepted', 'target'):
                        np.testing.assert_array_equal(a[key], b[key])
                    active = a['accepted']
                    x, y = a['values'][active][:, SUPPORTED], b['values'][active][:, SUPPORTED]
                    np.testing.assert_allclose(x, y, atol=2e-12, rtol=2e-12)
                    rows.append(dict(case=case.name, kernel=kernel, resolution=resolution,
                                     max_absolute_difference=float(np.max(np.abs(x-y))) if active.any() else None))
    parity = dict(other=str(other), same_selection=True, records=rows,
                  max_absolute_difference=max(r['max_absolute_difference'] or 0 for r in rows))
    (output/'device-comparison.json').write_text(json.dumps(parity, indent=2))


def report(output):
    plt = _pyplot()
    cases = smooth_cases()
    records, arrays = [], {}
    for case in cases:
        for kernel in KERNELS:
            for resolution in RESOLUTIONS:
                path = output/'data'/kernel/f'{case.name}-h{resolution}.npz'
                with np.load(path) as a:
                    a = dict(a)
                if str(a['schema']) != SCHEMA:
                    raise ValueError('incompatible study data')
                arrays[case.name, kernel, resolution] = a
                mask = a['accepted']
                error = np.abs(a['total'])
                estimate = a['reference_uncertainty']
                scale = a['scale']
                passing = accepted_channel_gate(a['total'], estimate, scale, mask, EPS32)
                passing[[3, 4, 5]] = False
                records.append(dict(case=case.name, kernel=kernel, resolution=resolution,
                    device=str(a['device']), accepted=int(mask.sum()), rejected=int((~mask).sum()),
                    channel_pass=passing.tolist(), all_supported_pass=bool(mask.any() and passing[list(SUPPORTED)].all()),
                    max_accepted_absolute=float(np.nanmax(error[mask])) if mask.any() else None,
                    max_accepted_relative=float(np.nanmax(error[mask]/scale)) if mask.any() else None,
                    budget_closure=float(np.nanmax(np.abs((a['numerical']+a['reconstruction']+a['model']-a['total'])[mask]))) if mask.any() else None))

    titles = {'total': 'Executed Scene: total error where accepted',
              'model': 'Exact d/n/curvature: local-model error',
              'reconstruction': 'Sampled jets: reconstruction contribution',
              'numerical': 'Table/evaluation discrepancy from the same model',
              'reference_uncertainty': 'Original-domain quadrature: empirical uncertainty'}
    for component, title in titles.items():
        fig, axes = plt.subplots(2, 6, figsize=(20, 10), constrained_layout=True)
        for j, ax in enumerate(axes.flat):
            data = np.array([np.abs(arrays[c.name, 'w4', 64][component][:, j])/arrays[c.name, 'w4', 64]['scale'][j] for c in cases], dtype=float)
            if component != 'reference_uncertainty':
                accepted = np.array([arrays[c.name, 'w4', 64]['accepted'] for c in cases])
                data = np.where(accepted, data, np.nan)
            im = ax.pcolormesh(DISTANCES, np.arange(len(cases)), np.ma.masked_invalid(data),
                              shading='nearest', **error_style())
            ax.set_ylim(len(cases)-.5, -.5)
            order_note = ('curvature²' if j == 0 else 'planar' if j in (1, 2, 8, 9, 10, 11)
                          else 'normal gradient only' if j in (6, 7) else 'not exposed')
            ax.set(title=LABELS[j]+'\n'+order_note, xlabel='$d/h$')
            ax.set_yticks(range(len(cases)), [c.name for c in cases] if j % 6 == 0 else [])
            if j in (3, 4, 5) and component != 'reference_uncertainty':
                ax.text(.5, .5, 'production SDF\nnot implemented', ha='center', va='center', transform=ax.transAxes)
        error_colorbar(fig, im, ax=list(axes.flat), detailed=True, shrink=.7)
        domain = 'all original-domain references' if component == 'reference_uncertainty' else 'accepted queries only; grey = rejected or unavailable'
        fig.suptitle(title+f'\nw4; spacing h/64; {domain}. Block-signal-relative; empirical, not certified bounds.', fontsize=12)
        fig.savefig(output/f'{component}-w4.png', dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    for resolution, ax in zip(RESOLUTIONS, axes):
        mask = np.array([arrays[c.name, 'w4', resolution]['accepted'] for c in cases])
        im = ax.pcolormesh(DISTANCES, np.arange(len(cases)), mask, shading='nearest', cmap='Blues', vmin=0, vmax=1)
        ax.set_ylim(len(cases)-.5, -.5)
        ax.set_yticks(range(len(cases)), [c.name for c in cases])
        ax.set(title=f'Grid spacing h/{resolution}', xlabel='$d/h$')
    fig.suptitle('Production domain guard: blue = accepted curved model; white = fallback requested\nAcceptance checks sampled-field validity, not integral accuracy. No fallback values are evaluated in this study.')
    fig.savefig(output/'acceptance.png', dpi=300)
    plt.close(fig)

    # Explicit per-channel gate aggregated over four kernels, all accepted
    # distances. Grey rows with no accepted query do not pass vacuously.
    fig, axes = plt.subplots(1, 3, figsize=(19, 7), constrained_layout=True)
    gate_tables = []
    for resolution, ax in zip(RESOLUTIONS, axes):
        gate = np.full((len(cases), 12), np.nan)
        for i, case in enumerate(cases):
            rows = [r for r in records if r['case'] == case.name and r['resolution'] == resolution and r['accepted']]
            if rows:
                gate[i] = np.all([r['channel_pass'] for r in rows], axis=0)
            gate[i, [3, 4, 5]] = np.nan
        gate_tables.append(gate)
        from matplotlib.colors import ListedColormap
        im = ax.imshow(np.ma.masked_invalid(gate), aspect='auto', vmin=0, vmax=1,
                       cmap=ListedColormap(['#c5410d', '#112b52']).with_extremes(bad='#d9dde3'))
        ax.set_yticks(range(len(cases)), [c.name for c in cases])
        ax.set_xticks(range(12), CHANNELS, rotation=70)
        ax.set_title(f'h/{resolution}: ε32 gate, all four kernels')
    fig.suptitle('Accepted-query accuracy matrix: navy = measured pass; orange = fail; grey = missing/no accepted samples\nReference estimate included; vector/tensor signal scales prevent division by symmetry zeros. No continuous-domain certificate.')
    fig.savefig(output/'acceptance-matrix.png', dpi=300)
    plt.close(fig)

    # Inspect curvature variation at a zero-curvature closest point separately
    # from the smooth circle with constant curvature.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for name, color in zip(('inflection-b0.03', 'inflection-b0.12', 'inflection-b0.3'), ('#1764ab', '#228557', '#8d4aa8')):
        case = next(c for c in cases if c.name == name)
        a = arrays[name, 'w4', 64]
        with np.load(output/'references'/'w4'/f'{name}.npz') as r:
            tangent = case.frame[2]@case.matrix.T
            axes[0].plot(DISTANCES, np.asarray(a['target'][:, 6:8]@tangent, float), color=color, label=f'{name}: true integral')
            axes[0].plot(DISTANCES, np.asarray(r['local_mass_tangential_derivative'], float), '--', color=color, label=f'{name}: ∂t local mass')
        axes[1].plot(DISTANCES, np.asarray(a['raw'][:, 6:8]@tangent, float), color=color, label=name)
        axes[2].semilogy(DISTANCES, np.maximum(np.abs(a['total'][:, 6]), 1e-18), color=color, label=name)
    axes[0].set(title='Tangential gradients and missing curvature variation', ylabel='$h G_t$')
    axes[1].set(title='Shipped gradient: only the normal component', ylabel='$h G_t$')
    axes[2].axhline(EPS32, color='black', ls=':', label='ε32 absolute guide')
    axes[2].set(title='Accepted Scene tangential gradient error', ylabel='Absolute error')
    for ax in axes:
        ax.set_xlabel('$d/h$')
        ax.legend(fontsize=6)
        ax.grid(alpha=.2)
    fig.savefig(output/'inflection-gradient.png', dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    for ax, name in zip(axes, ('circle-R3.0', 'circle-R24.0', 'ellipse-oblique', 'inflection-b0.3')):
        for component, color in zip(('model', 'reconstruction', 'numerical', 'total'), ('#8d4aa8', '#228557', '#b55b08', '#1764ab')):
            values = [max(float(np.max(np.abs(arrays[name, kernel, resolution][component][:, 0])))
                          for kernel in KERNELS) for resolution in RESOLUTIONS]
            ax.loglog(RESOLUTIONS, values, '-o', color=color, label=component)
        ax.axhline(EPS32, color='black', ls=':', label='ε32 absolute guide')
        ax.set(title=name, xlabel='Samples per support h/spacing', ylabel='Maximum absolute mass error')
        ax.set_xticks(RESOLUTIONS, [str(r) for r in RESOLUTIONS])
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle(f'Grid refinement separates SDF reconstruction from the local-model error floor\nMass only; maxima over {len(DISTANCES)} distances and four kernels. Contributions are signed in saved data; plotted maxima need not add.')
    fig.savefig(output/'resolution-mass.png', dpi=300)
    plt.close(fig)

    from warpSPHBoundaries.scene.implicitBodies import Tier3
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for kernel in KERNELS:
        table = Tier3(kernel, 'cpu')
        q0 = float(table.q[0])
        distance = np.linspace(-2*q0, 2*q0, 201)
        first = table.planar_moments(torch.tensor(distance))[2].numpy()
        gap = np.maximum(q0-np.abs(distance), 0)
        missing = float(table.F[0][0])*gap-float(table.dF[0][0])*gap**2/2
        target = -halfplane_channels(distance, kernel=kernel, dtype=np.longdouble)[:, 2]
        # Reconstruct the old clamped-antiderivative algorithm using the same
        # unchanged table. The actual before-fix Scene checkpoint is archived.
        for ax, error in zip(axes, (first-missing-target, first-target)):
            ax.semilogy(distance/q0, np.maximum(np.abs(error), 1e-18), label=kernel)
    for ax, title in zip(axes, ('Former endpoint clamp (reconstructed)', 'Corrected antiderivative continuation')):
        ax.axhline(EPS32, color='black', ls=':', label='ε32 absolute guide')
        ax.set(title=title, xlabel='Signed distance / first table node qmin', ylabel='Absolute first-moment evaluation error', ylim=(1e-14, 2e-6))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Numerical endpoint defect repaired independently of curvature-model error\nPlanar normal first moment; reference uses extended-precision closed forms. Tables unchanged.')
    fig.savefig(output/'endpoint-fix.png', dpi=300)
    plt.close(fig)

    oracle_rows = []
    for case in cases:
        for kernel in KERNELS:
            with np.load(output/'references'/kernel/f'{case.name}.npz') as r:
                oracle_rows.append(dict(case=case.name, kernel=kernel,
                    max_absolute=float(r['independent_difference'].max()),
                    max_precision_difference=float(r['independent_precision_difference'].max()),
                    max_relative=float((r['independent_difference']/signal_scales(r['target'])).max())))
    component_maxima = {component: [max(float(np.nanmax(np.abs(arrays[c.name, kernel, 64][component][:, group])))
                                      for c in cases for kernel in KERNELS)
                                  for group in ((0,), (1, 2), (6, 7), (8, 9, 10, 11))]
                        for component in ('model', 'reconstruction', 'numerical', 'total', 'reference_uncertainty')}
    summary = dict(schema=SCHEMA, model_version=MODEL_VERSION, channels=CHANNELS, kernels=KERNELS, resolutions=RESOLUTIONS,
                   distances=DISTANCES.tolist(), geometries=[asdict(c) for c in cases], records=records,
                   independent_checks=oracle_rows, component_maxima_h64=component_maxima,
                   accepted=sum(r['accepted'] for r in records), rejected=sum(r['rejected'] for r in records),
                   passed_records=sum(r['all_supported_pass'] for r in records),
                   max_budget_closure=max(r['budget_closure'] or 0 for r in records))
    before_device = 'cuda' if records[0]['device'].startswith('cuda') else 'cpu'
    before_path = output.parent/'sdf-accuracy-before-wall-fix'/before_device/'summary.json'
    if before_path.exists():
        before = json.loads(before_path.read_text())
        summary['endpoint_fix'] = dict(before_max_first_moment_numerical=before['component_maxima_h64']['numerical'][1],
                                       after_max_first_moment_numerical=component_maxima['numerical'][1])
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# Accepted smooth-SDF accuracy audit', '',
        f"Actual production Scene evaluation on **{summary['accepted']} accepted query/kernel/grid records**; "
        f"{summary['rejected']} rejected records remain explicitly uncomputed. No fallback result or reference is substituted into a rejected row.", '',
        f"**{summary['passed_records']}/{len(records)} geometry × kernel × grid sweeps meet the ε32 gate on all nine implemented moments.** "
        'The domain guard permits these smooth examples, but it does not establish the requested integral accuracy.', '',
        'The sample contains 13 smooth geometries: convex circles R/h=3,6,12,24; a concave cavity; rotated circles/ellipses; '
        'variable-curvature ellipses; and cubic walls with inflection or nonzero curvature plus curvature variation. '
        f'Each uses four kernels, {len(DISTANCES)} signed offsets (including the wall itself) and three SDF resolutions. This is a CPU/selected-device accuracy audit, not a throughput benchmark.', '',
        '## What the guard establishes', '',
        'The production sampled-cell guard checks denominator conditioning, gradient norm, curvature magnitude and grid coverage over the support. '
        'It does not estimate integral truncation or missing moment terms. Acceptance alone is not an accuracy certificate.', '',
        '![Accepted supports](acceptance.png)', '', '## Per-moment acceptance matrix', '',
        'The initial gate is |error| + reference estimate ≤ ε32 times the original-geometry signal, on every accepted sample in a row and all four kernels. '
        'A scalar, vector or tensor uses its maximum norm over the signed-distance sweep as its signal scale; this avoids division by zero for odd components. '
        'Rejected queries do not pass vacuously. Second scalar moments remain unsupported in production SDF even though the oracle evaluates them.', '',
        'Mass and gradient use the quadratic curvature series; m10/m01 and g1 remain planar. The methods therefore have different expected truncation orders.', '',
        '![Per-moment precision gate](acceptance-matrix.png)', '',
        '| Geometry | h/16 accepted / all 9 pass | h/32 accepted / all 9 pass | h/64 accepted / all 9 pass |',
        '|---|---|---|---|']
    for case in cases:
        cells = []
        for resolution in RESOLUTIONS:
            rows = [r for r in records if r['case'] == case.name and r['resolution'] == resolution]
            cells.append(f"{sum(r['accepted'] for r in rows)}/{len(KERNELS)*len(DISTANCES)} / {'yes' if all(r['all_supported_pass'] for r in rows) else 'no'}")
        lines.append('| '+case.name+' | '+' | '.join(cells)+' |')
    lines += ['', '## Total error on accepted queries', '',
        'Each figure retains the twelve-channel tier order. The shared colour map treats all errors >1 as unusable; '
        'ε64 and ε32 are visual reference levels. Raw absolute and signed arrays are retained alongside the plotted signal-relative values.', '',
        '![Executed Scene total errors](total-w4.png)', '', '## Separated error contributions', '',
        'The signed decomposition is: **table/evaluation + SDF reconstruction + local-model geometry = total**. '
        'The same-model reference uses extended precision planar closed forms plus an independent one-dimensional shape derivative for curvature coefficients. '
        'This identifies interpolation/evaluation discrepancies but is not a certified rounding-error bound. '
        f"The largest saved budget closure residual is {summary['max_budget_closure']:.3g}.", '',
        '| Finest-grid contribution, max absolute over all geometries/kernels/queries | m00 | m1 components | G components | g1 components |',
        '|---|---|---|---|---|',
        *['| '+component+' | '+' | '.join(f'{value:.3g}' for value in values)+' |' for component, values in component_maxima.items()], '',
        '![Exact-jet local-model errors](model-w4.png)', '',
        '![SDF reconstruction errors](reconstruction-w4.png)', '',
        '![Table and numerical evaluation discrepancies](numerical-w4.png)', '',
        '![Mass grid-refinement error contributions](resolution-mass.png)', '',
        '## Numerical endpoint correction', '',
        'Exact-on-wall samples exposed a separate first-moment implementation defect. The cumulative planar table starts at qmin=10⁻⁶. '
        'F0 was extended below that node, but its integral Lambda was clamped, omitting about qmin/2 at the wall. '
        'The production correction integrates the same linear F0 continuation down to the requested distance. It restores the zero wall slope of the even first moment; the table data and curvature model are unchanged.', '',
        *([f"Across the h/64 audit, the maximum first-moment table/evaluation discrepancy falls from "
           f"{summary['endpoint_fix']['before_max_first_moment_numerical']:.3g} to "
           f"{summary['endpoint_fix']['after_max_first_moment_numerical']:.3g}. "
           f"[Archived before-fix Scene report](../sdf-accuracy-before-wall-fix/{before_device}/report.md).", ''] if 'endpoint_fix' in summary else []),
        'The before curve below reconstructs the former endpoint algorithm using the unchanged table. '
        +('The archived measurements retain actual pre-fix Scene results.' if 'endpoint_fix' in summary else 'No original pre-fix Scene checkpoint was supplied to this run.'), '',
        '![First-moment endpoint correction](endpoint-fix.png)', '',
        '## Inflection: zero curvature does not imply a plane', '',
        'At the cubic inflection κ=0, the boundary changes curvature across the support. The true tangential mass gradient is nonzero. '
        'The shipped gradient is λ_d n and omits the tangential contribution from spatial variation of κ in its own mass model. '
        'Dashed lines show (F1+2κF2) κ_s/(1+dκ), the tangential derivative of the local mass model, not a replacement solver or a full high-order geometry correction.', '',
        'Small nonzero shipped tangential components in the middle panel arise from sampled-normal reconstruction; exact-jet production gradients at these inflections have zero tangential component.', '',
        '![Inflection gradient consistency](inflection-gradient.png)', '',
        '## Reference validation and its limits', '',
        'Targets integrate the actual ellipse or cubic-graph solid using Cartesian area quadrature, with analytic support and kernel-knot cuts. '
        'Independent boundary flux/integration-by-parts checks use mpmath at 45 and 60 digits, all twelve moments, one additional fluid-side point per geometry and kernel. '
        'This is 52 independent all-moment spot checks, not a high-precision certificate for every sample.', '',
        f"Maximum independent spot-check discrepancy: {max(r['max_absolute'] for r in oracle_rows):.3g} absolute, "
        f"{max(r['max_relative'] for r in oracle_rows):.3g} signal-relative. "
        f"Maximum discrepancy between stored 45/60-digit outputs: {max(r['max_precision_difference'] for r in oracle_rows):.3g}. "
        'Oracle outputs are stored as longdouble; agreement below its storage precision is not a zero-error certificate.', '',
        'The reference estimate combines 48/96-node inner refinement, adaptive outer estimates and a floating-point floor. '
        'It is empirical. Reference refinement is shown separately; it is not counted as an SDF model defect.', '',
        '![Original-domain quadrature uncertainty](reference_uncertainty-w4.png)', '',
        '## Reproduce', '',
        'Run `python scripts/studies/sdf_accuracy.py --device cpu`. Use `--report-only` to regenerate figures without changing measurements. '
        'Use `--device cuda:0 --output "$EVIDENCE/sdf-accuracy-cuda" --reference-cache "$EVIDENCE/sdf-accuracy-cpu"` '
        'for an executed GPU checkpoint sharing only the original-domain references. '
        'Each raw NPZ stores geometry, selection, sampled jets, Scene output, exact-jet model, uncertainties and all signed error terms.', '']
    if (output/'device-comparison.json').exists():
        parity = json.loads((output/'device-comparison.json').read_text())
        lines += ['## Device agreement', '',
                  f"Compared with `{parity['other']}`: selection agrees on all saved records; "
                  f"maximum absolute difference on actual accepted Scene moments is {parity['max_absolute_difference']:.3g}. "
                  'This is a device-parity check; the independent curved-domain integrals remain the accuracy reference.', '']
    (output/'report.md').write_text('\n'.join(lines))
    accessible_report(output/'report.md')
    import warpSPHBoundaries.validation.sdf_accuracy as implementation
    provenance = dict(schema=SCHEMA, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      reference_module_sha256=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),
                      device=records[0]['device'], reference_policy='Cartesian48/96 + adaptive outer; boundary45/60 spot checks')
    (output/'render-provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps({k: summary[k] for k in ('accepted', 'rejected', 'passed_records', 'max_budget_closure')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=evidence_path('sdf-accuracy-cpu'))
    parser.add_argument('--reference-cache', type=Path)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if not args.report_only:
        parser.error('This archived mixed-model checkpoint predates curvature corrections. Run scripts/studies/curvature_corrections.py for current production; --report-only redraws the historical cache.')
    if args.reference_cache:
        compare_device_checkpoint(args.output, args.reference_cache)
    report(args.output)
