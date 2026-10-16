"""Generate a paginated Markdown survey from cached audit measurements.

python -m warpSPHBoundaries.validation.audit_report

Rendering never runs a boundary evaluator or changes the numerical inputs.
New cases and methods in summary.json are discovered rather than whitelisted.
"""
import argparse
import hashlib
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

from warpSPHBoundaries.validation.paths import evidence_path

import numpy as np

from .atlas import CHANNELS, LABELS, _pyplot
from .error_colors import EPS32, EPS64, BAND_LEGEND, error_style, error_colorbar


METRICS = OrderedDict((
    ('absolute', ('max_absolute_error', 'Maximum absolute error')),
    ('relative', ('max_error_over_signal_scale', 'Error / per-case channel signal')),
))


def scheme_info(name, study='polygon'):
    """A priori expectations, separate from measured accuracy classifications."""
    if name == 'split-f64':
        return ('diagnostic; cancellation', 'Legacy split analytic expression. Exact polygon geometry, but cancellation makes this a diagnostic comparator.')
    if name.startswith('closed-'):
        return ('exact; cancellation risk', 'Closed-form polygon identity: no geometric approximation. Expect strong accuracy on resolved elements; cancellation and input rounding can dominate tiny elements, especially in float32.')
    if name.startswith(('gauss-', 'cheb-')):
        return ('quadrature; resolution', 'Integrates the original polygon geometry. Expect accuracy when the edge integrand is resolved; fixed nodes/panels and finite arithmetic give no uniform guarantee on grazing cuts or thin elements.')
    if name == 'P2-edge':
        return ('exact geometry; conditioning', 'Triangle volume moments through P2 FEM edge weights. No geometry surrogate; small or ill-conditioned triangles can amplify arithmetic error.')
    if name in ('scene-device','fused-device-mono','fused-device-cheb'):
        tag = 'closed + conditioned area' if name != 'fused-device-cheb' else 'edge quadrature + conditioned area'
        return (tag, 'Executed Scene/FusedWall surface backend, all twelve moments. Single small/sliver triangles use device area integration with empirical inner/outer refinement estimates and explicit failure status. Other polygons retain the existing edge route. The geometry is unchanged; fixed edge quadrature and finite arithmetic still require domain-specific validation. This label is an expectation, not a universal accuracy certificate.')
    if name == 'conditioned-cpu':
        return ('exact geometry; adaptive fallback', 'CPU candidate retaining the closed polygon identity on resolved elements and using conditioned triangle-area quadrature for small/sliver triangles. All twelve moments are evaluated. Empirical refinement estimates are not certified bounds; this candidate is not wired into Scene/CUDA.')
    if name == 'P2-hybrid':
        return ('hybrid; sliver risk', 'Triangle FEM edge/area hybrid. Current CPU implementation includes an adaptive small/sliver fallback; preserved baseline datasets predate it. Read checkpoint provenance before assessing the measured branch.')
    if name == 'SDF-forced':
        return ('poor near corners/sheets', 'Current forced model uses one order-two curvature/variation expansion for all twelve moments; archived checkpoints can predate this correction. This is not the distance-only half-plane model. A triangle can have an exact SDF, but surface-curvature recovery at an exterior vertex divides by zero even with unit gradient. One local smooth surface also misses medial axes and multiple faces inside support. Sampling adds separate geometry and differential-reconstruction errors.')
    if name == 'SDF-distance-only':
        return ('flat model; no curvature pole', 'Semi-analytic half-plane reference: mass depends only on signed distance, and oriented moments also use a unit SDF normal. All twelve moments use closed forms; no curvature recovery, series, validity selection or fallback is applied. Exact for a single plane. Corners and thin/multiple sheets produce geometric bias; medial axes can leave the normal undefined. Nonfinite oriented values there mean unavailable directions, not a curvature blow-up. Precision comparisons are empirical, not certified bounds.')
    if name == 'SDF-contour':
        return ('geometry; grid limited', 'Integrates the oriented marching-squares contour. Relative to the original polygon this measures representation-induced integral error, without a local flat/curvature model. Subgrid features may disappear.')
    if name == 'SDF-cell-guarded':
        return ('conservative domain; fallback dependent', 'Executed Scene with interval bounds on every blended-jet cell in the support bounding square, before the curvature series. Rejects poles, unresolved gradients and incomplete sampling coverage; uses the supplied original polygon fallback. This is a sampled-field domain guard, not a certified truncation-error bound. Read branch counts: accurate rejected queries validate the fallback, not the local model.')
    if name.startswith('fused-') or name == 'scene-polygon':
        return ('exact; implementation gate', 'Production polygon evaluator. The geometry is exact; large edge/indicator jumps are implementation defects, not a justified geometry approximation.')
    if name == 'SDF-guarded' or 'guarded' in name:
        return ('guarded; fallback dependent', 'Executed scene policy combines local SDF evaluation and explicit polygon fallback. Expected accuracy depends on both the acceptance policy and the fallback. Its executed mixed path has no complete cached error split.')
    if name.startswith('particles-'):
        return ('quadrature; spacing limited', 'Actual particle lattice integration. A fixed spacing cannot resolve arbitrarily small or thin geometry; missing features and sampling bias are expected.')
    if name.startswith('T3-') or name == 'ellipse-SDF-forced':
        return ('local model; domain limited', 'Local curvature model; current SDF adapters use the shared order-two family, while archived studies may retain planar first/mixed moments. Expected only for a smooth, isolated, slowly varying boundary over the kernel support. Check checkpoint provenance and the moment order used. Forced extrapolations are diagnostic.')
    if name.startswith('T4-disk-'):
        return ('series; restricted domain', 'Small-radius disk series. Only m00 is implemented here. Kernel-piece/support-rim rejections are outside the declared domain; the remaining values still need a truncation and evaluation budget.')
    if name.startswith('T4-strip-'):
        return ('series; thin exterior', 'Thin-strip width expansion, m00 only. Expected for small thickness and exterior queries d>a. Forced interior or wide-strip samples are not a general validity test of an exterior asymptotic model.')
    if name.startswith('polygon-') or name == 'ellipse-fused-polygon':
        return ('geometry surrogate', 'Finite polygon approximation to a curved boundary. The integral of that same polygon separates evaluator error from geometry-surrogate error against the curved reference.')
    if name.startswith('arc-'):
        return ('quadrature; boundary gate', 'Circle-arc quadrature on the intended disk geometry. Boundary and closest-approach conventions need independent checks; agreement between node counts alone cannot establish accuracy. Small moments and high-degree profiles also need conditioning checks. Read the progress tracker to distinguish preserved pre-fix measurements from new checkpoints.')
    if name.endswith('-lookup'):
        return ('table; fit/panel accuracy', 'Fitted lookup on the intended geometry. Accuracy depends on interpolation, panel transitions and limiting cases. No geometric approximation is introduced by replacing a disk/box with the same disk/box table.')
    if name.startswith('actual-tier-switch'):
        return ('hybrid; transition gate', 'Executed implicit scene selection between local curvature, small-disk series and polygon fallback. Check both sides of each transition; a tier threshold is not an accuracy certificate.')
    return ('expectation unspecified', 'New scheme discovered in the cached audit. Supply its expected regime and error model in scheme_info; measurements remain included.')


def _slug(value):
    return re.sub(r'[^a-zA-Z0-9_-]+', '-', str(value)).strip('-').lower()


def collections(summary):
    """Normalize polygon and specialized records without merging close parameters."""
    groups = OrderedDict()
    groups['polygon'] = [dict(r, source=f"{r['kernel']}/{r['case']}/data.npz") for r in summary['records']]
    for study in summary.get('specialized', []):
        kind, kernel = study['study'], study['kernel']
        for row in study['rows']:
            p = row['parameter']
            if kind == 'disk':
                case, folder = f'R/h={p:.12g}', f'disk-{kernel}/R-{p:.12g}'
            elif kind in ('strip', 'box'):
                case, folder = f'width/h={p:.12g}', f'{kind}-{kernel}/width-{p:g}'
            elif kind == 'ellipse':
                case, folder = f'b/a={p:.12g}', f'variable-curvature-ellipse/ratio-{p:g}'
            else:
                case, folder = row.get('case', f'parameter={p:.12g}'), None
            source = row.get('source', None if folder is None else f'specialized/{folder}/data.npz')
            groups.setdefault(kind, []).append(dict(row, case=case, family=kind, kernel=kernel, source=source))
    return groups


def _finite_max(values):
    a = np.asarray([np.nan if x is None else x for x in values], dtype=float)
    return float(np.max(a[np.isfinite(a)])) if np.isfinite(a).any() else None


def method_data(rows, name):
    """Keep all channels, missing cases and nonfinite columns visible."""
    cases = list(dict.fromkeys(r['case'] for r in rows))
    matrix = {k: np.full((len(CHANNELS), len(cases)), np.nan) for k in METRICS}
    outcomes, worst = {}, None
    for c, case in enumerate(cases):
        selected = [(r, m) for r in rows if r['case'] == case for m in r['methods'] if m['name'] == name]
        failures, assessed, complete = False, False, True
        for r, m in selected:
            failures |= m['nonfinite_implemented_values'] > 0
            supported = [CHANNELS.index(ch) for ch in m['supported_channels']]
            if m.get('in_domain_queries', 1) == 0:
                complete = False
            for key, (field, _) in METRICS.items():
                for j, v in enumerate(m[field]):
                    if v is None or not np.isfinite(v):
                        continue
                    assessed = True
                    old = matrix[key][j, c]
                    matrix[key][j, c] = max(v, 0 if np.isnan(old) else old)
                    if key == 'relative' and (worst is None or v > worst['error']):
                        worst = dict(error=v, absolute=m['max_absolute_error'][j],
                                     signal=m['signal_scale'][j], case=case, kernel=r['kernel'], channel=CHANNELS[j])
            complete &= all(m['max_error_over_signal_scale'][j] is not None for j in supported)
        rel = _finite_max(matrix['relative'][:, c])
        outcomes[case] = ('not run' if not selected else 'nonfinite' if failures else
                          'unassessed' if not assessed or not complete else
                          'within' if rel is not None and rel <= EPS32 else 'exceeds')
    return dict(cases=cases, matrix=matrix, outcomes=outcomes, worst=worst)


def signed_budget(arrays, name):
    """Measured signed contributions; do not split an unknown residual by assumption."""
    value, target = arrays[name], arrays['target']
    result = {'total': value-target}
    if name == 'SDF-forced' and 'SDF-contour' in arrays:
        result['geometry'] = arrays['SDF-contour']-target
        result['local + evaluation'] = value-arrays['SDF-contour']
    elif name == 'SDF-contour':
        result['geometry'] = value-target
    elif name == 'SDF-distance-only' and name+'_sampling_error' in arrays:
        for suffix, label in (('numerical', 'numerical'), ('sampling', 'SDF reconstruction'),
                              ('geometry', 'half-plane geometry')):
            result[label] = arrays[name+'_'+suffix+'_error']
    else:
        for suffix, label in (('numerical', 'numerical'), ('model', 'model / geometry')):
            key = f'{name}_{suffix}_error'
            if key in arrays and np.isfinite(arrays[key]).any():
                result[label] = arrays[key]
    return result


def budget_data(rows, name, data, output, cache):
    """Reduce each contribution separately; maxima are not additive budgets."""
    budgets = {}
    for r in rows:
        m = next((m for m in r['methods'] if m['name'] == name), None)
        if m is None:
            continue
        c = data['cases'].index(r['case'])
        scale = np.asarray(m['signal_scale'], dtype=np.longdouble)
        valid_channels = np.array([ch in m['supported_channels'] for ch in CHANNELS])
        if r['source'] is not None:
            path = output/r['source']
            if path.exists() and r['source'] not in cache:
                with np.load(path) as a:
                    cache[r['source']] = {key: a[key] for key in a.files}
            arrays = cache.get(r['source'])
            if arrays is not None and name in arrays:
                pieces = signed_budget(arrays, name)
                domain = arrays.get(name+'_domain', np.ones(len(arrays['target']), dtype=bool))
                mask = domain[:, None] & valid_channels[None]
                for label, values in pieces.items():
                    if label == 'total':
                        continue
                    maximum = np.array([_finite_max(np.abs(values[mask[:, j], j])/scale[j])
                                        if mask[:, j].any() else None for j in range(len(CHANNELS))], dtype=float)
                    a = budgets.setdefault(label, np.full_like(data['matrix']['relative'], np.nan))
                    a[:, c] = np.fmax(a[:, c], maximum)
        for field, label in (('max_refinement_estimate', 'refinement / order difference'),
                             ('max_reference_estimate', 'reference refinement')):
            if field in m:
                values = np.array([np.nan if v is None else v for v in m[field]], dtype=np.longdouble)/scale
                values[~valid_channels] = np.nan
                a = budgets.setdefault(label, np.full_like(data['matrix']['relative'], np.nan))
                a[:, c] = np.fmax(a[:, c], values.astype(float))
    return {k: v for k, v in budgets.items() if np.isfinite(v).any()}


def plot_method(path, study, name, data, metric, start, stop):
    plt = _pyplot()
    cases = data['cases'][start:stop]
    fig, ax = plt.subplots(figsize=(max(6.5, .48*len(cases)+2.5), 5.7), constrained_layout=True)
    im = ax.imshow(np.ma.masked_invalid(data['matrix'][metric][:, start:stop]), aspect='auto', **error_style())
    ax.set_yticks(range(len(CHANNELS)), LABELS, fontsize=9)
    ax.set_xticks(range(len(cases)), cases, rotation=42, ha='right', fontsize=8)
    for j, case in enumerate(cases):
        if data['outcomes'][case] == 'nonfinite':
            ax.text(j, -.85, '×', color='#00a0b5', ha='center', va='center', fontsize=13)
    for y in (2.5, 5.5, 7.5):
        ax.axhline(y, color='white', lw=.6, alpha=.8)
    error_colorbar(fig, im, ax=ax, detailed=True, shrink=.9)
    tag, _ = scheme_info(name, study)
    fig.suptitle(f'{study} | {name} [{tag}]\n{METRICS[metric][1]} · worst tested kernel per cell\n'
                 'Grey: unavailable/not run/outside domain; column ×: implemented nonfinite result.\n'+BAND_LEGEND, fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def plot_budget(path, study, name, data, budgets, start, stop):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 6, figsize=(18, 8.5), constrained_layout=True)
    palette = {'total': '#111111', 'geometry': '#b35806', 'local + evaluation': '#762a83',
               'numerical': '#2166ac', 'model / geometry': '#b35806',
               'SDF reconstruction': '#009e73', 'half-plane geometry': '#b35806',
               'refinement / order difference': '#009e73', 'reference refinement': '#777777'}
    curves = {'total': data['matrix']['relative'], **budgets}
    x = np.arange(stop-start)
    for j, ax in enumerate(axes.flat):
        assessed = False
        for label, values in curves.items():
            a = values[j, start:stop]
            if not np.isfinite(a).any():
                continue
            assessed = True
            style = '--' if 'difference' in label else ':' if label == 'reference refinement' else '-'
            ax.plot(x, np.where(np.isfinite(a), np.maximum(a, EPS64), np.nan), '.',
                    ls=style, lw=1, ms=3, color=palette[label], label=label)
        ax.axhline(EPS32, color='#c5410d', ls='--', lw=.7)
        ax.axhline(EPS64, color='#112b52', ls=':', lw=.7)
        ax.set(title=LABELS[j], yscale='log')
        step = max(1, int(np.ceil(len(x)/10)))
        ticks = np.unique(np.r_[x[::step], x[-1]])
        ax.set_xticks(ticks, start+ticks+1, fontsize=8)
        if not assessed:
            ax.set_facecolor('#d9dde3')
            ax.text(.5, .5, 'not implemented /\nnot assessed', ha='center', va='center', transform=ax.transAxes)
        if j >= 6:
            ax.set_xlabel('geometry case index (catalogue)', fontsize=8)
        ax.grid(alpha=.15)
    handles = {}
    for ax in axes.flat:
        h, labels = ax.get_legend_handles_labels()
        handles.update(zip(labels, h))
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=4, fontsize=9)
    fig.suptitle(f'{study} | {name}: separated error contributions and empirical estimates\n'
                 'Each curve: max |signed contribution| / original channel signal, over queries and kernels.\n'
                 'Discrete case indices, not a continuous parameter. Maxima need not add. Values ≤ε64 share a display floor.\n'
                 'Horizontal guides: ε32 dashed orange, ε64 dotted navy. Refinement differences are not bounds.', fontsize=11)
    fig.savefig(path)
    plt.close(fig)


STUDY_TEXT = {
    'polygon': ('Polygon surfaces and triangle volume moments',
                'Triangles, slivers and notches use the original polygon as target. Small triangles use extended-precision Duffy area quadrature; other fixtures use extended-precision polygon identities with independent polar spot checks. Local SDF assumptions are deliberately stressed, not assumed valid for every triangle.'),
    'disk': ('Disks, cavities and implicit tier transitions',
             'Radius sweeps include origin, tangencies, exact circle-boundary queries, kernel knots and both sides of tier thresholds. Analytic disk polar references are refined at 48/96 nodes; high-degree kernels use factored radial profiles. Cavity results are complemented to the solid convention. Error plots include forced local models; rejected T4 domains remain missing.'),
    'box': ('Thin and small boxes',
            'Box tables are checked on exact edges/corners and table panel joins while thickness decreases. References use polygon identities, with independent rectangular quadrature for high-degree kernels. These are axis-aligned boxes; this does not establish rotated-box or general polygon accuracy.'),
    'strip': ('Thin-strip expansions',
              'Widths and transverse offsets challenge exterior width expansions. A finite box is long enough that its ends are outside these kernel supports, so it matches the infinite strip reference there. Only m00 is implemented. Acceptance d>a is recorded separately from forced extrapolation.'),
    'ellipse': ('Smooth noncircular curvature',
                'True closest-point ellipse SDFs test varying curvature. The integration target is a 1024-edge polygon, with 512/1024 refinement as geometry/reference uncertainty; it is not an independent analytic ellipse oracle. Guarded fallback uses 256 edges. Geometry-split numbers for that fallback are relative to the 1024-edge target, not certified errors against the exact ellipse.'),
}


def _number(value):
    return 'unavailable' if value is None else f'{value:.3g}'


def _image(lines, label, path):
    lines.extend([f'![{label}]({path})', '', f'[Open figure — {label}]({path})', ''])


def _introduction(digest):
    return [
        '# Boundary integration audit', '',
        'Generated from cached measurements. Absolute and signal-relative errors are separate figures; each scheme has its own moment plots and stated expected regime. No boundary evaluator is run to build this report.', '',
        'Code fixes do not update cached measurements. Re-evaluate changed paths before interpreting this dataset as evidence for the current implementation.', '',
        '## Reading the results', '',
        'The intended hierarchy is closed form → accurate quadrature/simplified model → controlled approximation. An exact identity can still have an unstable implementation; an accurately evaluated local model can still represent the wrong geometry.', '',
        f'Channel order is fixed everywhere: `{", ".join(CHANNELS)}`. All fixtures use support h=1.', '',
        '- **Absolute error:** max over query points of |computed − reference|, in the atlas moment scaling.',
        '- **Signal-relative error:** that maximum divided by max over query points of |reference| for the same case, kernel and channel. It is not pointwise relative error at symmetry zeros. Each heatmap cell then takes the worst tested kernel.',
        '- **Colours:** navy ≤ ε64; blue–teal from ε64 to ε32; gold–orange from ε32 to 1; crimson for every error >1. ε32≈1.19e−7 and ε64≈2.22e−16 are visual guides, not certified bounds.',
        '- **Coverage:** grey means unavailable, unsupported, not run or outside a declared domain. Cyan × marks a column with nonfinite implemented results, even when other finite errors are shown. Missing/rejected data never count as an accuracy success.',
        '- **Expectations in brackets** describe the mathematical/representation regime before observing errors. They are not pass/fail badges.', '',
        '## Measured error decomposition', '',
        'For original geometry G, represented contour/polygon Gh, and computed values A, use signed quantities:', '',
        '```text',
        'total error                      = A − I(G)',
        'geometry-induced integral error  = I(Gh) − I(G)',
        'remaining evaluation/model error = A − I(Gh)',
        'total = geometry-induced integral error + remaining evaluation/model error',
        '```', '',
        'For sampled SDF polygons, I(Gh) is the integral of the actual oriented marching-squares contour (`SDF-contour`). This measures the effect of sampling/contour reconstruction separately from forcing a one-surface local curvature/planar model. The forced-SDF residual includes local-model error **and** interpolation/table/evaluation error; it is not labelled pure quadrature error. A zero geometry contribution would not validate the local model near a corner.', '',
        'This is geometry error in the integrated quantities, not a Hausdorff-distance or area-only metric. It depends on the kernel, query and moment. Contour integration itself has finite reference uncertainty, so this is a measured decomposition, not a certified exact split. If the contour vanishes, I(Gh)=0 and the lost signal is explicitly geometry error.', '',
        'The sampled SDF distance/normal/curvature are not necessarily the exact differential geometry of the extracted contour. Their reconstruction errors remain in the residual. A further split of local model error from table/evaluation and differential reconstruction needs independent same-model integrals and geometry jets.', '',
        'Where a same-model reference is cached, numerical = A − I(model) and model/geometry = I(model) − I(G). Curved-boundary polygon surrogates therefore have a separate geometry budget. Guarded mixed paths without a reference for each executed branch retain TOTAL only. Maxima of signed contributions generally do not add, and different curves can attain their maxima at different queries/kernels.', '',
        'Dashed refinement/order differences and dotted reference refinement are empirical estimates. Two resolutions can agree while sharing a bias. The report does not certify few-eps accuracy.', '',
        f'Source: [summary.json](summary.json), SHA-256 `{digest}`.', '',
    ]


def _overview(lines, rows):
    polygon_stats = []
    for name in dict.fromkeys(m['name'] for r in rows for m in r['methods']):
        d = method_data(rows, name)
        counts = {key: list(d['outcomes'].values()).count(key) for key in ('within', 'exceeds', 'nonfinite', 'unassessed', 'not run')}
        polygon_stats.append((name, d, counts))
    universal = [name for name, d, c in polygon_stats if c['within'] == len(d['cases'])]
    lines += ['## State of the polygon schemes', '',
              ('No tested polygon scheme remains at ε32 or better over every fixture on all of its implemented channels.' if not universal else
               'Schemes within ε32 on their implemented channels for every polygon fixture: '+', '.join(universal)+'.'), '',
              'Counts below require every tested kernel and implemented channel to meet the threshold. They do not imply support for missing moments, untested kernels or rejected queries.', '',
              '| Scheme and expectation | Within ε32 | Exceeds ε32 | Nonfinite/input collapse | Unassessed | Not run/inapplicable |',
              '|---|---:|---:|---:|---:|---:|']
    for name, d, c in polygon_stats:
        lines.append(f"| {name} [{scheme_info(name)[0]}] | {c['within']} | {c['exceeds']} | {c['nonfinite']} | {c['unassessed']} | {c['not run']} |")
    lines += ['', '### Absolute error across polygon schemes', '',
              'Small absolute error can hide total loss of a tiny element. This view is useful for unit-scale defects and the overall numerical floor.', '']
    _image(lines, 'Polygon absolute-error overview', 'overview-absolute.png')
    lines += ['### Signal-relative error across polygon schemes', '',
              'This view reveals loss of small moments. A red tiny-element cell can coexist with a very small absolute error; that is a relative-accuracy failure, not necessarily a unit-scale jump.', '']
    _image(lines, 'Polygon signal-relative overview', 'overview-relative.png')


def _spatial_evidence(lines, output):
    guard = output.parent/'sdf-guard-cpu'/'report.md'
    if guard.exists() and guard.resolve() != (output/'report.md').resolve():
        relative = Path(os.path.relpath(guard, output)).as_posix()
        lines += ['## SDF support-guard checkpoint', '',
                  f'The [fresh sampled-cell guard checkpoint]({relative}) validates rejection and executed polygon fallback. '
                  'It replaces the old discrete-probe policy; preserved baseline measurements below are not recomputed by generating this report. '
                  'The new guard bounds blended jets throughout the support square and rejects curvature poles, unresolved gradients and incomplete sampling coverage. '
                  'It does not certify the integral truncation error of accepted smooth models.', '']
    mechanism = output.parent/'sdf-corner-mechanism'/'report.md'
    if mechanism.exists():
        relative = Path(os.path.relpath(mechanism, output)).as_posix()
        lines += ['## Corner distance versus curvature', '',
                  'The forced SDF mass uses a curvature series, not the distance-only half-plane model. '
                  'An exact exterior-corner SDF is valid, but recovering closest-surface curvature has a zero denominator. '
                  'The old [0,1] value scale hid large raw predictions. '
                  f'The [focused mechanism report]({relative}) shows the continuous planar ablation, angular geometry bias and separate sampling/evaluation contributions.', '']
    folders = sorted((output/'maps').glob('*/moments.png'))
    if folders:
        lines += ['## Spatial inspections', '',
                  'These are cached baseline snapshots; their guarded rows can predate the sampled-cell guard. '
                  'Use the fresh support-guard checkpoint for current production behavior.', '',
                  'These cached maps locate errors around individual boundary elements. Each figure preserves the atlas moment order. '
                  'The forced local-model row is diagnostic; the guarded and fused rows show executed production paths. '
                  'Their absolute-error maps complement the per-case relative-error survey.', '']
        for path in folders:
            lines += [f'### {path.parent.name}', '']
            _image(lines, path.parent.name+' spatial scalar moments', path.relative_to(output).as_posix())
            other = path.parent/'gradient-moments.png'
            if other.exists():
                _image(lines, path.parent.name+' spatial gradient moments', other.relative_to(output).as_posix())
    _stress_evidence(lines, output)


def _stress_evidence(lines, output):
    """Lead with executed guarded fields; retain forced failures as links only."""
    stress = output.parent/'boundary-atlas'/'sdf-stress'
    source = stress/'summary.json'
    if source.exists():
        summary = json.loads(source.read_text())
        lines += ['## Local-model stress geometries', '',
                  'Current executed guarded results are shown where a matching checkpoint is available. '
                  'Forced models bypass the guard deliberately; their failures are retained as labelled diagnostic links, '
                  'not presented as production predictions. These are total errors, not a new geometry/model split.', '']
        for case in summary['cases']:
            lines += [f"### {case['name']}", '', case['assumption']+'.', '',
                      'Reference: '+case['reference']+'.', '']
            current = None
            if case['name'] == 'sharp-corner':
                for device in ('cuda', 'cpu'):
                    checkpoint = output.parent/f'sdf-guard-{device}'
                    if (checkpoint/'corner.npz').exists() and (checkpoint/'checkpoint.json').exists():
                        metadata = json.loads((checkpoint/'checkpoint.json').read_text())
                        if metadata.get('policy') == 'sampled-cell-interval-v1':
                            from .sdf_stress import publish_guarded_corner
                            current = publish_guarded_corner(stress/case['name'], checkpoint)
                            break
            if current is not None:
                lines += ['**Current production result: guarded Scene with original-polygon fallback.** '
                          'The following maps show the executed values and their error everywhere, including rejected supports evaluated by the fallback. '
                          'They do not mask the forced model with an acceptance flag.', '',
                          f"Policy: `{current['policy']}`; device: `{current['device']}`. "
                          f"[Raw executed values]({Path(os.path.relpath(checkpoint/'corner.npz', output)).as_posix()}) · "
                          f"[Source provenance]({Path(os.path.relpath(stress/case['name']/'guarded-provenance.json', output)).as_posix()}).", '']
                for file in ('guarded-v1-moments.png', 'guarded-v1-gradient-moments.png'):
                    path = stress/case['name']/file
                    _image(lines, 'sharp-corner CURRENT guarded '+file, Path(os.path.relpath(path, output)).as_posix())
            else:
                lines += ['No current executed guarded spatial checkpoint is attached for this stress fixture. '
                          'The links below are forced-model counterexamples, not the current guarded solver output.', '']
            distance_study = output.parent/'sdf-distance-only'
            distance_folder = distance_study/'stress'/case['name']
            if distance_study.resolve() != output.resolve() and (distance_folder/'moments.png').exists():
                lines += ['**Distance-only semi-analytic reference — no curvature.** '
                          'This is a separate local half-plane model, not the guarded solver: '
                          'mass uses signed distance only; oriented moments also use the sampled unit normal. '
                          'It avoids the curvature pole but retains corner/thin-feature geometry bias and normal ambiguities. '
                          f"[Full reference study and separated budgets]({Path(os.path.relpath(distance_study/'report.md', output)).as_posix()}).", '']
                for file in ('moments.png', 'gradient-moments.png', 'error-budget.png'):
                    path = distance_folder/file
                    if path.exists():
                        _image(lines, case['name']+' distance-only reference '+file, Path(os.path.relpath(path, output)).as_posix())
            old_mask = summary.get('production_policy') != 'sampled-cell-interval-v1'
            label = ('Historical forced diagnostic — guard bypassed; pre-guard policy mask' if old_mask else
                     'Forced diagnostic — guard bypassed')
            lines += ['**Forced-model diagnostic archive.** '+
                      ('The archived acceptance row uses the old probe policy and must not be read as current acceptance. ' if old_mask else '')+
                      'Large corner values are expected when the curvature guard is bypassed.', '']
            for file in ('moments.png', 'gradient-moments.png'):
                path = stress/case['name']/file
                if path.exists():
                    lines += [f'[{label}: {file}]({Path(os.path.relpath(path, output)).as_posix()})', '']


def refresh_stress_sections(root=evidence_path()):
    """Refresh report views only, preserving measurement caches and other sections."""
    from .report_view import accessible_report
    updated = []
    for path in sorted(Path(root).glob('*/report.md')):
        text = path.read_text()
        marker = '## Local-model stress geometries\n'
        if marker not in text:
            continue
        lines = []
        _stress_evidence(lines, path.parent)
        if not lines:
            continue
        text = re.sub(r'(?ms)^## Local-model stress geometries\n.*?(?=^## |\Z)', lambda _: '\n'.join(lines)+'\n', text)
        baseline_note = ('These are cached baseline snapshots; their guarded rows can predate the sampled-cell guard. '
                         'Use the fresh support-guard checkpoint for current production behavior.')
        if '## Spatial inspections\n' in text and baseline_note not in text:
            text = text.replace('## Spatial inspections\n', '## Spatial inspections\n\n'+baseline_note+'\n', 1)
        path.write_text(text)
        accessible_report(path)
        updated.append(path)
    return updated


def _method_section(lines, kind, rows, name, output, cache, manifest, max_cases):
    data = method_data(rows, name)
    budgets = budget_data(rows, name, data, output, cache)
    tag, explanation = scheme_info(name, kind)
    methods = [m for r in rows for m in r['methods'] if m['name'] == name]
    supported = [ch for ch in CHANNELS if any(ch in m['supported_channels'] for m in methods)]
    w = data['worst']
    lines += [f'### {name} [{tag}]', '', explanation, '',
              'Implemented channel union: '+', '.join(f'`{ch}`' for ch in supported)+'.', '']
    if any(m.get('accepted_queries') is not None for m in methods):
        accepted = sum(m.get('accepted_queries', 0) for m in methods)
        false_accept = sum(m.get('accepted_m00_errors_over_0_01', 0) for m in methods)
        lines += [f'Policy flags: {accepted} accepted query records, {false_accept} with measured m00 absolute error >0.01. '
                  'Counts include separate kernels/parameters. Plots include forced predictions where stored; acceptance alone does not certify accuracy.', '']
    if any(m.get('status') == 'outside_domain' or m.get('in_domain_queries', 1) == 0 for m in methods):
        lines += ['Some parameter/kernel rows have no evaluated in-domain queries. Those grey cells are unassessed, not accurate zeros.', '']
    if w:
        lines += [f"Worst finite signal-relative discrepancy: **{w['error']:.3g}**, `{w['case']}`, `{w['kernel']}`, `{w['channel']}` "
                  f"(absolute error {_number(w['absolute'])}; reference signal {_number(w['signal'])}).", '']
    if name == 'SDF-forced':
        lines += ['Budget curves separate **sampled-contour geometry error** from **local-model + evaluation residual**. '
                  'Both use the same original-geometry signal scale. This distinguishes the contour-geometry effect from the remaining local description and evaluation errors.', '',
                  '| Geometry case | Max contour geometry error / signal | Max local-model + evaluation residual / signal |',
                  '|---|---:|---:|']
        for j, case in enumerate(data['cases']):
            geometry = budgets.get('geometry')
            residual = budgets.get('local + evaluation')
            lines.append(f"| {case} | {_number(None if geometry is None else _finite_max(geometry[:, j]))} | "
                         f"{_number(None if residual is None else _finite_max(residual[:, j]))} |")
        lines += ['', 'Each table entry is maximized separately over implemented moments, queries and kernels. These maxima need not occur at the same point or add to the total maximum.', '']
    elif not any(k in budgets for k in ('numerical', 'model / geometry', 'geometry')):
        lines += ['A complete numerical/model split is unavailable for this executed path. Only total error and any empirical estimates are shown.', '']
    if name == 'SDF-contour':
        lines += ['These errors are the measured geometry contribution used in the forced-SDF decomposition. The contour evaluator is a reference integration, not an independent proof of zero numerical error.', '']
    prefix = _slug(kind)+'-'+_slug(name)
    for start in range(0, len(data['cases']), max_cases):
        stop = min(start+max_cases, len(data['cases']))
        page = 1+start//max_cases
        for metric in METRICS:
            path = f'report-plots/{prefix}-{metric}-{page}.png'
            plot_method(output/path, kind, name, data, metric, start, stop)
            manifest['plots'].append(path)
            lines += [f'**{METRICS[metric][1]} — cases {start+1}–{stop}.**', '']
            _image(lines, f'{kind} {name} {metric} error page {page}', path)
        if budgets:
            path = f'report-plots/{prefix}-budget-{page}.png'
            plot_budget(output/path, kind, name, data, budgets, start, stop)
            manifest['plots'].append(path)
            lines += ['**Contributions and empirical estimates.** Missing curves indicate unavailable separation; zero curves share the ε64 display floor.', '']
            _image(lines, f'{kind} {name} separated error budget page {page}', path)
    manifest['schemes'].append(dict(study=kind, name=name, expectation=tag, cases=data['cases'],
                                    outcomes=data['outcomes'], worst=w,
                                    budgets={k: [[None if not np.isfinite(x) else float(x) for x in row] for row in a] for k, a in budgets.items()}))


def _limits(lines, summary, output):
    lines += ['## Representation and reference limits', '',
              'Input rounding is separate from evaluator error: after coordinates collapse, a more accurate integral cannot recover the original shape.', '',
              '| Coordinate dtype | Input fixtures | Collapsed | Max finite m00 geometry error / original signal |',
              '|---|---:|---:|---:|']
    for dtype in dict.fromkeys(r['dtype'] for r in summary.get('representation', [])):
        rows = [r for r in summary['representation'] if r['dtype'] == dtype]
        relative = [r['input_geometry_m00_error']/abs(r['local_reference_m00']) for r in rows
                    if r['input_geometry_m00_error'] is not None and r['local_reference_m00'] != 0]
        lines.append(f"| {dtype} | {len(rows)} | {sum(r['collapsed'] for r in rows)} | {_number(max(relative) if relative else None)} |")
    lines += ['', 'Polygon references share identities with some evaluators, so their agreement alone is not independent validation. '
              'The finite high-precision polar spot checks are linked below; their uncertainty cannot be generalized to every query. '
              'Refined polygon ellipse references do not establish analytic-ellipse or few-eps accuracy.', '']
    coverage_path = output/'coverage.json'
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text())
        for limit in coverage.get('explicit_limits', []):
            lines.append('- '+limit)
        if coverage.get('reference_signal_precision_limit'):
            lines.append('- '+coverage['reference_signal_precision_limit'])
        lines.append('')
    evidence = ', '.join(f'[{label}]({file})' for label, file in
                         [('Coverage', 'coverage.json'), ('failure reproducers', 'findings.json'),
                          ('input representation', 'representation.json'),
                          ('independent oracle spot checks', 'independent-oracle-checks.json')]
                         if (output/file).exists())
    lines += ['## Evidence and reproduction', '',
              '- [Full measurements](summary.json) and [report classifications and separated budgets](report-data.json).',
              '- '+evidence+'.' if evidence else 'No supplementary evidence files in this checkpoint.',
              '- Raw signed contributions remain in each per-case `data.npz`. In particular, `SDF-contour_model_error` is the contour-geometry contribution; `SDF-forced − SDF-contour` is the remaining local/evaluation residual.', '',
              '```bash',
              '# Evaluate new measurements and build this report',
              'python -m warpSPHBoundaries.validation.robustness --device cuda:0',
              '# Rebuild only the report and its figures from cached numerical data',
              'python -m warpSPHBoundaries.validation.audit_report --dpi 300 --max-cases 24',
              '```', '',
              'Add polygon fixtures in `adversarial.polygon_cases`, or a specialized study with its cached source path, '
              'then rerun the measurement command. The report discovers new cases/methods and paginates automatically. '
              'Add the expected regime to `scheme_info`; new schemes without one are explicitly marked unspecified. '
              'This survey prioritizes 2D. It does not establish 3D surface/tetrahedral coverage, general curved-tube solvers, '
              'or a universally accurate production dispatcher.', '']


def generate(output=evidence_path('robustness'), dpi=None, max_cases=24):
    output = Path(output)
    if max_cases < 1:
        raise ValueError('max_cases must be positive')
    source = output/'summary.json'
    summary = json.loads(source.read_text())
    if tuple(summary['channels']) != tuple(CHANNELS):
        raise ValueError('Unexpected moment order in cached summary')
    dpi = summary.get('dpi', 300) if dpi is None else dpi
    _pyplot().rcParams['savefig.dpi'] = dpi
    from .robustness import plot_overview
    plot_overview(output, summary['records'])
    groups = collections(summary)
    (output/'report-plots').mkdir(exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = dict(summary_sha256=digest, channels=list(CHANNELS), dpi=dpi,
                    max_cases_per_plot=max_cases, schemes=[], plots=[])
    lines = _introduction(digest)
    _overview(lines, groups['polygon'])
    lines += ['## Survey sections', '']
    for kind in groups:
        title = STUDY_TEXT.get(kind, (kind, 'Additional cached study.'))[0]
        lines.append(f'- [{title}](#{_slug(kind)}-survey)')
    lines += ['', 'The following sections expand automatically with cached cases and methods. Figures paginate after '
              f'{max_cases} geometry cases, preserving channel order and the shared precision scale.', '']
    cache = {}
    for kind, rows in groups.items():
        if not rows:
            continue
        title, text = STUDY_TEXT.get(kind, (kind, 'Additional cached study; consult its numerical source.'))
        text = summary.get('reference_descriptions', {}).get(kind, text)
        lines += [f'<a id="{_slug(kind)}-survey"></a>', '', f'## {title}', '', text, '',
                  'Tested kernels: '+', '.join(dict.fromkeys(r['kernel'] for r in rows))+'.', '']
        lines += ['Geometry case indices for the contribution graphs (discrete fixtures, not a continuous x axis):', '',
                  '| Index | Geometry case | Family | Parameter |',
                  '|---:|---|---|---:|']
        catalogue = {}
        for r in rows:
            catalogue.setdefault(r['case'], r)
        for i, (case, r) in enumerate(catalogue.items(), 1):
            lines.append(f"| {i} | {case} | {r['family']} | {r['parameter']:.12g} |")
        lines.append('')
        for name in dict.fromkeys(m['name'] for r in rows for m in r['methods']):
            print(f'Report: {kind} / {name}', flush=True)
            _method_section(lines, kind, rows, name, output, cache, manifest, max_cases)
        cache.clear()
    _spatial_evidence(lines, output)
    _limits(lines, summary, output)
    (output/'report-data.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    path = output/'report.md'
    path.write_text('\n'.join(lines))
    from .report_view import accessible_report
    accessible_report(path)
    print(f'Wrote {path} ({len(manifest["schemes"])} scheme studies, {len(manifest["plots"])} separate plots)', flush=True)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=evidence_path('robustness'))
    p.add_argument('--dpi', type=int)
    p.add_argument('--max-cases', type=int, default=24)
    p.add_argument('--refresh-stress-only', action='store_true', help='Refresh stress sections across reports under --output; keep other sections and numerical caches')
    a = p.parse_args()
    if a.refresh_stress_only:
        for path in refresh_stress_sections(a.output):
            print(f'Refreshed {path}')
    else:
        generate(a.output, a.dpi, a.max_cases)


if __name__ == '__main__':
    main()
