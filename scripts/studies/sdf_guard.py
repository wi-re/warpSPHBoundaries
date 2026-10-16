"""Executed SDF domain-guard checkpoint against the preserved polygon fixtures."""
import argparse
import json
from pathlib import Path

from warpSPHBoundaries.validation.paths import evidence_path

import numpy as np
import torch

from warpSPHBoundaries.scene.scene import SdfRep, SurfaceRep, _sdfTier3
from warpSPHBoundaries.validation.adversarial import (
    Measurement, SUPPORTED, make_sdf, polygon_cases, scene_channels)
from warpSPHBoundaries.validation.atlas import CHANNELS, _pyplot
from warpSPHBoundaries.validation.audit_report import generate, method_data
from warpSPHBoundaries.validation.corner_audit import CORNER, production_channels
from warpSPHBoundaries.validation.error_colors import error_style, error_colorbar
from warpSPHBoundaries.validation.report_view import accessible_report
from warpSPHBoundaries.validation.robustness import save_measurements
from warpSPHBoundaries.validation.sdf_stress import CASES


def spatial(output, device, grid_size):
    """Actual Scene/fused results around the corner; never substitute a reference."""
    grid = np.linspace(-1.2, 1.2, grid_size)
    x, y = np.meshgrid(grid, grid)
    points = np.column_stack((x.ravel(), y.ravel()))
    target = CORNER.reference(points, 'w4')
    guarded, fused = production_channels(points, device=device)
    rep = SdfRep.fromFunction(lambda p: torch.tensor(CORNER.distance(p.numpy())),
                             (-2.5, -2.5), (2.5, 2.5), 1/64).to(device)
    p = torch.tensor(points, dtype=torch.float64, device=device)
    fallback = _sdfTier3(rep, p, torch.ones(len(p), dtype=torch.float64, device=device), 'w4', device)[2].cpu().numpy()
    np.savez_compressed(output/'corner.npz', grid=grid, points=points, target=target,
                        guarded=guarded, fused=fused, fallback=fallback, channels=CHANNELS)
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    fields = [target[:, 0], guarded[:, 0], np.abs(guarded[:, 0]-target[:, 0])]
    titles = ['Original corner: reference mass', 'Executed guarded Scene: mass', 'Executed guarded Scene: absolute error']
    for i, (ax, values) in enumerate(zip(axes, fields)):
        if i < 2:
            im = ax.pcolormesh(grid, grid, values.reshape(x.shape), shading='nearest', cmap='viridis', vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, shrink=.7)
        else:
            im = ax.pcolormesh(grid, grid, values.reshape(x.shape), shading='nearest', **error_style())
            error_colorbar(fig, im, ax=ax, shrink=.7, detailed=True)
        ax.plot([-1.2, 0, 0], [0, 0, -1.2], color='#333333', lw=1)
        ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect='equal', xlabel='$x/h$', ylabel='$y/h$', title=titles[i])
    fig.suptitle('Corner pole guarded before series evaluation; original polygon used on rejected supports\n'
                 f'{device}, Wendland C4, spacing h/64. Error colours: navy ≤ ε64; blue–teal ≤ ε32; gold–orange ≤1; crimson >1.', fontsize=10)
    fig.savefig(output/'corner.png', dpi=300)
    plt.close(fig)
    # Same fixed moment ordering as the other tier reports, including visible gaps.
    fig, axes = plt.subplots(2, 6, figsize=(18, 6), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        values = np.abs(guarded[:, j]-target[:, j]).reshape(x.shape)
        ax.set_facecolor('#e9e9e9')
        im = ax.pcolormesh(grid, grid, np.ma.masked_invalid(values), shading='nearest', **error_style())
        if j in (3, 4, 5):
            ax.text(.5, .5, 'not requested\nin this checkpoint', transform=ax.transAxes, ha='center', va='center')
        ax.plot([-1.2, 0, 0], [0, 0, -1.2], color='#333333', lw=.6)
        ax.set(title=CHANNELS[j], xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect='equal')
    error_colorbar(fig, im, ax=list(axes.flat), shrink=.7, detailed=True)
    fig.suptitle('Executed guarded Scene: absolute errors in the shared twelve-channel order; grey = not recorded in this checkpoint')
    fig.savefig(output/'corner-moments.png', dpi=300)
    plt.close(fig)
    return dict(grid=grid_size, guarded_m00_max_error=float(np.nanmax(fields[2])),
                guarded_max_error=float(np.nanmax(np.abs(guarded-target))),
                fused_max_error=float(np.nanmax(np.abs(fused-target))), fallback_queries=int(fallback.sum()))


def run(baseline, output, device, grid_size):
    baseline, output = Path(baseline), Path(output)
    if baseline.resolve() == output.resolve():
        raise ValueError('preserve baseline')
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((baseline/'summary.json').read_text())
    cases = {c.name: c for c in polygon_cases()}
    records, branches = [], []
    for row in source['records']:
        with np.load(baseline/row['kernel']/row['case']/'data.npz') as a:
            points, target, uncertainty = a['points'], a['target'], a['reference_estimate']
        case = cases[row['case']]
        rep = make_sdf(case, points, 1/64).to(device)
        p = torch.tensor(points, dtype=torch.float64, device=device)
        _, _, fallback = _sdfTier3(rep, p, torch.ones(len(p), dtype=torch.float64, device=device), row['kernel'], device)
        values, estimate, status = scene_channels(rep, points, row['kernel'], device, return_diagnostics=True)
        local = ~fallback.cpu().numpy()
        branch = dict(case=row['case'], kernel=row['kernel'], queries=len(points),
                      fallback=int(fallback.sum().item()), nonfallback=int(local.sum()))
        branches.append(branch)
        folder = output/row['kernel']/row['case']
        folder.mkdir(parents=True, exist_ok=True)
        methods = save_measurements(folder, target, uncertainty, [
            Measurement('SDF-cell-guarded', values,
                        target if bool(fallback.all().item()) else None,
                        estimate=estimate, status=status, supported=SUPPORTED,
                        note='Executed Scene with all-cell support guard and original polygon fallback; nine SDF-contract channels. Fully fallback rows have an independent same-geometry reference; device refinement estimates are empirical.')],
            points=points, vertices=case.vertices, fallback=fallback.cpu().numpy())
        records.append(dict(row, methods=methods))
        print(device, row['kernel'], row['case'], branch['fallback'], '/', len(points), 'fallback', flush=True)
    summary = dict(source, records=records, device=device, checkpoint='SDF sampled-cell interval guard',
                   specialized=[], representation=[])
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    # Execute exactly the two previously false-accepted supports.
    features = []
    for name, point in [('thin-strip', [0., .2]), ('narrow-gap', [.05, .3])]:
        case = next(c for c in CASES if c.name == name)
        rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy())),
                                 (-2.5, -2.5), (2.5, 2.5), 1/64).to(device)
        p = torch.tensor([point], dtype=torch.float64, device=device)
        rejected = _sdfTier3(rep, p, torch.ones(1, dtype=torch.float64, device=device), 'w4', device)[2]
        features.append(dict(case=name, point=point, fallback=bool(rejected.item())))
    corner = spatial(output, device, grid_size)
    data = method_data(records, 'SDF-cell-guarded')
    counts = {k: list(data['outcomes'].values()).count(k) for k in ('within', 'exceeds', 'nonfinite', 'unassessed')}
    checkpoint = dict(device=device, counts=counts, worst=data['worst'], branches=branches,
                      corner=corner, prior_false_accepts=features, policy='sampled-cell-interval-v1')
    (output/'checkpoint.json').write_text(json.dumps(checkpoint, indent=2, allow_nan=False)+'\n')
    (output/'coverage.json').write_text(json.dumps(dict(explicit_limits=[
        '21 preserved polygon fixtures, four kernels, nine implemented SDF-contract channels, original high-precision references.',
        'Fresh actual Scene and supplied original-geometry fallback; scalar second moments remain unexposed in the SDF contract.',
        'Cell intervals bound blended distance/gradient/Laplacian throughout a square containing the support ball.',
        'This conservatively rejects some usable supports. It is a domain guard, not a certified integral-error bound.',
        'Subgrid geometry absent from input samples cannot be detected. Prefer the original polygon as fallback.',
        'Smooth curved SDFs retain sampling, truncation and planar higher-moment errors; no universal epsilon32 claim.',
        'Corner spatial maps use an extended-precision polygon closed form; errors are measured, not certified bounds.']), indent=2)+'\n')
    path = generate(output, 300)
    text = path.read_text()
    title, rest = text.split('\n', 1)
    intro = f'''## SDF domain-guard checkpoint

Device: `{device}`. A new conservative guard checks every blended-jet cell
covered by the support bounding square. It rejects nonfinite jets, gradient
norm intervals outside [0.9,1.1], curvature denominator below 0.25,
|kappa h| above 0.5, and incomplete sampled-grid/stencil coverage. Rejected
curvatures do not enter the table series; a finite mass outside [0,1] (up to
roundoff slack) or nonfinite evaluated output also requests fallback.
Mass is not clamped to hide model failure. Fully empty/solid supports bypass
differential geometry only with a conservative distance-interpolation margin.

The old finite probe set could miss a corner or medial axis. The new bounds
cover each entire cell; a cached square-minimum table makes per-query region
checks independent of the number of covered cells. Fused SDF evaluation
already lowers to the supplied polygon/contour and does not use this series.

| Scoped polygon result (all tested kernels and implemented moments) | Count |
|---|---:|
| Within float32 epsilon of the original signal scale | {counts['within']}/21 |
| Exceeds | {counts['exceeds']} |
| Nonfinite | {counts['nonfinite']} |

These are fresh executed-path results, not substituted reference values. The
branch counts in [checkpoint.json](checkpoint.json) distinguish local/whole
support results from fallback; fallback accuracy is not a validation of the
local curvature approximation. Of {sum(b['queries'] for b in branches)} polygon
query/kernel records, {sum(b['fallback'] for b in branches)} used fallback.
These targeted sharp/tiny/thin polygon fixtures therefore assess rejection
and fallback accuracy, not curved-SDF acceleration. Tests separately retain usable planes and a
smooth sampled disk (R=3h), and reject the former thin-strip/narrow-gap misses.

On fully fallback rows, the original geometry is preserved: the numerical
budget compares the executed integrator to the high-precision same-geometry
reference, and the geometry/model contribution is zero. Adaptive device-area
refinement estimates are retained where available. These are empirical
estimates, not certified bounds; analytic edge rows have no such estimate.

![Executed corner mass and error](corner.png)

![Corner errors in the common twelve-channel order](corner-moments.png)

This is a **sampled-field domain guard**, not a universal model-error
certificate. It bounds the jets actually used by the evaluator. It cannot
recover geometry absent from the samples, and does not bound omitted
curvature terms or the planar higher-moment approximation on accepted smooth
surfaces. For accuracy requirements stricter than the validated local-model
domain, use the original polygon integrator. The support square is deliberately
conservative; performance and false rejections have not been optimized here.
Reference and fallback errors remain finite; the general report below retains
unsupported channels and reference limitations.

Reproduce: `python scripts/studies/sdf_guard.py --device {device} --output {output}`.

'''
    path.write_text(title+'\n\n'+intro+rest)
    accessible_report(path)
    print(json.dumps(checkpoint, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default=evidence_path('triangle-conditioning'))
    p.add_argument('--output', default=evidence_path('sdf-guard-cpu'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--grid', type=int, default=129)
    a = p.parse_args()
    run(a.baseline, a.output, a.device, a.grid)
