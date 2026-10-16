"""Shareable circle comparison: mass and first spatial kernel moment.

Run from the checkout: python scripts/studies/curved_boundary_example.py
The default evidence destination is outside the repository. No cached audit is
needed. --report-only redraws the saved measurements without executing solvers.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from warpSPHBoundaries.scene.scene import SdfRep, SurfaceRep, _sdfCurvature
from warpSPHBoundaries.validation.adversarial import scene_channels
from warpSPHBoundaries.validation.atlas import _pyplot
from warpSPHBoundaries.validation.distance_only import halfplane_channels, sample_distance_normal
from warpSPHBoundaries.validation.error_colors import EPS32
from warpSPHBoundaries.validation.paths import evidence_path
from warpSPHBoundaries.validation.sdf_accuracy import SmoothCase, area_reference


RADIUS = 3.
CHANNELS = [0, 2]  # m00 and m01; the outward normal is (0,1), h=1.
METHODS = ('distance_only', 'corrected', 'polygon32', 'polygon256', 'polygon1024')
LABELS = ('SDF: distance only / local plane', 'SDF: curvature correction (order 2)',
          'Polygon: 32 edges', 'Polygon: 256 edges', 'Polygon: 1,024 edges')
COLORS = ('#d16b19', '#1764ab', '#9954a5', '#228557', '#b84752')
STYLES = ('--', '-', '-.', ':', (0, (4, 2)))


def measure(output):
    output.mkdir(parents=True, exist_ok=True)
    distances = np.linspace(-.95, .95, 81)
    case = SmoothCase('presentation-circle', a=RADIUS, b=RADIUS)
    points = case.points(distances)
    ref, uncertainty = [], []
    for i, point in enumerate(points):
        fine, ef = area_reference(case, point, 'w4', 96, 2e-12)
        coarse, ec = area_reference(case, point, 'w4', 48, 1e-11)
        ref.append(fine[CHANNELS])
        uncertainty.append(np.abs(fine-coarse)[CHANNELS]+ef+ec)
        if i % 20 == 0:
            print(f'curved reference {i+1}/{len(points)}', flush=True)
    target, uncertainty = np.asarray(ref), np.asarray(uncertainty)
    rep = SdfRep.fromFunction(lambda p: torch.tensor(case.distance(p.numpy())),
                             points.min(0)-1.35, points.max(0)+1.35, 1/64)
    pos = torch.tensor(points, dtype=torch.float64)
    _, rejected = _sdfCurvature(rep, pos, torch.ones(len(pos), dtype=torch.float64), 'w4', 'cpu')
    assert not rejected.any(), 'This illustration must measure the accepted SDF branch.'
    sampled = sample_distance_normal(rep, points)
    values = dict(distance_only=halfplane_channels(sampled['distance'], sampled['normal'], 'w4')[:, CHANNELS],
                  corrected=scene_channels(rep, points, 'w4', 'cpu')[:, CHANNELS])
    diagnostics = {}
    for count in (32, 256, 1024):
        # Inscribed polygon, with the top centred on an edge rather than a
        # vertex. The polygon is a distinct approximation to the same circle.
        theta = np.pi/2+(np.arange(count)+.5)*2*np.pi/count
        vertices = RADIUS*np.column_stack((np.cos(theta), np.sin(theta)))
        polygon = SurfaceRep.polygon(vertices)
        actual, estimates, status = scene_channels(polygon, points, 'w4', 'cpu', return_diagnostics=True)
        assert (status >= 0).all(), 'Polygon quadrature failed.'
        values[f'polygon{count}'] = actual[:, CHANNELS]
        diagnostics[f'polygon{count}_vertices'] = vertices
        diagnostics[f'polygon{count}_estimate'] = estimates[:, CHANNELS]
        diagnostics[f'polygon{count}_status'] = status
    assert all(np.isfinite(value).all() for value in values.values())
    np.savez_compressed(output/'data.npz', distances=distances, points=points,
                        target=target, reference_uncertainty=uncertainty,
                        accepted=~rejected.numpy(), **values, **diagnostics)
    print('measured circle, distance-only SDF, corrected Scene SDF and three Scene polygons', flush=True)


def report(output):
    with np.load(output/'data.npz') as archive:
        data = dict(archive)
    plt = _pyplot()
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11})
    fig = plt.figure(figsize=(18, 9.4), facecolor='white')
    grid = fig.add_gridspec(2, 3, width_ratios=(1.05, 1.25, 1.25),
                           left=.055, right=.98, bottom=.16, top=.79, hspace=.35, wspace=.29)
    geometry = fig.add_subplot(grid[:, 0])
    mass, first = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])
    mass_error, first_error = fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[1, 2])
    x = np.linspace(-1.45, 1.45, 1001)
    wall = np.sqrt(RADIUS**2-x*x)-RADIUS
    geometry.fill_between(x, -1.6, wall, color='#e7ebef')
    geometry.plot(x, wall, color='#151b24', lw=2.5, label='True circular wall')
    for count, color in ((32, COLORS[2]), (256, COLORS[3])):
        vertices = data[f'polygon{count}_vertices']
        vertices = np.vstack((vertices, vertices[0]))-np.array([0., RADIUS])
        geometry.plot(vertices[:, 0], vertices[:, 1], color=color, lw=1.7,
                      ls='-.' if count == 32 else ':')
    d = .25
    inside = np.abs(x) <= 1
    extent = np.sqrt(np.maximum(0, 1-x[inside]**2))
    lo, hi = d-extent, np.minimum(d+extent, wall[inside])
    geometry.fill_between(x[inside], lo, hi, where=hi >= lo, color='#8ebbdc', alpha=.55)
    geometry.add_patch(Circle((0, d), 1, fill=False, color='#505963', lw=1.3))
    geometry.plot([-1.4, 1.4], [0, 0], '--', color=COLORS[0], lw=1.4)
    geometry.plot([0, 0], [-.95, .95], color='#505963', lw=1, ls=':')
    geometry.scatter([0], [d], s=45, color='#151b24', zorder=6)
    geometry.annotate('query $x$', (0, d), xytext=(.17, .44), fontsize=11)
    geometry.annotate('', (0, 1.02), (0, .65), arrowprops=dict(arrowstyle='->', color='#151b24', lw=1.6))
    geometry.text(.09, .88, r'$\mathbf{n}$', fontsize=14)
    geometry.text(-1.25, -1.14, 'SOLID', color='#6a747f', fontsize=13, weight='bold')
    geometry.text(-1.25, 1.02, 'FLUID', color='#6a747f', fontsize=13, weight='bold')
    geometry.text(-.94, -.48, 'integrated region', color='#244f70', fontsize=10)
    geometry.set(xlim=(-1.45, 1.45), ylim=(-1.3, 1.3), aspect='equal',
                 xlabel='Tangential coordinate / $h$', ylabel='Normal coordinate / $h$',
                 title='A. Curved wall and kernel support')
    geometry.text(.5, -.18, 'Circle: $R=3h$  |  support radius: $h$\nIllustrated query: $d/h=0.25$\nOrange dashed line: local-plane model',
                  transform=geometry.transAxes, ha='center', va='top', fontsize=10, linespacing=1.6)
    geometry.spines[['top', 'right']].set_visible(False)
    ds = data['distances']
    target = np.asarray(data['target'], dtype=float)
    scales = np.max(np.abs(target), axis=0)
    metrics = {}
    handles = [mass.plot(ds, target[:, 0], color='#151b24', lw=2.6, label='Curved-domain reference')[0]]
    first.plot(ds, target[:, 1], color='#151b24', lw=2.6)
    for key, label, color, style in zip(METHODS, LABELS, COLORS, STYLES):
        actual = np.asarray(data[key], dtype=float)
        error = np.abs(actual-target)/scales
        handles.append(mass.plot(ds, actual[:, 0], color=color, ls=style, lw=1.9, label=label)[0])
        first.plot(ds, actual[:, 1], color=color, ls=style, lw=1.9)
        for j, ax in enumerate((mass_error, first_error)):
            ax.semilogy(ds, np.maximum(error[:, j], 1e-15), color=color, ls=style, lw=1.9)
        metrics[key] = dict(max_absolute_error=np.max(np.abs(actual-target), axis=0).tolist(),
                            max_error_over_peak_reference=np.max(error, axis=0).tolist())
    mass.set(title=r'B. Boundary mass $m_{00}=\int_{\Omega_s}W_h\,dA$', ylabel='$m_{00}$')
    first.set(title=r'C. First spatial moment $m_n=\int_{\Omega_s}(y-x)\!\cdot\!n\,W_h\,dA$', ylabel='$m_n/h$')
    for j, ax in enumerate((mass_error, first_error)):
        uncertainty = np.asarray(data['reference_uncertainty'][:, j], dtype=float)/scales[j]
        ax.fill_between(ds, 1e-15, np.maximum(uncertainty, 1e-15), color='#bdc4cc', alpha=.6)
        ax.axhline(EPS32, color='#505963', ls=(0, (2, 2)), lw=1.2)
        ax.text(.02, EPS32*1.25, 'float32 epsilon', transform=ax.get_yaxis_transform(), fontsize=9, color='#505963')
        ax.set(ylim=(1e-13, .3), ylabel='Absolute error / peak reference magnitude',
               title=('D. Mass error', 'E. First-moment error')[j])
    for ax in (mass, first, mass_error, first_error):
        ax.set(xlim=(-.95, .95), xlabel='Signed distance from the true wall, $d/h$')
        ax.axvline(0, color='#c9cfd5', lw=1)
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Curvature corrections recover the kernel moment of a curved wall', x=.51, y=.975,
                 fontsize=21, weight='bold')
    fig.text(.51, .925, 'Wendland C4 kernel  •  circle radius $R/h=3$  •  SDF grid spacing $h/64$  •  CPU float64',
             ha='center', fontsize=12, color='#505963')
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.51, .893), ncol=3,
               frameon=False, fontsize=10, columnspacing=2)
    fig.text(.055, .055, 'All errors use the same true circle. Polygon curves include the error from replacing that circle by straight edges.\n'
             'Corrected SDF and polygon values execute Scene; distance-only SDF uses a sampled distance/normal and closed-form half-plane moments.\n'
             'Grey error band: empirical curved-reference quadrature uncertainty. Precision line is a guide; this example is not a general accuracy certificate.',
             fontsize=10, color='#505963', linespacing=1.5)
    fig.savefig(output/'comparison.png', dpi=320)
    fig.savefig(output/'comparison.svg')
    plt.close(fig)
    summary = dict(radius_over_h=RADIUS, kernel='w4', sdf_spacing_over_h=1/64,
                   queries=len(ds), sdf_accepted=int(data['accepted'].sum()),
                   channel_order=['m00', 'm_n/h'], error_normalization=scales.tolist(),
                   max_reference_uncertainty=np.max(data['reference_uncertainty'], axis=0).astype(float).tolist(),
                   methods=metrics,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Curved-wall comparison for sharing', '',
             'Circle R/h=3, Wendland C4, SDF spacing h/64, 81 normal offsets. '
             'The first spatial moment is the component in the outward wall normal, m_n/h; m00 is included for context.', '',
             '[High-resolution PNG](comparison.png) · [Vector SVG](comparison.svg)', '',
             '| Method | Maximum mass error / peak | Maximum first-moment error / peak |', '|---|---:|---:|']
    for key, label in zip(METHODS, LABELS):
        e = metrics[key]['max_error_over_peak_reference']
        lines.append(f'| {label} | {e[0]:.6g} | {e[1]:.6g} |')
    lines += ['', 'All methods are compared against independent Cartesian integration of the true curved domain. '
              'The polygon is inscribed and centred on an edge at the top; 32/256/1024 edges are geometry resolutions, not quadrature orders. '
              'Thus polygon total error includes its geometric bias. Corrected SDF executes the accepted Scene model; no fallback value is substituted. '
              'Distance-only SDF uses the independent first-derivative sampler and closed-form half-plane moments. '
              'The plots retain signed first moments; errors are absolute differences normalized by the peak reference magnitude over this cut, not pointwise relative errors.', '',
              f"Maximum reference uncertainty estimates (48/96 refinement plus adaptive outer estimates): {summary['max_reference_uncertainty']}. "
              'These are empirical estimates, not certified bounds. The corrected model retains higher-order truncation and sampled-field error.', '',
              'Reproduce with `python scripts/studies/curved_boundary_example.py`. '
              '`--report-only` redraws data.npz. Generated evidence defaults to the external development archive.', '']
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=evidence_path('curved-boundary-example'))
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if not args.report_only:
        measure(args.output)
    report(args.output)
