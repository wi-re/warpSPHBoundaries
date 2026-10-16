"""Reproduce a corner's curvature pole and compare actual production paths.

The forced T3 field deliberately bypasses validity. The guarded field executes
Scene.precompute with an explicit polygon fallback; the provider field executes
AnalyticBoundary.aggregate, which lowers this SDF to that supplied polygon.
No target/reference values are substituted into either production result.
"""
import argparse
import json
from pathlib import Path

from warpSPHBoundaries.validation.paths import evidence_path

import numpy as np

from .atlas import CHANNELS, _pyplot
from .sdf_stress import CASES, sampled_model, _axes

SUPPORTED = (0, 1, 2, 6, 7, 8, 9, 10, 11)
CORNER = next(case for case in CASES if case.name == 'sharp-corner')


def production_channels(points, spacing=1/64, device='cuda:0'):
    """Evaluate the real guarded scene and fused provider in common channel order."""
    import torch
    from warpSPHCore import (GradientScheme, OperationDirection,
                             OperationProperties, ParticleState, WarpOperation)
    from ..scene.scene import Body, Scene, SdfRep, SurfaceRep
    from ..scene.provider import AnalyticBoundary
    from ..scene.fused import WallOutput

    rep = SdfRep.fromFunction(
        lambda p: torch.as_tensor(CORNER.distance(p.numpy()), dtype=torch.float64),
        (-2.5, -2.5), (2.5, 2.5), spacing,
        fallback=SurfaceRep.box(CORNER.lo, CORNER.hi))
    scene = Scene([Body(bodyId=0, reps=[rep])], device)
    positions = torch.as_tensor(np.asarray(points), dtype=torch.float64, device=device)
    ones = torch.ones(len(points), dtype=torch.float64, device=device)
    ps = ParticleState(positions=positions, supports=ones, masses=ones, densities=ones,
                       kinds=torch.zeros(len(points), dtype=torch.int32, device=device))
    props = OperationProperties(kernel='w4', operation=WarpOperation.Density,
                                gradientMode=GradientScheme.Naive,
                                operationMode=OperationDirection.BoundaryToFluid)
    pm = scene.pairMoments(ps, props)
    guarded = torch.zeros((len(points), len(SUPPORTED)), dtype=torch.float64, device=device)
    for entry in pm.entries:
        for moment in entry['surface']+entry['implicit']:
            values = torch.cat((moment.lam[:, None], moment.m1, moment.g0,
                                moment.g1.reshape(-1, 4)), dim=1)
            guarded.index_add_(0, moment.q.long(), values)
    wall = AnalyticBoundary(scene).aggregate(ps, 1., kernel='w4', lean=True)
    first = wall.evaluate((WallOutput('m1', 0, 'm1'),))['m1'][0]
    provider = torch.cat((wall.out['lam'][0, :, None], first, wall.out['G'][0],
                          wall.out['Cov'][0].reshape(-1, 4)), dim=1)
    result = []
    for values in (guarded, provider):
        channels = np.full((len(points), len(CHANNELS)), np.nan)
        channels[:, SUPPORTED] = values.detach().cpu().numpy()
        result.append(channels)
    return tuple(result)


def plot_audit(path, grid, target, raw, guarded, provider, refinement):
    from .error_colors import error_style, error_colorbar, BAND_LEGEND
    plt = _pyplot()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    n = len(grid)
    fields = [target[:, 0], raw['values'][:, 0], guarded[:, 0],
              np.abs(raw['values'][:, 0]-target[:, 0]),
              np.abs(guarded[:, 0]-target[:, 0]),
              np.abs(provider[:, 0]-target[:, 0])]
    titles = ['Polygon reference $m_{00}$', 'FORCED T3 (invalid regions included)',
              'Actual guarded scene + polygon fallback',
              'Forced absolute error', 'Guarded scene absolute error',
              'Fused provider absolute error (polygon lowering)']
    for i, values in enumerate(fields):
        ax = axes.flat[i]
        if i < 3:
            im = ax.pcolormesh(grid, grid, values.reshape(n, n), shading='nearest',
                               cmap='viridis', vmin=0, vmax=1, rasterized=True)
        else:
            im = ax.pcolormesh(grid, grid, np.ma.masked_invalid(values).reshape(n, n),
                               shading='nearest', rasterized=True, **error_style())
        _axes(ax, CORNER, grid)
        ax.set_title(titles[i], fontsize=10)
        if i < 3:
            fig.colorbar(im, ax=ax, shrink=.7, extend='both')
        else:
            error_colorbar(fig, im, ax=ax, shrink=.7, detailed=True)
    ax = axes[2, 0]
    im = ax.pcolormesh(grid, grid, raw['fallback'].reshape(n, n), shading='nearest',
                       cmap='Purples', vmin=0, vmax=1, rasterized=True)
    _axes(ax, CORNER, grid)
    ax.set_title('Actual policy requests fallback (purple)', fontsize=10)
    fig.colorbar(im, ax=ax, shrink=.7, ticks=[0, 1])
    ax = axes[2, 1]
    for j, label in enumerate(('exterior (0.2, 0.2)', 'interior diagonal (-0.4, -0.4)')):
        ax.loglog(refinement['spacing'], refinement['error'][:, j], 'o-', label=label)
    ax.set(xlabel='SDF spacing / h', ylabel='forced absolute $m_{00}$ error',
           title='Refinement does not cure the curvature pole')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    axes[2, 2].axis('off')
    axes[2, 2].text(0, .95,
        'Exterior x,y > 0:\n'
        r'$d=\sqrt{x^2+y^2},\quad\Delta d=1/d$'+'\n'
        r'$\kappa=\Delta d/(1-d\Delta d)$'+' has a pole.\n\n'
        'Interior x=y<0:\nclosest face is not unique;\nthe SDF has a medial-axis ridge.\n\n'
        'Both require a nonlocal geometry representation.\n'
        'The explicit polygon fallback integrates both faces.\n\n'
        'Top-row field colours clip at [0,1].\n'
        'Error colours share precision bands; >1 is unusable.\n'
        'Errors include numerical evaluation and model error;\n'
        'they are measured discrepancies, not bounds.', va='top', fontsize=10)
    fig.suptitle('Convex corner: forced model failure versus executed fallback\n'+BAND_LEGEND, fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def run(output, grid_size=129, dpi=300, spacing=1/64, device='cuda:0'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _pyplot().rcParams['savefig.dpi'] = dpi
    grid = np.linspace(-1.2, 1.2, grid_size)
    x, y = np.meshgrid(grid, grid)
    points = np.column_stack((x.ravel(), y.ravel()))
    target = CORNER.reference(points, 'w4')
    raw = sampled_model(CORNER, points, spacing=spacing)
    guarded, provider = production_channels(points, spacing, device)
    probe_points = np.array([[.2, .2], [-.4, -.4]])
    probe_target = CORNER.reference(probe_points, 'w4')
    spacings = 1/np.array([32., 64., 128., 256.])
    predictions = [sampled_model(CORNER, probe_points, spacing=s) for s in spacings]
    refinement = dict(spacing=spacings,
                      error=np.array([np.abs(p['values'][:, 0]-probe_target[:, 0]) for p in predictions]))
    plot_audit(output/'corner-audit.png', grid, target, raw, guarded, provider, refinement)
    np.savez_compressed(output/'data.npz', grid=grid, points=points, target=target,
                        guarded=guarded, provider=provider, **raw,
                        **{'refinement_'+k: v for k, v in refinement.items()})
    summary = dict(kernel='w4', grid_size=grid_size, spacing=spacing, device=device,
                   channels=CHANNELS, unavailable_channels=CHANNELS[3:6],
                   forced_max_absolute_error=float(np.nanmax(np.abs(raw['values']-target))),
                   guarded_max_absolute_error=float(np.nanmax(np.abs(guarded-target))),
                   provider_max_absolute_error=float(np.nanmax(np.abs(provider-target))),
                   guarded_m00_max_absolute_error=float(np.max(np.abs(guarded[:, 0]-target[:, 0]))),
                   provider_m00_max_absolute_error=float(np.max(np.abs(provider[:, 0]-target[:, 0]))),
                   guarded_fallback_max_absolute_error=float(np.nanmax(np.abs(guarded[raw['fallback']]-target[raw['fallback']]))),
                   guarded_fluid_m00_max_absolute_error=float(np.max(np.abs(
                       guarded[raw['distance'] >= 0, 0]-target[raw['distance'] >= 0, 0]))),
                   production_policy='sampled-cell-interval-v1', production_policy_changed_by_report=False,
                   reference='extended precision polygon closed form')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=evidence_path('boundary-atlas/sdf-stress/sharp-corner/audit'))
    parser.add_argument('--grid', type=int, default=129)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--sdf-spacing', type=float, default=1/64)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run(args.output, args.grid, args.dpi, args.sdf_spacing, args.device)


if __name__ == '__main__':
    main()
