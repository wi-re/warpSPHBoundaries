"""Diagnostic-only forced T3 predictions where local SDF assumptions can fail.

These examples call the existing sampled SdfRep and _sdfTier3 validity policy;
they do not change production thresholds or turn off production fallbacks.
The raw prediction is intentionally displayed even when that policy rejects it.
Discrepancies include SDF reconstruction, local-model truncation and table
evaluation; they are NOT reported as isolated model errors or certified bounds.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

import numpy as np

from .atlas import CHANNELS, LABELS, _pyplot, disk_reference, polygon_channels


@dataclass(frozen=True)
class StressCase:
    name: str
    assumption: str
    kind: str
    lo: tuple = (-4., -4.)
    hi: tuple = (0., 0.)
    radius: float = .25
    centres: tuple = ((0., 0.),)
    cut_axis: str = 'x'

    def distance(self, points):
        points = np.asarray(points, dtype=float)
        if self.kind == 'disks':
            return np.min([np.linalg.norm(points-c, axis=1)-self.radius for c in self.centres], axis=0)
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        q = np.abs(points-(lo+hi)/2)-(hi-lo)/2
        return np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(axis=1), 0)

    def reference(self, points, kernel, nodes=96):
        if self.kind == 'disks':
            # All disks in these fixtures are disjoint: sums are the exact union
            # integral. No use of a min-SDF as an integration reference.
            return sum(disk_reference(np.asarray(points)-c, self.radius, kernel, nodes) for c in self.centres)
        (x0, y0), (x1, y1) = self.lo, self.hi
        vertices = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return polygon_channels(vertices, points, kernel, dtype=np.longdouble)

    def outline(self, ax):
        if self.kind == 'disks':
            angle = np.linspace(0, 2*np.pi, 400)
            for x, y in self.centres:
                ax.plot(x+self.radius*np.cos(angle), y+self.radius*np.sin(angle), color='#252525', lw=.7)
        else:
            (x0, y0), (x1, y1) = self.lo, self.hi
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color='#252525', lw=.7)

    def cut_points(self, cut):
        if self.cut_axis == 'y':
            return np.column_stack([np.zeros_like(cut), cut])
        if self.cut_axis == 'diagonal':
            return np.column_stack([cut, cut])/np.sqrt(2)
        return np.column_stack([cut, np.zeros_like(cut)])


CASES = (
    StressCase('tight-disk', 'High curvature: R/h=0.25, so h kappa=4', 'disks'),
    StressCase('thin-strip', 'Opposite solid boundary inside support: thickness/h=0.24',
               'box', lo=(-4., -.12), hi=(4., .12), cut_axis='y'),
    StressCase('sharp-corner', 'No smooth closest surface at a convex corner', 'box', cut_axis='diagonal'),
    StressCase('narrow-gap', 'Two boundary sheets: gap/h=0.15 and a medial axis',
               'disks', radius=3., centres=((-3.075, 0.), (3.075, 0.))),
)


def sampled_model(case, points, kernel='w4', spacing=1/64):
    """Raw T3/planar channels plus the UNMODIFIED production fallback decision."""
    import torch
    from ..scene.scene import SdfRep

    # Query window +/-1.2 plus every h-radius probe fits inside +/-2.5.
    rep = SdfRep.fromFunction(lambda p: torch.as_tensor(case.distance(p.numpy()), dtype=torch.float64),
                             (-2.5, -2.5), (2.5, 2.5), spacing)
    return sampled_rep_model(rep, points, kernel)


def sampled_rep_model(rep, points, kernel='w4'):
    """Evaluate a CPU sampled SDF, permitting shared grids across stress kernels."""
    import torch
    from ..scene.scene import _sdfTier3
    from ..scene.implicitBodies import Tier3

    positions = torch.as_tensor(np.asarray(points, dtype=float), dtype=torch.float64)
    supports = torch.ones(len(points), dtype=torch.float64)
    d, normal, curvature, norm = rep.signed(positions)
    table = Tier3(kernel, 'cpu')
    variation, variation_covered = rep.curvatureVariation(positions)
    _, _, fallback = _sdfTier3(rep, positions, supports, kernel, 'cpu')
    out = table.curvature_moments(d, normal, curvature, variation).numpy()
    # Ablation: the distance/normal-only half-plane model is distinct from the
    # curvature-corrected series. In particular, an exact exterior-corner SDF
    # has a pole in recovered *surface* curvature, not in its distance.
    planar = table.curvature_moments(d, normal, torch.zeros_like(d), torch.zeros_like(d)).numpy()
    active = d.numpy() < 1
    full = d.numpy() <= -1-np.sqrt(2.)*rep.spacing
    accepted = (~fallback.numpy()) & active & ~full
    return dict(values=out, planar_values=planar, distance=d.numpy(), normal=normal.numpy(), curvature=curvature.numpy(),
                variation=variation.numpy(), variation_covered=variation_covered.numpy(),
                implemented_channels=CHANNELS, model_version='shape-order2-v1',
                gradient_norm=norm.numpy(), accepted=accepted, fallback=fallback.numpy(), active=active)


def _axes(ax, case, grid):
    case.outline(ax)
    ax.set(xlim=(grid[0], grid[-1]), ylim=(grid[0], grid[-1]), aspect='equal')


def plot_moments(output, case, grid, target, prediction, channels):
    """Keep the ordinary atlas's channel order; expose rejection and saturation."""
    from .error_colors import error_style, error_colorbar, BAND_LEGEND
    plt = _pyplot()
    fig, axes = plt.subplots(4, 6, figsize=(18, 11), constrained_layout=True)
    count = len(grid)
    for col, channel in enumerate(channels):
        ref = target[:, channel].reshape(count, count)
        raw = prediction['values'][:, channel].reshape(count, count)
        error = np.abs(raw-ref)
        accepted = prediction['accepted'].reshape(count, count)
        limit = max(float(np.max(np.abs(ref))), 1e-12)
        for row, values in enumerate((ref, raw, error, np.where(accepted, error, np.nan))):
            ax = axes[row, col]
            ax.set_facecolor('#e9e9e9')
            if row < 2:
                im = ax.pcolormesh(grid, grid, np.ma.masked_invalid(values), shading='nearest',
                                   cmap='RdBu_r', vmin=-limit, vmax=limit, rasterized=True)
            else:
                im = ax.pcolormesh(grid, grid, np.ma.masked_invalid(values),
                                   shading='nearest', rasterized=True, **error_style())
            _axes(ax, case, grid)
            if channel in (3, 4, 5) and row and not np.isfinite(prediction['values'][:, channel]).any():
                ax.text(.5, .5, 'undefined forced model', transform=ax.transAxes, ha='center', color='#555555')
            if row == 0:
                ax.set_title(LABELS[channel])
            if col == 0:
                ax.set_ylabel(('Reference', 'Forced curvature series', 'Total absolute error', 'Error where policy accepts')[row]+'\n$y/h$')
            else:
                ax.tick_params(labelleft=False)
            if row == 3:
                ax.set_xlabel('$x/h$')
            else:
                ax.tick_params(labelbottom=False)
            if row in (1, 3):
                if row == 3:
                    error_colorbar(fig, im, ax=axes[row-1:row+1, col], shrink=.5, pad=.01)
                else:
                    fig.colorbar(im, ax=axes[row-1:row+1, col], shrink=.5, pad=.01, extend='both')
    fig.suptitle(f'{case.name}: FORCED DIAGNOSTIC — GUARD BYPASSED; this is not the executed solver output\n'
                 'Field colours saturate at reference scale. Grey: undefined / missing; last row masks fallback / outside support.\n'
                 'Total error includes SDF sampling, model and tables.\n'+BAND_LEGEND, fontsize=10)
    fig.savefig(output)
    plt.close(fig)


def publish_guarded_corner(folder, checkpoint):
    """Render current executed corner moments from the guard checkpoint only.

    No evaluator is run and no numerical cache is rewritten. Versioned image
    names prevent an already-open browser from reusing the old forced figure.
    """
    from .error_colors import error_style, error_colorbar, BAND_LEGEND
    folder, checkpoint = Path(folder), Path(checkpoint)
    source = checkpoint/'corner.npz'
    metadata = json.loads((checkpoint/'checkpoint.json').read_text())
    if metadata.get('policy') != 'sampled-cell-interval-v1':
        raise ValueError('Guarded corner requires a current sampled-cell checkpoint')
    provenance = dict(render_version=1, policy=metadata['policy'], device=metadata['device'],
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    stamp = folder/'guarded-provenance.json'
    files = ('guarded-v1-moments.png', 'guarded-v1-gradient-moments.png')
    if stamp.exists() and json.loads(stamp.read_text()) == provenance and all((folder/f).exists() for f in files):
        return provenance
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(source) as data:
        grid, target, guarded = data['grid'], data['target'], data['guarded']
        if target.shape != (len(grid)**2, len(CHANNELS)) or guarded.shape != target.shape:
            raise ValueError('Unexpected guarded corner grid/channel shape')
    plt = _pyplot()
    case = next(c for c in CASES if c.name == 'sharp-corner')
    for filename, channels in zip(files, (range(6), range(6, 12))):
        fig, axes = plt.subplots(3, 6, figsize=(18, 8.5), constrained_layout=True)
        for col, channel in enumerate(channels):
            ref = target[:, channel].reshape(len(grid), len(grid))
            values = guarded[:, channel].reshape(ref.shape)
            error = np.abs(values-ref)
            limit = max(float(np.max(np.abs(ref))), 1e-12)
            for row, field in enumerate((ref, values, error)):
                ax = axes[row, col]
                ax.set_facecolor('#e9e9e9')
                if row < 2:
                    im = ax.pcolormesh(grid, grid, np.ma.masked_invalid(field), shading='nearest',
                                       cmap='RdBu_r', vmin=-limit, vmax=limit, rasterized=True)
                else:
                    im = ax.pcolormesh(grid, grid, np.ma.masked_invalid(field), shading='nearest',
                                       rasterized=True, **error_style())
                _axes(ax, case, grid)
                if row and channel in (3, 4, 5):
                    ax.text(.5, .5, 'checkpoint:\nnot recorded', transform=ax.transAxes, ha='center', va='center')
                if row == 0:
                    ax.set_title(LABELS[channel])
                if col == 0:
                    ax.set_ylabel(('Reference', 'Executed guarded Scene\n+ polygon fallback', 'Total absolute error')[row]+'\n$y/h$')
                else:
                    ax.tick_params(labelleft=False)
                if row == 2:
                    ax.set_xlabel('$x/h$')
                    error_colorbar(fig, im, ax=ax, shrink=.65, pad=.01)
                else:
                    ax.tick_params(labelbottom=False)
                    if row == 1:
                        fig.colorbar(im, ax=axes[:2, col], shrink=.65, pad=.01)
        fig.suptitle(f"sharp-corner: CURRENT GUARDED RESULT — {metadata['device']}, {metadata['policy']}\n"
                     'Actual Scene output with original-polygon fallback; not a masked forced prediction. Grey: unsupported channels.\n'+BAND_LEGEND,
                     fontsize=10)
        fig.savefig(folder/filename, dpi=300)
        plt.close(fig)
    stamp.write_text(json.dumps(provenance, indent=2)+'\n')
    return provenance


def plot_cut(output, case, cut, target, uncertainty, prediction):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    raw = prediction['values']
    error = np.abs(raw-target)
    ax = axes[0, 0]
    ax.plot(cut, target[:, 0], color='#222222', label='target geometry reference')
    ax.plot(cut, raw[:, 0], color='#b34d23', label='forced sampled-SDF curvature series')
    ax.plot(cut, prediction['planar_values'][:, 0], '--', color='#1764ab', label='distance-only half-plane ablation')
    ax.plot(cut, np.where(prediction['accepted'], raw[:, 0], np.nan), color='#26834a', lw=2, label='model accepted by current policy')
    ax.set(ylabel='$m_{00}$', ylim=(-.25, 1.25))
    ax.text(.02, .96, 'Prediction clipped by axis limits where divergent', transform=ax.transAxes, va='top', fontsize=8)
    ax.legend(fontsize=8)
    axes[0, 1].semilogy(cut, np.maximum(error[:, 0], 1e-16), color='#b34d23', label='total $m_{00}$ discrepancy')
    if np.isfinite(uncertainty[:, 0]).any():
        axes[0, 1].semilogy(cut, np.maximum(uncertainty[:, 0], 1e-16), ':', color='#666666', label='reference refinement estimate')
    axes[0, 1].axhline(.01, color='#aaaaaa', ls='--', label='0.01 diagnostic threshold')
    axes[0, 1].set(ylabel='absolute error', ylim=(1e-16, 1e3))
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].semilogy(cut, np.maximum(np.abs(prediction['curvature']), 1e-8), label=r'$|h\kappa|$')
    axes[1, 0].semilogy(cut, np.maximum(np.abs(prediction['gradient_norm']-1), 1e-8), label=r'$||\nabla d|-1|$')
    axes[1, 0].axhline(.5, color='#3279ac', ls=':', label='curvature threshold 0.5')
    axes[1, 0].axhline(.1, color='#d0801b', ls=':', label='gradient threshold 0.1')
    axes[1, 0].set(ylabel='local validity diagnostics', ylim=(1e-8, 1e6))
    axes[1, 0].legend(fontsize=8)
    for j in (1, 2, 6, 7, 8, 11):
        axes[1, 1].semilogy(cut, np.maximum(error[:, j], 1e-16), label=LABELS[j])
    axes[1, 1].set(ylabel='absolute error (support units)', ylim=(1e-16, 1e3))
    axes[1, 1].legend(fontsize=8, ncol=2)
    for ax in axes.flat:
        ax.fill_between(cut, 0, 1, where=prediction['fallback'], transform=ax.get_xaxis_transform(),
                        color='#aa99bb', alpha=.18)
        ax.set(xlabel=f'{case.cut_axis} cut coordinate / h')
        ax.grid(alpha=.15)
    fig.suptitle(f'{case.name} | {case.assumption}\n'
                 'Purple bands: fallback requested by the sampled-cell interval guard. Estimates are not certified bounds.', fontsize=11)
    fig.savefig(output)
    plt.close(fig)


def plot_summary(output, grid, records):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from .error_colors import error_style, error_colorbar, BAND_LEGEND
    plt = _pyplot()
    fig, axes = plt.subplots(len(records), 4, figsize=(15, 13), constrained_layout=True)
    count = len(grid)
    for row, (case, target, prediction) in enumerate(records):
        raw = prediction['values'][:, 0].reshape(count, count)
        ref = target[:, 0].reshape(count, count)
        status = np.where(prediction['fallback'], 1, np.where(prediction['active'], 0, 2)).reshape(count, count)
        status = np.where((status == 0) & (np.abs(raw-ref) > .01), 3, status)
        for col, values in enumerate((ref, raw, np.abs(raw-ref), status)):
            ax = axes[row, col]
            ax.set_facecolor('#dddddd')
            if col < 2:
                im = ax.pcolormesh(grid, grid, np.ma.masked_invalid(values), cmap='viridis', vmin=0, vmax=1, shading='nearest', rasterized=True)
            elif col == 2:
                im = ax.pcolormesh(grid, grid, np.ma.masked_invalid(values), shading='nearest', rasterized=True, **error_style())
            else:
                im = ax.pcolormesh(grid, grid, values, cmap=ListedColormap(['#64b980', '#b69bce', '#ededed', '#e69b38']),
                                   norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4), shading='nearest', rasterized=True)
            _axes(ax, case, grid)
            if row == 0:
                ax.set_title(('Target $m_{00}$', 'Forced curvature series $m_{00}$', 'Total absolute error', 'Policy + observed error')[col])
            if col == 0:
                ax.set_ylabel(case.name+'\n$y/h$')
            else:
                ax.tick_params(labelleft=False)
            if row == len(records)-1:
                ax.set_xlabel('$x/h$')
            if row == len(records)-1:
                if col == 2:
                    bar = error_colorbar(fig, im, ax=axes[:, col], shrink=.45, pad=.01, detailed=True)
                else:
                    bar = fig.colorbar(im, ax=axes[:, col], shrink=.45, pad=.01)
                if col == 3:
                    bar.set_ticks([0, 1, 2, 3], labels=['accepted', 'fallback', 'outside support', 'accepted; error > 0.01'])
    fig.suptitle('Where local SDF boundary models break down\n'
                 'Forced predictions are diagnostic extrapolations. Value colours clip at [0,1].\n'
                 'Orange policy cells: accepted with observed scalar error > 0.01 (retrospective diagnostic). Current sampled-cell interval guard.\n'+BAND_LEGEND, fontsize=10)
    fig.savefig(output)
    plt.close(fig)


def run_stress(output, kernel='w4', grid_size=129, cut_size=129, spacing=1/64):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(-1.2, 1.2, grid_size)
    x, y = np.meshgrid(grid, grid)
    points = np.column_stack([x.ravel(), y.ravel()])
    cut = np.linspace(-1.2, 1.2, cut_size)
    records, summaries = [], []
    for case in CASES:
        print(f'Building SDF stress case: {case.name}', flush=True)
        folder = output/case.name
        folder.mkdir(exist_ok=True)
        prediction = sampled_model(case, points, kernel, spacing)
        target = case.reference(points, kernel)
        cut_points = case.cut_points(cut)
        cut_prediction = sampled_model(case, cut_points, kernel, spacing)
        cut_target = case.reference(cut_points, kernel)
        if case.kind == 'disks':
            uncertainty = np.abs(cut_target-case.reference(cut_points, kernel, 48))
        else:
            uncertainty = np.full_like(cut_target, np.nan)  # no certified or refinement estimate for this closed form
        plot_moments(folder/'moments.png', case, grid, target, prediction, range(6))
        plot_moments(folder/'gradient-moments.png', case, grid, target, prediction, range(6, 12))
        plot_cut(folder/'cut-diagnostics.png', case, cut, cut_target, uncertainty, cut_prediction)
        np.savez_compressed(folder/'data.npz', grid=grid, points=points, target=target,
                            total_error=prediction['values']-target, cut=cut, cut_points=cut_points,
                            cut_target=cut_target, cut_reference_estimate=uncertainty,
                            **prediction, **{'cut_'+k: v for k, v in cut_prediction.items()})
        fluid_near = (prediction['distance'] >= 0) & prediction['active']
        accepted = prediction['accepted'] & fluid_near
        error = np.abs(prediction['values'][:, 0]-target[:, 0])
        finite = np.isfinite(prediction['values'][:, 0])
        summaries.append({'name': case.name, 'assumption': case.assumption,
                          'geometry': asdict(case),
                          'reference': 'disjoint disk polar integrals' if case.kind == 'disks' else 'extended-precision polygon closed form',
                          'near_fluid_samples': int(fluid_near.sum()), 'accepted_near_fluid_samples': int(accepted.sum()),
                          'accepted_samples_with_m00_error_over_0_01': int((accepted & (error > .01)).sum()),
                          'nonfinite_forced_m00_samples': int((~finite).sum()),
                          'forced_m00_outside_0_1': int((finite & ((prediction['values'][:, 0] < 0) | (prediction['values'][:, 0] > 1))).sum()),
                          'max_accepted_m00_error': float(error[accepted & finite].max()) if (accepted & finite).any() else None})
        records.append((case, target, prediction))
    plot_summary(output/'overview.png', grid, records)
    summary = {'kernel': kernel, 'support': 1, 'grid_size': grid_size, 'sdf_spacing_over_h': spacing,
               'export_dpi': float(_pyplot().rcParams['savefig.dpi']),
               'error_type': 'total discrepancy: SDF reconstruction + local model + table evaluation',
               'diagnostic_threshold': .01, 'diagnostic_threshold_is_acceptance_requirement': False,
               'forced_predictions_are_validated_solutions': False, 'production_policy': 'sampled-cell-interval-v1',
               'production_policy_changed_by_report': False,
               'cases': summaries}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
