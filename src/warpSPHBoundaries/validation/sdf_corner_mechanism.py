"""Separate a valid corner distance field from the singular curvature model.

Run python -m warpSPHBoundaries.validation.sdf_corner_mechanism. This is an
offline ablation, not a production fallback or a new acceptance policy.
"""
import argparse
import json
from pathlib import Path

from warpSPHBoundaries.validation.paths import evidence_path

import numpy as np

from ..edge import np2d
from .atlas import CHANNELS, _pyplot
from .corner_audit import CORNER
from .sdf_stress import sampled_model


def planar_mass_reference(distance, kernel='w4'):
    """Independent half-plane closed form, at unit support and |d| <= 1.5.

    The other three sides of this rectangle are outside every support ball.
    This calls the polygon integral, independently of the Tier3 lookup table.
    """
    distance = np.asarray(distance, dtype=np.longdouble)
    if np.any(np.abs(distance) > 1.5):
        raise ValueError('reference rectangle requires |d| <= 1.5')
    points = np.column_stack((np.zeros_like(distance), distance))
    vertices = np.array([[-4, -4], [4, -4], [4, 0], [-4, 0]], dtype=np.longdouble)
    vertices = np.broadcast_to(vertices, (len(points), 4, 2))
    return np2d.moment(vertices, points, kernel, (0, 0), dtype=np.longdouble, unsplit=True)


def plot_mechanism(path, cut, target, prediction, planar_reference, angle, arc_target, arc_prediction):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(cut, target[:, 0], color='#222222', label='original corner: polygon integral')
    ax.plot(cut, planar_reference, '--', color='#1764ab', label='half-plane: exact distance + closed form')
    ax.plot(cut, prediction['planar_values'][:, 0], ':', color='#d0801b', label='half-plane: sampled distance + table')
    ax.set(xlabel='$x/h$ along $y/h=0.2$', ylabel='$m_{00}$', ylim=(-.01, .24),
           title='Distance-only model remains continuous')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    raw = prediction['values'][:, 0]
    ax.plot(cut, raw, color='#b34d23', label='forced curvature series: raw values')
    ax.plot(cut, target[:, 0], color='#222222', label='original corner')
    ax.set_yscale('symlog', linthresh=1)
    ax.axhspan(0, 1, color='#64b980', alpha=.2, label='physical mass range [0, 1]')
    ax.set(xlabel='$x/h$ along $y/h=0.2$', ylabel='$m_{00}$ (symlog scale)',
           title='The old colour scale hid values far above 1')
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(angle, arc_target[:, 0], color='#222222', label='original corner')
    ax.plot(angle, arc_prediction['planar_values'][:, 0], color='#d0801b', label='sampled-distance half-plane')
    ax.axhline(float(planar_mass_reference([.2])[0]), color='#1764ab', ls='--', label='exact-distance half-plane (constant)')
    ax.set(xlabel='angle around vertex (degrees)', ylabel='$m_{00}$',
           title='Constant distance $d/h=0.2$: geometry bias varies with angle')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    # Algebraically equivalent to 1-d*lap(d), using the public signed() result.
    denominator = 1/(1+prediction['distance']*prediction['curvature'])
    ax.plot(cut, denominator, color='#8d4aa8', label='sampled curvature denominator')
    exact_denominator = np.where(cut == 0, np.nan, np.where(cut > 0, 0., 1.))
    ax.plot(cut, exact_denominator, '--', color='#222222', label='exact denominator; seam undefined')
    ax.set_yscale('symlog', linthresh=1e-5)
    ax.set(xlabel='$x/h$ along $y/h=0.2$', ylabel=r'$1-d\,\Delta d$',
           title=r'Exterior corner: $\Delta d=1/d$; curvature denominator vanishes')
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.axvline(0, color='#888888', ls=':', lw=.8)
    fig.suptitle('Valid signed distance; singular smooth-surface curvature model\n'
                 'Wendland C4, h=1, SDF spacing h/64. Forced fields are diagnostic; selection uses the sampled-cell guard.', fontsize=12)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_budget(path, cut, target, prediction, planar_reference, spacings, refinement):
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    geometry = planar_reference-target[:, 0]
    evaluation = prediction['planar_values'][:, 0]-planar_reference
    for values, label, color in (
        (geometry, 'half-plane geometry/model bias', '#1764ab'),
        (evaluation, 'distance sampling + table/evaluation residual', '#d0801b'),
        (prediction['values'][:, 0]-target[:, 0], 'forced curvature series: total discrepancy', '#b34d23')):
        axes[0].semilogy(cut, np.maximum(np.abs(values), 1e-18), color=color, label=label)
    axes[0].set(xlabel='$x/h$ along $y/h=0.2$', ylabel='absolute mass discrepancy', title='Separate half-plane model and evaluation contributions')
    axes[0].legend(fontsize=8)
    for j, (label, color) in enumerate((('forced curvature series: total', '#b34d23'),
                                       ('distance-only half-plane: total', '#1764ab'),
                                       ('distance-only: sampling/evaluation', '#d0801b'))):
        axes[1].loglog(spacings, refinement[:, j], 'o-', color=color, label=label)
    axes[1].set(xlabel='SDF grid spacing / h', ylabel='absolute mass discrepancy', title='At (0.2, 0.2): refinement cannot recover finite corner curvature')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axhline(np.finfo(np.float32).eps, color='#26834a', ls=':', label=None)
        ax.axhline(np.finfo(np.float64).eps, color='#222222', ls=':', label=None)
        ax.grid(alpha=.2)
    fig.suptitle('Measured errors, not certified bounds. Dotted levels: float32 and float64 epsilon.\n'
                 'The planar budget adds with signs before absolute values; it is not a curvature-series error decomposition.', fontsize=10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run(output, samples=601):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cut = np.linspace(-.6, .6, samples)
    points = np.column_stack((cut, np.full_like(cut, .2)))
    target = CORNER.reference(points, 'w4')
    prediction = sampled_model(CORNER, points)
    exact_distance = CORNER.distance(points)
    planar_reference = planar_mass_reference(exact_distance)
    angle = np.linspace(0., 90., 181)
    theta = np.deg2rad(angle)
    arc_points = .2*np.column_stack((np.cos(theta), np.sin(theta)))
    arc_target = CORNER.reference(arc_points, 'w4')
    arc_prediction = sampled_model(CORNER, arc_points)
    spacings = 1/np.array([32., 64., 128., 256.])
    probe = np.array([[.2, .2]])
    probe_target = CORNER.reference(probe, 'w4')[0, 0]
    probe_planar = planar_mass_reference(CORNER.distance(probe))[0]
    refined = [sampled_model(CORNER, probe, spacing=s) for s in spacings]
    refinement = np.array([[abs(p['values'][0, 0]-probe_target),
                            abs(p['planar_values'][0, 0]-probe_target),
                            abs(p['planar_values'][0, 0]-probe_planar)] for p in refined])
    plot_mechanism(output/'mechanism.png', cut, target, prediction, planar_reference,
                   angle, arc_target, arc_prediction)
    plot_budget(output/'error-budget.png', cut, target, prediction, planar_reference, spacings, refinement)
    np.savez_compressed(output/'data.npz', channels=CHANNELS, cut=cut, points=points,
                        target=target, exact_distance=exact_distance, planar_reference=planar_reference,
                        planar_model_error=planar_reference-target[:, 0],
                        planar_evaluation_error=prediction['planar_values'][:, 0]-planar_reference,
                        angle=angle, arc_points=arc_points, arc_target=arc_target,
                        arc_planar=arc_prediction['planar_values'], spacings=spacings,
                        refinement=refinement, **prediction)
    base = refined[1]
    summary = dict(kernel='w4', samples=samples, spacing=1/64, probe=probe[0].tolist(),
                   reference_mass=float(probe_target), forced_curvature_mass=float(base['values'][0, 0]),
                   sampled_planar_mass=float(base['planar_values'][0, 0]),
                   exact_distance_planar_mass=float(probe_planar),
                   reconstructed_curvature=float(base['curvature'][0]),
                   gradient_norm=float(base['gradient_norm'][0]),
                   fallback_requested=bool(base['fallback'][0]),
                   constant_distance_reference_range=[float(arc_target[:, 0].min()), float(arc_target[:, 0].max())],
                   production_policy='sampled-cell-interval-v1', production_policy_changed_by_report=False)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lo, hi = summary['constant_distance_reference_range']
    report = f'''# Corner SDF: distance-only bias versus a curvature pole

The exterior corner has a valid exact signed distance. The previous **forced SDF**
label meant the second-order smooth-surface curvature series, not a distance-only
half-plane approximation. Its field colour scale clipped values to [0,1]; the
apparent jump to 1 concealed much larger raw predictions.

![Distance-only continuity and curvature-series pole](mechanism.png)

For the solid x<=0, y<=0 and exterior x,y>0, d=sqrt(x²+y²), |grad d|=1 and
Delta d=1/d. The code converts offset-level curvature into closest-surface
curvature with kappa=Delta d/(1-d Delta d). Its denominator is exactly zero
throughout this exterior quadrant. The offset contours are smooth circles;
their common closest boundary point is a nonsmooth vertex, with no finite
surface curvature. At the transition from a face projection to a vertex
projection, d and its gradient remain continuous but its second derivatives
change. The SDF is valid; the smooth-surface expansion is outside its domain.

Sampling perturbs the zero denominator, producing large finite curvatures
with either sign. This is not a physical mass discontinuity and is not fixed
by refining the distance grid or clamping the mass. At (0.2,0.2), h=1,
Wendland C4 and spacing h/64:

| Quantity | Value |
|---|---:|
| Original polygon mass | {probe_target:.12g} |
| Half-plane, exact distance, independent closed form | {probe_planar:.12g} |
| Half-plane, sampled distance and production table | {base['planar_values'][0, 0]:.12g} |
| Forced second-order curvature series, raw mass | {base['values'][0, 0]:.12g} |
| Recovered h*kappa | {base['curvature'][0]:.12g} |
| Sampled gradient norm | {base['gradient_norm'][0]:.12g} |
| Actual policy requests fallback | {bool(base['fallback'][0])} |

The distance-only prediction behaves as expected: continuous across the seam,
and constant along an **exact constant-distance** arc. Its error relative to
the actual corner is not constant on that arc: the reference mass spans
{lo:.8g} to {hi:.8g} at d/h=0.2. The amount of solid inside the kernel depends
on angle as well as distance. A radial kernel and one distance value do not
remove that information. Sampled distance introduces a small additional
angular variation into the half-plane prediction.

![Half-plane error budget and SDF refinement](error-budget.png)

For the half-plane ablation, the signed split is:

```
model/geometry bias = I(half-plane at exact distance) - I(original corner)
sampling/evaluation = table(sampled distance, kappa=0) - I(half-plane at exact distance)
total               = model/geometry bias + sampling/evaluation
```

This isolates the half-plane geometric bias from distance sampling and table
evaluation. It is distinct from the survey's sampled-contour representation
budget, and does not isolate a meaningful finite curvature-series model at
the pole. No quadrature is used for these references: the independent polygon
closed form runs in extended precision, including the half-plane rectangle
whose other sides are beyond support. Errors are measured discrepancies, not
certified bounds. The Maple identity checks are in maple/15_sdf_corner.mpl.

Raw data retains the standard twelve-channel order; unavailable scalar second
moments in the SDF models remain NaN. This focused report plots m00 only.
The forced series and distance-only fields are diagnostics. The production
sampled-cell interval guard requests the polygon fallback at this probe;
see the [guard checkpoint](../sdf-guard-cpu/report.md). It checks the blended
jets throughout the support square, but does not certify model truncation
accuracy or resolve geometry absent from the samples.

Reproduce from the repository root:

```
python -m warpSPHBoundaries.validation.sdf_corner_mechanism
```
'''
    (output/'report.md').write_text(report)
    from .report_view import accessible_report
    accessible_report(output/'report.md')
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=evidence_path('sdf-corner-mechanism'))
    parser.add_argument('--samples', type=int, default=601)
    args = parser.parse_args()
    run(args.output, args.samples)


if __name__ == '__main__':
    main()
