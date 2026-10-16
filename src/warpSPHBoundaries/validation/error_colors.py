"""Shared precision bands for error heatmaps; raw magnitudes are never changed.

Two equal-width logarithmic colour bands distinguish double-to-single epsilon
from single epsilon to unit error. All errors >1 share the unusable over-colour;
zeros / errors <= double epsilon have a navy floor. Epsilon bands are visual
reference levels, not an accuracy certificate or a relative-error definition.
"""
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize

EPS64 = float(np.finfo(np.float64).eps)
EPS32 = float(np.finfo(np.float32).eps)
UNUSABLE = '#a50036'
MISSING = '#d9dde3'
BAND_LEGEND = ('Navy: ≤ε64; blue–teal: ε64–ε32; gold–orange: ε32–1; red: >1 (unusable).\n'
               'ε64 = 2.22×10⁻¹⁶; ε32 = 1.19×10⁻⁷. Epsilon levels are visual guides, not certified bounds.')


class PrecisionErrorNorm(Normalize):
    """Map [eps64,eps32] to [0,.5] and [eps32,1] to [.5,1], in log space."""

    def __init__(self, clip=False):
        super().__init__(vmin=EPS64, vmax=1., clip=clip)

    def __call__(self, value, clip=None):
        values, scalar = self.process_value(value)
        raw = np.ma.getdata(values)
        # Clamp only the logarithm's argument, preserving masks and the actual
        # over/under tests. Even longdouble extremes cannot overflow this cast.
        log = np.log10(np.clip(raw, EPS64, 1).astype(float))
        low = .5*(log-np.log10(EPS64))/(np.log10(EPS32)-np.log10(EPS64))
        high = .5+.5*(log-np.log10(EPS32))/-np.log10(EPS32)
        mapped = np.where(raw <= EPS32, low, high)
        mapped = np.where(raw < EPS64, -1., np.where(raw > 1, 2., mapped))
        if clip is None:
            clip = self.clip
        if clip:
            mapped = np.clip(mapped, 0, 1)
        result = np.ma.array(mapped, mask=np.ma.getmaskarray(values) | np.isnan(raw))
        return result[0] if scalar else result

    def inverse(self, value):
        value = np.ma.asarray(value)
        log = np.where(value <= .5,
                       np.log10(EPS64)+2*value*(np.log10(EPS32)-np.log10(EPS64)),
                       np.log10(EPS32)+2*(value-.5)*-np.log10(EPS32))
        result = np.ma.array(np.power(10., log), mask=np.ma.getmaskarray(value))
        return result[()] if result.ndim == 0 else result


def error_style():
    """Fresh style for each mappable: fixed scales across cases and moments."""
    precise = LinearSegmentedColormap.from_list('double_to_single', ['#112b52', '#2166ac', '#00a6b2', '#73d9c2'])
    coarse = LinearSegmentedColormap.from_list('single_to_unit', ['#fff0ad', '#f6b24c', '#ec7014', '#c5410d'])
    cmap = ListedColormap(np.vstack((precise(np.linspace(0, 1, 128)), coarse(np.linspace(0, 1, 128)))), name='precision_error')
    cmap = cmap.with_extremes(under='#112b52', over=UNUSABLE, bad=MISSING)
    return dict(cmap=cmap, norm=PrecisionErrorNorm())


def error_colorbar(fig, mappable, *, ax, detailed=False, **kwargs):
    bar = fig.colorbar(mappable, ax=ax, extend='both', **kwargs)
    ticks = [EPS64, 1e-12, EPS32, 1e-3, 1] if detailed else [EPS64, EPS32, 1]
    labels = [r'$\varepsilon_{64}$', r'$10^{-12}$', r'$\varepsilon_{32}$', r'$10^{-3}$', '1'] if detailed else [r'$\varepsilon_{64}$', r'$\varepsilon_{32}$', '1']
    bar.set_ticks(ticks, labels=labels)
    bar.ax.axhline(EPS32, color='#263238', lw=.65)
    bar.ax.set_title('>1\nunusable' if detailed else '>1', fontsize=8 if detailed else 7, pad=8)
    return bar
