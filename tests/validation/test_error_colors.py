"""Error colours preserve precision regimes without spending range above 1."""
import numpy as np
import pytest

pytest.importorskip('matplotlib')
from matplotlib.colors import to_rgba

from warpSPHBoundaries.validation.error_colors import (
    EPS64, EPS32, UNUSABLE, MISSING, PrecisionErrorNorm, error_style, error_colorbar)
from warpSPHBoundaries.validation.atlas import _pyplot


def test_precision_boundaries_have_equal_colour_band_width_and_invert():
    norm = PrecisionErrorNorm()
    np.testing.assert_allclose(norm([EPS64, EPS32, 1]), [0, .5, 1], atol=1e-15, rtol=0)
    values = np.geomspace(EPS64, 1, 101)
    assert (np.diff(norm(values)) > 0).all()
    np.testing.assert_allclose(norm.inverse(norm(values)), values, rtol=8e-15)


def test_every_unusable_magnitude_has_the_same_colour():
    style = error_style()
    rgba = style['cmap'](style['norm']([1+1e-10, 10., 1e10, 1e25, np.inf]))
    np.testing.assert_array_equal(rgba, np.tile(to_rgba(UNUSABLE), (5, 1)))
    assert not np.array_equal(style['cmap'](style['norm'](1)), rgba[0])


def test_single_epsilon_visibly_separates_the_two_precision_bands():
    style = error_style()
    colours = style['cmap'](style['norm']([EPS32*(1-1e-8), EPS32]))
    assert np.linalg.norm(colours[0, :3]-colours[1, :3]) > .5


def test_zero_and_sub_double_errors_have_a_floor_and_missing_is_distinct():
    style = error_style()
    rgba = style['cmap'](style['norm']([0., EPS64/2, EPS64, np.nan]))
    np.testing.assert_array_equal(rgba[:3], np.tile(rgba[0], (3, 1)))
    np.testing.assert_array_equal(rgba[-1], to_rgba(MISSING))
    assert not np.array_equal(rgba[0], rgba[-1])


def test_masked_values_and_extreme_longdouble_do_not_lose_their_status():
    style = error_style()
    values = np.ma.array([EPS64, 1e30], mask=[True, False], dtype=np.longdouble)
    with np.errstate(all='raise'):
        normed = style['norm'](values)
        extreme = style['norm'](np.finfo(np.longdouble).max)
    assert normed.mask.tolist() == [True, False]
    assert extreme > 1


def test_colourbar_exposes_both_epsilon_boundaries_and_unit_cutoff(tmp_path):
    plt = _pyplot()
    fig, ax = plt.subplots()
    im = ax.imshow([[0., EPS64], [EPS32, 1e25]], **error_style())
    bar = error_colorbar(fig, im, ax=ax)
    np.testing.assert_array_equal(bar.get_ticks(), [EPS64, EPS32, 1.])
    assert bar.extend == 'both'
    fig.savefig(tmp_path/'precision-bands.png')
    plt.close(fig)
