"""2D tier-selection study: ordering of the models vs R/h for a disk obstacle (records the regime map used in docs/tier-selection-2d.md)."""

from warpSPHBoundaries.edge import tier_select


def test_regime_map():
    rows = {r["R"]: r for r in tier_select.run(verbose=False)}
    # tier 4 series: excellent for small obstacles, unavailable once the disk touches the support rim / knot
    assert rows[0.05]["t4_K2"] < 1e-8 and rows[0.2]["t4_K2"] < 1e-4 and rows[0.5]["t4_K2"] is None
    # tier 3 needs R > h - d and improves with R; second order always beats planar there
    for R in (1.0, 2.0, 4.0, 8.0, 16.0):
        assert rows[R]["t3_2"] < rows[R]["t3_0"] / 10
    assert rows[16.0]["t3_2"] < rows[4.0]["t3_2"] < rows[1.0]["t3_2"]
    # tier 3 second order beats a polygon with edges h/8 for R >= 2h (comparable at R = h)
    assert rows[1.0]['t3_2'] < 1.5 * rows[1.0]['t2_h8']
    for R in (2.0, 4.0, 8.0, 16.0):
        assert rows[R]["t3_2"] < rows[R]["t2_h8"]
    # small obstacles: polygons with >= 8 edges are far worse than the series
    assert rows[0.1]["t4_K2"] < rows[0.1]["t2_h8"] / 1e3
    # the gap: for 0.35 <= R/h <= 0.5 neither series nor tier 3 applies (tier 2 / curved elements needed)
    assert rows[0.5]["t3_2"] is None and rows[0.5]["t4_K2"] is None
