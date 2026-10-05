"""Regenerate the golden fixtures (mpmath reference values, tens of seconds):
    python scripts/make_fixtures.py edge [out.json]    -> tests/fixtures/edge2d_golden.json      (edgebound.edge.fixtures)
    python scripts/make_fixtures.py fem                -> tests/fixtures/edge2d_fem_golden.json  (edgebound.edge.fem_fixtures)
"""
import sys

from edgebound.edge import fem_fixtures, fixtures

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "edge"
    if what == "edge":
        fixtures.generate(sys.argv[2] if len(sys.argv) > 2 else fixtures.DEFAULT_PATH)
    elif what == "fem":
        fem_fixtures.build()
    else:
        raise SystemExit(__doc__)
