"""Generate the boundary moment atlas; install the package's scripts extra first.

    python scripts/studies/boundary_atlas.py
    python scripts/studies/boundary_atlas.py --radii 4 --grid 17 --cut 17

See warpSPHBoundaries.validation.atlas for adapter and error-budget conventions.
"""
from warpSPHBoundaries.validation.atlas import main

if __name__ == '__main__':
    main()
