"""warpSPHBoundaries selects float64 as the warpSPHCore precision when nothing is configured (edge/precision.py), and that has to happen before any test module imports warpSPHCore
(warpSPHCore's own default is float32): importing warpSPHBoundaries here, before the test modules are collected, guarantees it."""
import warpSPHBoundaries  # noqa: F401
