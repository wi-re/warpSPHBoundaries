"""edgebound selects float64 as the warpSPHCore precision when nothing is configured (edge/precision.py), and that has to happen before any test module imports warpSPHCore
(warpSPHCore's own default is float32): importing edgebound here, before the test modules are collected, guarantees it."""
import edgebound  # noqa: F401
