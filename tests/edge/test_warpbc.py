"""Warp P1 pair-weight engine: golden FEM fixtures (all degenerate classes incl. tiny far elements via the far-field Gauss branch), CPU and CUDA."""
import json
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
import warp as wp

from warpSPHBoundaries.edge import warpbc
from warpSPHBoundaries.edge.fem_fixtures import PATH

DATA = json.loads(Path(PATH).read_text())
DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("k", DATA["meta"]["kernels"])
def test_p1_weights_match_golden_fixtures(device, k):
    worst, who = 0.0, None
    for c in DATA["cases"]:
        x = (F(c["x"][0]), F(c["x"][1]))
        h = float(F(c["h"]))
        rel = np.array([[float(F(a) - x[0]), float(F(b) - x[1])] for a, b in c["polygon"]])
        w, G = warpbc.pair_weights(np.array([0]), np.array([0]), np.zeros((1, 2)), np.array([h]), rel, np.array([[0, 1, 2]]), k, device=device)
        e = c["weights"]["1"][k]
        rw = np.array([float(mp.mpf(s)) for s in e["w"]])
        rG = np.array([[float(mp.mpf(s)) for s in g] for g in e["G"]])
        sc = np.abs(rw).max()
        if sc == 0:
            continue
        err = max(np.abs(w[0] - rw).max(), np.abs(G[0] - rG).max() * h) / sc
        assert err < 2e-8, (c["id"], err)
        if err > worst:
            worst, who = err, c["id"]
    assert worst < 2e-8


@pytest.mark.parametrize("name", ["quartic", "quintic", "b7", "b8", "poly6", "w2", "cubic"])
def test_all_supported_kernels_on_random_pairs(name):
    from warpSPHBoundaries.edge import np_fem
    rng = np.random.default_rng(4)
    N = 50
    V = rng.uniform(-1, 1, (N, 3, 2))
    X = rng.uniform(-.5, .5, (N, 2))
    sup = rng.uniform(0.7, 1.3, N)
    w, G = warpbc.pair_weights(np.arange(N), np.arange(N), X, sup, V.reshape(-1, 2), np.arange(3 * N).reshape(N, 3), name, device=DEVICES[-1])
    wn, Gn = np_fem.weights(V, X, name, 1, sup, grad=True)
    assert np.abs(w - wn).max() < 2e-11 and np.abs(G - Gn).max() < 2e-10
