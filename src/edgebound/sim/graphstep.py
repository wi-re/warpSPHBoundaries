"""CUDA-graph replay of the whole `DeltaSPH2D` step (docs/plan-wall-evaluation.md step 5b), after warpSPH's `utils/cudaGraph.py` (`GraphedStateFunction`, `deferVerletChecks`).

The eager step is CPU-bound (about 1200 small kernels, ~14 warp / torch module calls whose host cost dwarfs their GPU time).  `_step_core` has no host read and no data-dependent shape, so it is
captured once into a `torch.cuda.CUDAGraph` on a side stream shared with warp (`wp.stream_from_torch`), against private copies of the state (x, v, rho, g); `dt` lives in the persistent device buffer `dt_t`.
Every call: the host value of `dt` is written into `dt_t`, the state is copied into the static inputs, the graph is replayed, and one device-to-host read brings back (a) the OR of the deferred Verlet
validity flags and (b) the new dt.  If any flag is set the replayed result is discarded and the step is run eagerly (where the Verlet check rebuilds the list for real); the next call then captures again
against the new adjacency (the key is the identity of the Verlet list).  Whenever no flag is set, the eager checks would all have kept the prior list, so the replay is the eager result.

Requirements (else the eager step is used): `cfg.fluidWarp` and `cfg.fusedWall` (the Verlet list and the fused wall are the capturable pieces), `SurfaceRep` / `BoxRep` walls, no `wallViscosityForm = "pairwise"` (index lists).  Bodies may move (prescribed motion): they are part of the integrated state (system.py), so the graph takes their state at the start of the step as one device input (`sim._bodyIn` [3, B, 3]: centre / angle, velocity / omega, acceleration / alpha, filled from the host bodies before every replay), integrates them with the particles and returns the final state in `sim._bodyOut`, which the host bodies take over after the replay (one host read together with the validity flags and dt).  A time-dependent gravity (`gravityFn`) is fine: `g` is a static input updated every call.
"""
import contextlib
import gc

import numpy as np
import torch
import warp as wp
from warpSPHCore import ParticleState, deferVerletChecks

from ..scene.scene import BoxRep, SurfaceRep

F64 = torch.float64


@contextlib.contextmanager
def _no_gc():
    """no garbage collection during a capture: finalising a dead warp stream mid-capture makes an illegal CUDA call (warpSPH `_captureGuard`)."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def graphable(sim):
    cfg = sim.cfg
    if not (cfg.fluidWarp and cfg.fusedWall and cfg.fixedAdjacency) or sim.scene is None:
        return False
    if cfg.wallViscosity and cfg.viscosity and cfg.wallViscosityForm == "pairwise":
        return False
    for b in sim.scene.bodies:
        if any(not isinstance(r, (SurfaceRep, BoxRep)) for r in b.reps):
            return False
    return torch.cuda.is_available() and str(sim.dev).startswith("cuda")


class GraphedStep:
    def __init__(self, sim):
        self.sim = sim
        self.graph = None
        self.key = None
        self.stats = dict(captures=0, replays=0, eager=0)
        self.carry_ref, self.carry_pose = None, None

    def _pose_id(self):
        """what identifies the bodies as the last replay left them: the objects (device tensors) and values (host floats) the write-back set."""
        return [(b.center, b.linearVelocity, b.angle, b.angularVelocity) for b in self.sim.scene.bodies]

    def _same_pose(self):
        old, new = self.carry_pose, self._pose_id()
        return old is not None and len(old) == len(new) and all(a[0] is b[0] and a[1] is b[1] and a[2] == b[2] and a[3] == b[3] for a, b in zip(old, new))

    def _refresh_carry(self):
        """eagerly evaluate the wall at the current positions and poses and write it into the carry buffers."""
        sim = self.sim
        sim._graphMode = False
        ps = ParticleState(positions=sim.x, supports=sim.Hvec, masses=torch.full_like(sim.rho, sim.m), kinds=sim.kinds, densities=sim.rho)
        sim._carry.copy_from(sim._fused_state(ps))

    def _key(self):
        fw = self.sim._fluidwarp
        return (id(fw._prior) if fw is not None and fw._prior is not None else None, tuple(self.sim.x.shape), self.sim._fused_key())

    def _capture(self):
        sim = self.sim
        saved = (sim.x, sim.v, sim.rho, sim.g, sim.dt_t.clone(), sim.surface, sim.surfaceDilated)
        bsaved = [(b.center, b.linearVelocity, b.linearAcceleration, b.angle, b.angularVelocity, b.angularAcceleration, getattr(b, "_cs", None)) for b in sim.scene.bodies]
        sim._graphMode = True
        sim._carry, sim._carryNext, sim._carryEnabled = None, False, sim.cfg.noPen == "impulse" and sim.scene is not None
        sim._body_pack()                                                       # warm-up / capture see the bodies as they are now (the values are replaced before every replay)
        try:
            sim.x, sim.v, sim.rho = saved[0].clone(), saved[1].clone(), saved[2].clone()
            sim._step_core()                                                  # warm-up: module loads, per-position-set caches, the first Verlet list; its result is discarded
            sim.x, sim.v, sim.rho, sim.g = saved[0].clone(), saved[1].clone(), saved[2].clone(), saved[3].clone()
            sim.dt_t.copy_(saved[4])
            if sim._carry is not None:
                sim._carry.reset_derived()                                       # adopted from the warm-up: its lazily built arrays belong to the warm-up, the capture rebuilds them
            self.sx, self.sv, self.srho, self.sg = sim.x, sim.v, sim.rho, sim.g
            stream = torch.cuda.Stream()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with _no_gc(), deferVerletChecks() as flags, wp.ScopedStream(wp.stream_from_torch(stream)), torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
                forces, nopen = sim._step_core()
            self.out = (sim.x, sim.v, sim.rho, forces, nopen, sim._bodyOut)
            self.flags = list(flags)
            self.graph = graph
            self.key = self._key()
            self.carry_ref = None                                                # the carry is refreshed before the first replay
            self.stats["captures"] += 1
        finally:
            sim.x, sim.v, sim.rho, sim.g = saved[0], saved[1], saved[2], saved[3]
            sim.dt_t.copy_(saved[4])
            sim.surface, sim.surfaceDilated = saved[5], saved[6]
            sim._graphMode = False
            for b, st in zip(sim.scene.bodies, bsaved):
                b.center, b.linearVelocity, b.linearAcceleration, b.angle, b.angularVelocity, b.angularAcceleration, b._cs = st

    def step(self):
        sim = self.sim
        sim.dt_t.fill_(sim.dt)
        dt = sim.dt
        if self.graph is None or self.key != self._key():
            self._capture()
        sim._body_pack()
        if sim._carry is not None:
            if self.carry_ref is not sim.x or not self._same_pose():           # state changed since the last replay (first step, eager fallback, a caller): rebuild the carry at the current positions
                self._refresh_carry()
        sim._graphMode = True
        try:
            self.sx.copy_(sim.x)
            self.sv.copy_(sim.v)
            self.srho.copy_(sim.rho)
            self.sg.copy_(sim.g)
            self.graph.replay()
        finally:
            sim._graphMode = False
        bad = torch.stack(self.flags).any().to(F64) if self.flags else torch.zeros((), dtype=F64, device=sim.dev)
        host = torch.cat([bad.reshape(1), sim.dt_t.reshape(1), self.out[5].reshape(-1)] if sim.nb else [bad.reshape(1), sim.dt_t.reshape(1)]).tolist()               # the one host read of the step
        bad_h, dt_h = host[0], host[1]
        if bad_h:                                                             # the Verlet list is no longer valid: this step eagerly (rebuilds), the next one captures against the new list
            self.stats["eager"] += 1
            sim.dt_t.fill_(dt)
            sim._graphMode = False
            return sim._step_eager()
        self.stats["replays"] += 1
        if sim.nb:
            sim._bodyOut = self.out[5]
            sim._body_writeback(host[2:])
        sim.x, sim.v, sim.rho = self.out[0].clone(), self.out[1].clone(), self.out[2].clone()
        self.carry_ref = sim.x
        self.carry_pose = self._pose_id()
        forces, nopen = self.out[3], self.out[4]
        if forces is not None:
            sim.wallLoads = forces.clone()
            sim.wallForce = sim.wallLoads[0, :, :2]
        sim.nopen_count = nopen.clone() if isinstance(nopen, torch.Tensor) else nopen
        sim.time += dt
        if sim.gravityFn is not None:
            sim.g = torch.tensor(sim.gravityFn(sim.time), dtype=F64, device=sim.dev)
        sim.dt = dt_h
        return sim.time
