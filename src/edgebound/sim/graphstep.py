"""CUDA-graph replay of the whole `DeltaSPH2D` step (docs/plan-wall-evaluation.md step 5b), after warpSPH's `utils/cudaGraph.py` (`GraphedStateFunction`, `deferVerletChecks`).

The eager step is CPU-bound (about 1200 small kernels, ~14 warp / torch module calls whose host cost dwarfs their GPU time).  `_step_core` has no host read and no data-dependent shape, so it is
captured once into a `torch.cuda.CUDAGraph` on a side stream shared with warp (`wp.stream_from_torch`), against private copies of the state (x, v, rho, g); `dt` lives in the persistent device buffer `dt_t`.
Every call: the host value of `dt` is written into `dt_t`, the state is copied into the static inputs, the graph is replayed, and one device-to-host read brings back (a) the OR of the deferred Verlet
validity flags and (b) the new dt.  If any flag is set the replayed result is discarded and the step is run eagerly (where the Verlet check rebuilds the list for real); the next call then captures again
against the new adjacency (the key is the identity of the Verlet list).  Whenever no flag is set, the eager checks would all have kept the prior list, so the replay is the eager result.

Requirements (else the eager step is used): `cfg.fluidWarp` and `cfg.fusedWall` (the Verlet list and the fused wall are the capturable pieces), `SurfaceRep` / `BoxRep` walls, no `wallViscosityForm = "pairwise"` (index lists).  Bodies may move (prescribed motion): the poses of the three stages of a step (before the first RHS, after the first half move, after the second) are computed on the host with the arithmetic of `Body.move` and fed to the graph as device inputs (`stage` [3, B, 8]: centre, velocity, cos, sin, omega, alpha); the bodies are bound to them while the step is captured.  A time-dependent gravity (`gravityFn`) is fine: `g` is a static input updated every call.
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
        nb = len(sim.scene.bodies)
        self.stage = torch.zeros((3, nb, 8), dtype=F64, device=sim.dev)         # per stage and body: cx, cy, vx, vy, cos, sin, omega, alpha

    def _bind(self, k):
        """bind the bodies to the device pose of stage k (called from `_step_core` while the graph is captured / warmed up)."""
        for bi, b in enumerate(self.sim.scene.bodies):
            st = self.stage[k, bi]
            b.center, b.linearVelocity, b._cs = st[0:2], st[2:4], (st[4], st[5])
            b.angularVelocity, b.angularAcceleration = st[6], st[7]

    def _pack(self, advance):
        """fill `stage` from the host: stage 0 = the bodies as they are, stages 1 and 2 = after `Body.move(dt/2)` once and twice (same arithmetic; python floats and the device centre / velocity read back once)."""
        sim = self.sim
        bodies = sim.scene.bodies
        dev = torch.cat([torch.cat([b.center, b.linearVelocity]) for b in bodies]).tolist() if bodies else []
        h = 0.5 * sim.dt
        rows = []
        for bi, b in enumerate(bodies):
            cx, cy, vx, vy = dev[4 * bi:4 * bi + 4]
            ang, om, al = float(b.angle), float(b.angularVelocity), float(b.angularAcceleration)
            ax, ay = [float(a) for a in b.linearAcceleration.tolist()]
            st = []
            for k in range(3):
                st.append([cx, cy, vx, vy, float(np.cos(ang)), float(np.sin(ang)), om, al])
                if advance:                                                   # Body.move: centre with the old velocity, angle with the old omega, then the velocities
                    cx, cy = cx + h * vx, cy + h * vy
                    ang = ang + h * om
                    vx, vy = vx + h * ax, vy + h * ay
                    om = om + h * al
            rows.append(st)
        if rows:
            self.stage.copy_(torch.tensor(rows, dtype=F64).permute(1, 0, 2))
        return rows

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
        bsaved = [(b.center, b.linearVelocity, b.angularVelocity, b.angularAcceleration, getattr(b, "_cs", None)) for b in sim.scene.bodies]
        sim._graphMode = True
        sim._graphBind = self._bind
        sim._carry, sim._carryNext, sim._carryEnabled = None, False, sim.cfg.noPen == "impulse" and sim.scene is not None
        self._pack(advance=False)                                              # warm-up / capture see the bodies at their current pose in every stage (values are replaced before every replay)
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
            self.out = (sim.x, sim.v, sim.rho, forces, nopen)
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
            sim._graphBind = None
            for b, st in zip(sim.scene.bodies, bsaved):
                b.center, b.linearVelocity, b.angularVelocity, b.angularAcceleration, b._cs = st

    def step(self):
        sim = self.sim
        sim.dt_t.fill_(sim.dt)
        dt = sim.dt
        if self.graph is None or self.key != self._key():
            self._capture()
        rows = self._pack(advance=True)
        if sim._carry is not None:
            now0 = [r[0][:6] for r in rows]
            if self.carry_ref is not sim.x or self.carry_pose != now0:           # state changed since the last replay (first step, eager fallback, a caller): rebuild the carry at the current positions
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
        bad_h, dt_h = torch.stack([bad, sim.dt_t[()]]).tolist()               # the one host read of the step
        if bad_h:                                                             # the Verlet list is no longer valid: this step eagerly (rebuilds), the next one captures against the new list
            self.stats["eager"] += 1
            sim.dt_t.fill_(dt)
            sim._graphMode = False
            return sim._step_eager()
        self.stats["replays"] += 1
        for b in sim.scene.bodies:                                            # the bodies end the step where the eager step leaves them (two half moves)
            b.move(0.5 * dt)
            b.move(0.5 * dt)
        sim.x, sim.v, sim.rho = self.out[0].clone(), self.out[1].clone(), self.out[2].clone()
        self.carry_ref = sim.x
        self.carry_pose = [r[2][:6] for r in rows]
        forces, nopen = self.out[3], self.out[4]
        if forces is not None:
            sim.wallForce = forces.clone()
        sim.nopen_count = nopen.clone() if isinstance(nopen, torch.Tensor) else nopen
        sim.time += dt
        if sim.gravityFn is not None:
            sim.g = torch.tensor(sim.gravityFn(sim.time), dtype=F64, device=sim.dev)
        sim.dt = dt_h
        return sim.time
