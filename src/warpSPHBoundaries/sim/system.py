"""`DeltaSPH2D` as a warpSPHIntegrators system (phase 4 of HANDOFF A4, docs/plan-wall-evaluation.md): the step is `symplecticEuler(system, dt, f)` of the library instead of the hand-coded predictor-corrector.

    state     `DeltaSPHState`: positions, velocities, densities (a DRIFT field, `drift_key='drhodt_kin'`: the time-centred continuity `cfg.timeCentred` is the library's drift-field mechanism),
              and the BODIES: `bodyPositions` [B, 3] (centre, angle) and `bodyVelocities` [B, 3] (linear, angular) are integrated fields like the particles, `bodyAccelerations` [B, 3] the
              prescribed motion (constant per step; a coupled or time-dependent motion replaces `dbvdt` in `deltaSPHRhs`).  The stage states of the integrator carry the stage poses: the scene
              sees the half-step pose in the second right-hand side and the final pose in `finalize`, no `Body.move` inside the step.
    f         `deltaSPHRhs(system, dt)` = `DeltaSPH2D.rhs` at the stage state -> (update, aux); aux holds the kinematic-rate closure of the stage (`drift_rates`), the accelerations (time step) and the loads.
    finalize  particle shifting and the no-penetration impulse on the final state (the system's `finalize` hook, as warpSPH's `WeaklyCompressibleSystem`), at the final body poses.

The library's symplectic Euler is the scheme of the old `_step_core` term by term (half step with k0, `v += dt k1`, `x += dt/2 v0 + dt/2 v^{n+1}`, `rho += dt k1 + dt (K(vbar) - kin(k1))`); bodies advance as `x_b += dt/2 v_b^n + dt/2 v_b^{n+1}`, `v_b += dt a_b`, which is exact for a constant acceleration (the old two explicit `Body.move(dt/2)` were first order in the acceleration at the end of the step; the half-step pose and all velocities are unchanged).
"""
from dataclasses import dataclass
from typing import Optional

import torch

from warpSPHIntegrators import (BaseIntegrationSystem, BaseState, ComponentUpdateSpec, PositionUpdateSpec, integrated, constant, reference_state, tagged, update_component, update_position)

F64 = torch.float64


@dataclass
class DeltaSPHState(BaseState):
    positions: torch.Tensor = integrated("dxdt", tags=("position",))
    velocities: torch.Tensor = integrated("dvdt", tags=("velocity",))
    densities: torch.Tensor = integrated("drhodt", tags=("density",), drift_key="drhodt_kin")
    bodyPositions: torch.Tensor = integrated("dbxdt", tags=("body_position",))        # [B, 3]: centre x, y, angle
    bodyVelocities: torch.Tensor = integrated("dbvdt", tags=("body_velocity",))       # [B, 3]: linear x, y, angular
    bodyAccelerations: torch.Tensor = constant(tags=("body_acceleration",))           # [B, 3]: prescribed


@dataclass
class DeltaSPHUpdate:
    dxdt: torch.Tensor = tagged(tags=("position_derivative",))
    dvdt: torch.Tensor = tagged(tags=("velocity_derivative",))
    drhodt: torch.Tensor = tagged(tags=("density_derivative",))
    dbxdt: torch.Tensor = tagged(tags=("body_position_derivative",))
    dbvdt: torch.Tensor = tagged(tags=("body_velocity_derivative",))
    drhodt_kin: Optional[torch.Tensor] = tagged(default=None)             # the velocity-linear part of drhodt (the continuity equation), the drift field's rate


@dataclass
class DeltaSPHSystem(BaseIntegrationSystem):
    state: DeltaSPHState = reference_state(tags=("physics_state",))
    sim: object = None
    t: float = 0.0

    # ------------------------------------------------------------------------------------------------ construction and the scene binding
    @staticmethod
    def of(sim, bodies):
        """the system at the solver's current state; `bodies` [3, B, 3] = (positions, velocities, accelerations) of the bodies (`DeltaSPH2D._body_pack`)."""
        return DeltaSPHSystem(DeltaSPHState(sim.x, sim.v, sim.rho, bodies[0], bodies[1], bodies[2]), sim, float(sim.time))

    def initializeNewState(self, *args, **kwargs):
        return DeltaSPHSystem(self.state.initializeNewState(), self.sim, self.t)

    def bind(self):
        """the scene's bodies at the pose, velocity and acceleration of this state (device tensors: a captured step reads them from its static inputs)."""
        st = self.state
        cs = torch.stack([torch.cos(st.bodyPositions[:, 2]), torch.sin(st.bodyPositions[:, 2])]) if len(self.sim.scene.bodies) else None
        for i, b in enumerate(self.sim.scene.bodies):
            bp, bv, ba = st.bodyPositions[i], st.bodyVelocities[i], st.bodyAccelerations[i]
            b.center, b.angle, b._cs = bp[0:2], bp[2], (cs[0, i], cs[1, i])
            b.linearVelocity, b.angularVelocity = bv[0:2], bv[2]
            b.linearAcceleration, b.angularAcceleration = ba[0:2], ba[2]

    # ------------------------------------------------------------------------------------------------ lifecycle hooks
    def initialize(self, dt, *args, **kwargs):
        self.sim._stage = 0
        return self

    def preprocess(self, initialState, dt, *args, **kwargs):
        if self.sim.scene is not None:
            self.bind()
        return self

    def finalize(self, initialState, dt, returnValues, updateValues, weights=..., *args, **kwargs):
        """shifting, then the no-penetration impulse, on the final state at the final body poses (the order of the old step); `dt` is the full step."""
        sim, st = self.sim, self.state
        if sim.scene is not None:
            self.bind()
        sim.x, sim.v, sim.rho = st.positions, st.velocities, st.densities
        if sim.cfg.shifting:
            sim.x = sim.x + sim.shift(dt)
        nopen = sim.no_penetration()
        sim.apply_pinned()
        st.positions, st.velocities = sim.x, sim.v
        auxs = [r[0] for r in returnValues]                                                  # one aux per stage (acc, forces, kinematic); the last stage's acceleration sets the next time step
        aux = auxs[-1]
        forces = None
        if len(weights) == len(auxs):                                                        # the stage loads weighted as the scheme weights its stages ([0, 1] for the symplectic Euler: the second only, bit for bit)
            for w, a in zip(weights, auxs):
                if a["forces"] is not None and float(w) != 0.0:
                    forces = float(w) * a["forces"] if forces is None else forces + float(w) * a["forces"]
        if forces is None:
            forces = aux["forces"]
        sim._finalAux = dict(acc=aux["acc"], forces=forces, nopen=nopen, nopenLoad=sim._nopenLoad)
        sim.dt_t.copy_(sim._next_dt(aux["acc"]))                                              # the adaptive time step of the next step (a hook of the system; in place: the device scalar is a persistent buffer)
        return self

    # ------------------------------------------------------------------------------------------------ typed update interface
    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        update_position(self, update, spec, "position", "position_derivative", "velocity", "velocity_derivative")
        return update_position(self, update, spec, "body_position", "body_position_derivative", "body_velocity", "body_velocity_derivative")

    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        update_component(self, update, spec, "velocity", "velocity_derivative")
        return update_component(self, update, spec, "body_velocity", "body_velocity_derivative")

    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, "density", "density_derivative")

    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        self.apply_position_update(update, PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend), **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self

    # ------------------------------------------------------------------------------------------------ drift field
    def drift_fields_enabled(self, **kwargs):
        """the time-centred continuity (`cfg.timeCentred`); ALWAYS for the Verlet pair: their end-of-step evaluation sits at the drifted positions with the half-step velocity, and without the density drifted
        with them the equation of state sees the old density there (measured, hydrostatic tank, 0.3 s: rho in [0.79, 1.32] and v_max 12 m/s for Velocity Verlet, [0.98, 1.28] for Leap Frog, against
        [0.9998, 1.0027] and 0.40 with the drift field)."""
        return bool(self.sim.cfg.timeCentred or self.sim.cfg.integrator in ("Velocity Verlet", "Leap Frog"))

    def drift_rates(self, velocities, aux=None, **kwargs):
        """the continuity rate at this state's configuration with the fluid moving at `velocities`: the closure of the stage's right-hand side (the same positions, densities, wall and bodies)."""
        if self.sim.scene is not None:
            self.bind()
        kinematic = aux[0]["kinematic"]
        return DeltaSPHUpdate(dxdt=None, dvdt=None, drhodt=None, dbxdt=None, dbvdt=None, drhodt_kin=kinematic(velocities))


def deltaSPHRhs(system, dt, *args, **kwargs):
    """the integrator's right-hand side at a stage state: `DeltaSPH2D.rhs` (the wall loads are booked on the second stage only, the one that drives the step)."""
    sim, st = system.sim, system.state
    stage = sim._stage
    sim._stage += 1
    acc, drho, forces = sim.rhs(st.positions, st.velocities, st.densities, want_forces=sim._wantForces(stage))
    update = DeltaSPHUpdate(dxdt=st.velocities, dvdt=acc, drhodt=drho, dbxdt=st.bodyVelocities, dbvdt=st.bodyAccelerations, drhodt_kin=sim._kin)
    return update, dict(kinematic=sim._kinematic, acc=acc, forces=forces)
