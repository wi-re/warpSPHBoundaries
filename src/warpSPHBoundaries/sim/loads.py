"""Drag and lift of a body in a flow: booking the loads of a run, recovering them from exported frames, and the coefficients.

`LoadHistory.record(sim)` books, after every step, the load of the fluid on every body from the solver (`wallLoads`: pressure, wall viscous, no-penetration impulse; Fx, Fy, torque about the body centre; 2D force per unit
depth).  `loads_from_frames(sim, frames, times)` recovers pressure + viscous loads from exported states (x, v, rho) of consecutive frames only (the wall terms are a pure function of the state and the body pose,
`DeltaSPH2D.loadsAt`; the booked load is time-centred at the half step, so the mean of the two frames of the interval is used; the no-penetration impulse is not recoverable and is zero unless particles press into the
wall).  The positions of the frames may be raw, unwrapped trajectories of a periodic run.  `coefficients` rotates the force into the stream direction (drag) and across it (lift, +90 degrees) and normalises by
1/2 rho U^2 D; `strouhal` is the dominant frequency of a signal (FFT, parabolic peak interpolation) times D / U.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class LoadHistory:
    t: list = field(default_factory=list)
    loads: list = field(default_factory=list)                 # per step [3 terms, B, 3]

    def record(self, sim):
        """book the loads of the step just taken (time = the end of the step)."""
        self.t.append(float(sim.time))
        self.loads.append(sim.wallLoads.detach().cpu().numpy().copy())
        return self

    def total(self, body=0):
        """[n, 3]: (Fx, Fy, torque) of `body`, the three terms summed."""
        return np.array(self.loads)[:, :, body, :].sum(1)

    def series(self, body=0):
        return np.array(self.t), self.total(body)


def loads_from_frames(sim, frames, times=None, body=0):
    """(t_mid [n-1], loads [n-1, 3]) of the intervals between consecutive frames: the mean of `sim.loadsAt` (pressure + wall viscous) at the two frames of each interval.  `frames`: (x, v, rho) tuples; the bodies of
    `sim.scene` stay at their current pose (a moving body needs its pose per frame: set it on the scene before each call)."""
    L = [sim.loadsAt(*f)[:, body, :].sum(0).detach().cpu().numpy() for f in frames]
    out = 0.5 * (np.array(L[:-1]) + np.array(L[1:]))
    t = None if times is None else 0.5 * (np.asarray(times)[:-1] + np.asarray(times)[1:])
    return t, out


def coefficients(loads, rho, U, D, direction=(1.0, 0.0)):
    """(C_D, C_L) of force rows [n, >=2]: the component along `direction` (the stream) and the one 90 degrees counter-clockwise from it, over 1/2 rho U^2 D."""
    e = np.asarray(direction, float)
    e = e / np.linalg.norm(e)
    n = np.array([-e[1], e[0]])
    q = 0.5 * rho * U * U * D
    return (loads[:, :2] @ e) / q, (loads[:, :2] @ n) / q


def window_mean(t, y, t0, t1=None):
    t = np.asarray(t)
    m = (t >= t0) & (True if t1 is None else t <= t1)
    return float(np.mean(np.asarray(y)[m]))


def strouhal(t, y, D, U, t0=0.0):
    """D f / U with f the dominant frequency of the signal y(t) for t >= t0 (uniform sampling assumed): mean removed, Hann window, FFT peak with parabolic interpolation."""
    t, y = np.asarray(t), np.asarray(y)
    m = t >= t0
    t, y = t[m], y[m]
    dt = float(np.mean(np.diff(t)))
    y = (y - y.mean()) * np.hanning(len(y))
    n = 8 * len(y)
    sp = np.abs(np.fft.rfft(y, n))
    k = int(np.argmax(sp[1:])) + 1
    if 1 <= k < len(sp) - 1:
        a, b, c = np.log(sp[k - 1] + 1e-300), np.log(sp[k] + 1e-300), np.log(sp[k + 1] + 1e-300)
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    f = k / (n * dt)
    return f * D / U
