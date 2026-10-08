"""Boundary-condition policies as wall-velocity closures (docs/audit-warpsph-boundary-hooks.md s.5, plan "Remaining items" 4).

warpSPH writes a policy (`BCType`) into the velocity of the wall's ghost particles (`modules/mdbc/velocity.py`); the analytic boundary has no ghost particles, so the same policy is a function that returns
the velocity of the wall continuum seen by a fluid particle.  With `w = v - u_body` the fluid's velocity relative to the wall and `n` the wall normal:

    noSlip    u_g = u_body - w_t            (the tangential slip is reversed about the wall; the normal part is dropped: normal component u_body . n)
    freeSlip  u_g = u_body + w_t - w_n      (the published form: the normal part is reflected, the tangential part kept)
    zeros     u_g = 0                       (pinned: the wall velocity is zero whatever the body does)
    constant  u_g = value                   (pinned: a prescribed constant wall velocity; `None` = the body's velocity)

`extended` (MLS extrapolation, open boundaries) is not supported (dropped, audit s.4).  The relative velocity of the fluid to the wall in the viscous, continuity and no-penetration terms is `v - u_g`: for
`zeros` / `constant` the pinned value replaces the rigid-body field `Body.velocityAt` in all of them (`DeltaSPH2D._bvel`, the contact-point velocity and the relative no-penetration law); `noSlip` and `freeSlip`
are the closures the wall viscosity forms already implement analytically (`Body.bc` selects per body: noSlip = the configured `noslip*` form, freeSlip = the exact wall Laplacian with the symmetric mirror,
`wallViscosityForm = 'laplacian'`), so for them `wallVelocity` is the reference definition the analytic forms are tested against (tests/sim/test_bc_closures.py).
"""
import torch

POLICIES = ("noSlip", "freeSlip", "zeros", "constant")


def wallVelocity(policy, v, normal, bodyVelocity, value=None):
    """the velocity of the wall continuum `u_g` [..., 2] for fluid velocity `v`, unit `normal` and the wall's own rigid-body velocity `bodyVelocity` (all broadcastable to [..., 2]); `value` the pinned vector
    of `constant`.  Elementwise, no host synchronisation."""
    if policy not in POLICIES:
        raise ValueError("BC policy must be one of %s, got %r" % (POLICIES, policy))
    if policy == "zeros":
        return torch.zeros_like(v + bodyVelocity)
    if policy == "constant":
        return (bodyVelocity if value is None else torch.as_tensor(value, dtype=v.dtype, device=v.device)).expand_as(v + bodyVelocity).clone()
    w = v - bodyVelocity
    wn = (w * normal).sum(-1, keepdim=True) * normal
    wt = w - wn
    if policy == "noSlip":
        return bodyVelocity - wt
    return bodyVelocity + wt - wn


def pinned(policy):
    """True where the policy replaces the rigid-body velocity of the wall (`zeros`, `constant`) in the relative-velocity terms."""
    return policy in ("zeros", "constant")
