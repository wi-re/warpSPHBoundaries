# δ⁺-SPH on analytic walls — where things stand (resume note, 2026-10-02)

Read first: `deltasph-plan.md` (strategy, §3b port + 3D), `deltasph-validation.md` (results), `deltasph-porting-notes.md` (term map warpSPH ↔ here, §4b ported / not ported, change log). This file is the short way back in.

## State
All code and docs are committed (last code commit 30fae1e, docs after). Solver `src/edgebound/sim/deltasph2d.py` (`DeltaSPH2D`; cases `hydrostatic_tank`, `english_wedge`, `marrone_dambreak`, `sloshing_tank`), scoring / runners
`deltasph_validation.py`, snapshots `deltasph_snap.py`, videos `dfsph_video.py` (also used for DFSPH), comparison figure `deltasph_compare.py`, tests `tests/sim/test_deltasph.py` (10) + `test_scene.py` (50) + `test_dfsph.py` (12): all passing at the last run
(72 on the three files). Scene additions: `Scene.inside`, `Scene.signed_distance`; shared fix in `dfsph2d.neighbor_pairs` (cell 1.01 H, dense path N ≤ 8000).

| case | against | result |
|---|---|---|
| flat tank (English §4.1) | warpSPH + mDBC | profile RMSE 0.0019 vs 0.0225, KE 1.3e-7 vs 1.1e-6 (surface = volume = SDF) |
| English wedge | warpSPH + mDBC | all probe checks pass, profile better than mDBC; settled KE 3.8e-4 vs 1e-6 (layout inconsistency at a smooth slope: open, packing deferred) |
| Marrone 3.1 dam break | warpSPH sun2017DeltaSPH | P1 arrival 2.49 vs 2.48, P1 curve overlaps, KE within 3 %; P2 / ceiling probe qualitative |
| SPHERIC TC10 sloshing | warpSPH + measurement | flow (KE) and impact times match warpSPH; impact peaks same order; wall MLS probe unusable |

## Reproduce
```
python scripts/deltasph_validation.py tank 0.02 4.0 surface            # ~5 min
python scripts/deltasph_validation.py wedge 0.02 4.0 surface           # ~8-13 min
python scripts/deltasph_validation.py dambreak 67 1.9 out.npz shifting=True noPen=impulse   # ~20 min, snapshots in out.npz
python scripts/deltasph_validation.py sloshing 200 7.0 out.npz shifting=True noPen=impulse  # ~90 min
python scripts/dfsph_video.py out.mp4 "label=out.npz" --vmax=5             # reads the snap_* keys of the runners directly
```
Reference runs (warpSPH, branch `dev`, never modified; its own rules: one GPU run at a time): `python scripts/probe_deltaSPHMarrone.py --nx 67 --c0Ratio 40 --out <dir>`;
`python scripts/probe_englishWedge.py --dp 0.02 [--wedge|--no-wedge] --tLimit 4`; `python examples/sloshingTank/run_sloshingTank.py --scheme wcsph --tLimit 7 --no-video --out <dir>`.
Initial particles and interior boxes of the warpSPH cases were dumped with `results/deltasph/dump_marrone.py` / `dump_slosh.py` (they import warpSPH).

## Artefacts
* tracked, small: `results/deltasph/` (series of the dam breaks A (no PST / noPen) and B, the sloshing run, the warpSPH references, the comparison figures, the wedge-ramp reproducer `ramp_build.py`, the dump scripts).
* **not tracked** (`.tmp/`, ignored): the snapshot files and the videos in `.tmp/delta/` (`dambreak_marrone_analytic.mp4`, `sloshing_analytic.mp4`, `wedge_ours.mp4`, `final/` wedge reference video, `ref/` Marrone reference video and frames, `slosh_ref/`). Regenerate with the commands above if needed.

## Open items (ranked)
1. **Viscous wall term as an exact Laplacian operation** (user correction): `ν_eff (v_b − v_i) Δλ` with `Δλ = ∫∇²W dA` (tier 3: second derivative of the planar table; tier 2: edge integral of `∂_n W`), first moments of `∇²W` for a linear mirror field. Replace the polar quadrature, calibrate `ν_eff` against the pair form (`α c0 H/(ξ 2(d+2))`, unverified), check the effect on the dam break (expected small).
2. **Wall-pressure sensor for analytic walls** (the first-order MLS probe at a wall point is erratic at impacts); take it from the wall model (`P_b`, or force per length).
3. **Shifting wall term exactly**: the tensile control `∫W⁴∇W dA` is the gradient of the kernel `W⁵/5` (an exact tier integral with a new kernel table entry, degree-25 polynomial for C2: Chebyshev compile). Michel-2022 shifting (warpSPH's sloshing default) is not ported.
4. Wedge KE: general body-fitted packing (deferred by the user: airfoils, complex geometry).
5. Moving bodies in δ⁺ (no-penetration law and viscous term assume static walls), exact force bookkeeping (viscous / shifting / no-penetration not booked), curved-wall mirror normals (per-query vector field).
6. Studies: dam break at H/dx = 80 / 320, sloshing nx = 100 / 400, no-shift and Michel variants, tiers 1 and 3 on the dam break, the hexagon in δ⁺.
7. **Port to warpSPH** (plan §3b): dimension-generic interface, batched primitive instances (fibre bundles: a bundle of disks in cross-flow as the first demo), fixed-capacity reusable adjacency, boundary-provider integration, regression suite.

## To investigate: how openMaelstrom does these things (user, 2026-10-02)
`~/dev/openMaelstrom` (not read yet). Reported by the user: its surface detection is **cover-vector based and works with an SDF boundary**, and it **supports boundary friction terms**. Pointers found by grep (unverified, to be read):
`SPH/surface/surfaceDetection.cu(h)` (surface detection), `SPH/boundary/volumeBoundary.cu(h)` (volume / SDF boundaries), `SPH/DFSPH/dfsph.cu` (`boundaryFrictionKernel` l.452, `volumeFrictionKernel` l.491), `SPH/IISPH17/iisph17.cu`, `SPH/integration/simple.cu` (shifting?), `SPH/convection/AkinciTension.cu`.
Questions: (a) how the cover vector and the empty-cone test are evaluated against an SDF / volume boundary instead of boundary particles (and whether it is an analytic or a sampled evaluation: our Barecasco wall counts are polar sampling of the solid); (b) the friction term: formulation (per boundary, per volume), and whether it can use `λ` / `∇λ` and the SDF directly, replacing our sampled viscous wall term; (c) how shifting treats the SDF boundary (what, if anything, replaces the wall part of the tensile control).
Expected consequence (user, to be confirmed by the code): with an SDF-based cover vector and a friction term that needs only SDF quantities, **shifting is the only remaining term whose wall part has no existing exact weight**.
Our own accounting of the sampled wall terms: viscous wall term (→ exact Laplacian operation, item 1), detector wall counts (cover vector = gradient of the kernel `K = r`, exact; only the cone count is a clipping problem, and only `> 0.5` matters), shifting tensile control (`W⁵/5`, item 3, exact in principle with a new kernel table entry rather than quadrature, conditioning to be checked).
