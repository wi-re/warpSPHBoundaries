"""Dam-break comparison figure and tables: python scripts/deltasph_compare.py out.png \"label=run.npz\" ...  against warpSPH's Marrone output (path below, run from the repo root).
   Run files from `python scripts/deltasph_validation.py dambreak`; the reference `probe_deltaSPHMarrone.py --nx 67 --c0Ratio 40 --out .tmp/delta/ref`."""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from edgebound import paths
ref = np.load(paths.tmp_dir() / "delta" / "ref" / "sun2017DeltaSPH_nx67_c40.npz", allow_pickle=True)
runs = [(a.split("=")[0], np.load(a.split("=")[1])) for a in sys.argv[2:]]
fig, ax = plt.subplots(3, 2, figsize=(13, 10))
tr = ref["tStar"]
for k, (name, key) in enumerate((("P1 (z=0.16)", 0), ("P2 (z=0.584)", 1), ("P3 (z=1.0, ceiling)", 2))):
    a = ax[k, 0]
    a.plot(tr, ref[f"pProbe{key}Star"], "k", lw=0.8, label="warpSPH (PST + noPen, mDBC)")
    for lab, r in runs:
        a.plot(r["tStar"], r[f"pProbe{key}Star"], lw=0.8, label=lab)
    a.set_ylabel("P*"); a.set_title(name); a.set_xlim(2, 7.7)
    if key == 2: a.set_yscale("symlog", linthresh=1); 
    else: a.set_ylim(-0.05, 1.6)
    if k == 0: a.legend(fontsize=8)
ax[0, 1].plot(tr, ref["kineticEnergy"], "k", lw=0.8); ax[0, 1].set_title("kinetic energy"); 
for lab, r in runs: ax[0, 1].plot(r["tStar"], r["kineticEnergy"], lw=0.8)
ax[1, 1].plot(tr, ref["maxVelocity"], "k", lw=0.8); ax[1, 1].set_title("max |v|"); ax[1, 1].set_ylim(0, 40)
for lab, r in runs: ax[1, 1].plot(r["tStar"], r["maxVelocity"], lw=0.8)
ax[2, 1].plot(tr, ref["minDensity"], "k", lw=0.8); ax[2, 1].plot(tr, ref["maxDensity"], "k", lw=0.8); ax[2, 1].set_title("density range"); ax[2, 1].set_ylim(0.5, 1.6)
for lab, r in runs:
    ax[2, 1].plot(r["tStar"], r["minDensity"], lw=0.8); ax[2, 1].plot(r["tStar"], r["maxDensity"], lw=0.8)
for a in ax.ravel(): a.set_xlabel("t*") if a in ax[2] else None
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=110)
for q in range(2):
    print("P%d  t*: ref | ours  at t* = 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0" % (q + 1))
    for lab, r in runs:
        f = lambda d, t, k: float(np.interp(t, d["tStar"], d[k]))
        print("  ref ", " ".join("%.3f" % f(ref, t, f"pProbe{q}Star") for t in (3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7)))
        print("  %-4s" % lab, " ".join("%.3f" % f(r, t, f"pProbe{q}Star") for t in (3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7)))
print("KE at t*=2,3,4,5,6,7: ref ", " ".join("%.3f" % np.interp(t, tr, ref["kineticEnergy"]) for t in (2, 3, 4, 5, 6, 7)))
for lab, r in runs: print("                  %-4s" % lab, " ".join("%.3f" % np.interp(t, r["tStar"], r["kineticEnergy"]) for t in (2, 3, 4, 5, 6, 7)))
