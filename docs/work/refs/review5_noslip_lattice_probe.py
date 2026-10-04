# reviewer probe for the no-slip wall viscosity: operator consistency on Couette (u = a y, exact lap = 0) and Poiseuille (u = y (Y - y), exact lap = -2)
# above a flat wall y = 0 (solid y < 0, wall at rest).  Lattice dp = 1, H = 4 (C2, h/dx = 2), rho = 1, V = dp^2, fac = 1, nu_eff = fac/8.
import numpy as np
dp, H = 1.0, 4.0
c = 7.0
W   = lambda r: np.where(r < H, c/(np.pi*H*H)*(1-r/H)**4*(1+4*r/H), 0.0)
dW  = lambda r: np.where(r < H, -140.0*(r/H)*(1-r/H)**3/(np.pi*H**3), 0.0)
def lapW(r):
    q = r/H; sp = -20*q*(1-q)**3; spp = -20*(1-q)**3 + 60*q*(1-q)**2        # s' and s'' of s = (1-q)^4 (1+4q)
    return np.where(r < H, c/(np.pi*H**4)*(spp + sp/np.maximum(q, 1e-300)), 0.0)
nu = 1.0/8.0
# fluid lattice (x periodic-ish: wide), rows k = 0..K
K = 14; xs = (np.arange(-14, 15)+0.0)*dp; ys = (np.arange(K)+0.5)*dp
XX, YY = np.meshgrid(xs, ys, indexing="ij"); Xf, Yf = XX.ravel(), YY.ravel()
def bulk_pairwise(ux, i_pos, field):               # Monaghan pairwise (fac=1) over the fluid lattice, particle at i_pos, velocity field(x,y) -> (ux,uy)
    px, py = i_pos; vi = np.array(field(px, py)); d = np.stack([px - Xf, py - Yf], 1); r = np.hypot(d[:,0], d[:,1]); m = (r > 1e-12) & (r < H)
    vj = np.array(field(Xf, Yf)).T; vij = vi[None] - vj
    mu = (vij*d).sum(1)/(r*r + 1e-14*H*H)
    gW = dW(r)[:, None]*d/np.maximum(r, 1e-300)[:, None]
    return (dp*dp*(mu[:, None]*gW)*m[:, None]).sum(0)
# solid half-plane integrals by polar midpoint (own, fine grid)
def solid_polar(z, f, nr=600, nt=1200):               # int_solid f(r, theta) dA, solid y' < 0, particle at (0, z)
    r = (np.arange(nr)+0.5)/nr*H; th = (np.arange(nt)+0.5)/nt*2*np.pi; R, T = np.meshgrid(r, th, indexing="ij")
    Y = z + R*np.sin(T); sol = Y < 0
    return (f(R, T)*sol*R).sum()*(H/nr)*(2*np.pi/nt), R, T, sol
def lap_lambda(z):                                     # B(z)
    return solid_polar(z, lambda R, T: lapW(R))[0]
def M2(z):                                             # int_solid W'/r yhat (x) yhat dA  (pairwise coefficient tensor), yhat = unit(x_i - x') direction ... sign as in the solver (W' < 0)
    out = np.zeros((2, 2))
    for a in range(2):
        for b in range(2):
            fa = lambda R, T, a=a, b=b: dW(R)*[np.cos(T), np.sin(T)][a]*[np.cos(T), np.sin(T)][b]
            out[a, b] = solid_polar(z, fa)[0]
    return out
def ghost_lattice_total(i_pos, field, vwall=(0.0, 0.0)):       # truth: odd-extension ghost lattice (v_g = 2 v_w - v(mirror)) with the pairwise bulk operator
    px, py = i_pos; vi = np.array(field(px, py)); tot = np.zeros(2)
    Xg, Yg = Xf, -Yf
    vg = 2*np.array(vwall)[:, None] - np.array(field(Xf, Yf))
    d = np.stack([px - Xg, py - Yg], 1); r = np.hypot(d[:,0], d[:,1]); m = r < H
    vij = vi[None] - vg.T; mu = (vij*d).sum(1)/(r*r + 1e-14*H*H); gW = dW(r)[:, None]*d/np.maximum(r, 1e-300)[:, None]
    return (dp*dp*(mu[:, None]*gW)*m[:, None]).sum(0)
def run(name, field, exact_lap_x):
    print("\n%s: exact nu lap u_x = %.4f (nu = 1/8), rows z/dp = 0.5, 1.5, 2.5, 3.5" % (name, nu*exact_lap_x))
    print("  %-6s %-12s %-12s %-12s %-12s %-12s" % ("z", "bulk only", "+ ghost lat.", "+ pairwise", "+ Laplacian", "target"))
    for k in range(4):
        z = (k+0.5)*dp; pos = (0.0, z)
        # only valid if the truncated top (rows up to K) does not matter: support reaches z + H <= K*dp
        b = bulk_pairwise(None, pos, field)[0]
        gl = ghost_lattice_total(pos, field)[0]
        vrel = np.array(field(0.0, z))                                   # wall at rest
        pair = 2.0*(M2(z) @ vrel)[0]*(-1.0)**0                           # acc = fac wm/rho * 2 M2 v_rel  (no-slip: v_ij = 2 v_rel)
        lam = lap_lambda(z); lapacc = 2.0*nu*lam*(0.0 - vrel[0])         # acc = 2 nu_eff Delta-lambda (v_w - v_i)
        print("  %-6.2f %-12.5f %-12.5f %-12.5f %-12.5f %-12.5f" % (z, b, b+gl, b+pair, b+lapacc, nu*exact_lap_x))
a = 1.0
run("Couette  u = a y        ", lambda x, y: (a*np.asarray(y)*np.ones_like(np.asarray(x, dtype=float)), 0*np.asarray(y)*np.ones_like(np.asarray(x, dtype=float))), 0.0)
Y = 12.0
run("Poiseuille u = y (Y - y)", lambda x, y: (np.asarray(y)*(Y - np.asarray(y))*np.ones_like(np.asarray(x, dtype=float)), 0*np.asarray(y)*np.ones_like(np.asarray(x, dtype=float))), -2.0)

# ---- Chiron et al. 2019 form (Eq. 91-92): Morris bulk over the fluid lattice + no-slip flux term, both / gamma, nu = 1 (Morris has coefficient 1)
from scipy import integrate
def morris_bulk(pos, field):
    px, py = pos; vi = np.array(field(px, py)); d = np.stack([px - Xf, py - Yf], 1); r = np.hypot(d[:,0], d[:,1]); m = (r > 1e-12) & (r < H)
    vj = np.array(field(Xf, Yf)).T
    return (2.0*dp*dp*((vi[None]-vj)*(dW(r)/np.maximum(r, 1e-300))[:, None])*m[:, None]).sum(0)
def gamma_fluid(z):                                    # int_{y>0} W dA  (the renormalisation factor of the truncated fluid support)
    return 1.0 - solid_polar(z, lambda R, T: W(R))[0]
def flux_wall(z, vrel):                                # 2 (v_i - v_s)/d_n int W ds along the wall chord
    L = np.sqrt(max(H*H - z*z, 0.0))
    I = integrate.quad(lambda s: float(W(np.hypot(s, z))), -L, L, epsabs=1e-13, epsrel=1e-13)[0]
    return -2.0*vrel*I/z                                   # n outward of the fluid: (x-y).n = -d_n
def run_chiron(name, field, exact_lap_x):
    print("\n%s (Chiron/Morris form, nu = 1): exact lap u_x = %.3f   [ghost-lattice Morris truth in the last column]" % (name, exact_lap_x))
    print("  %-6s %-12s %-12s %-12s %-12s" % ("z", "Morris bulk", "bulk/gamma", "Chiron tot.", "ghost-lat. Morris"))
    for k in range(4):
        z = (k+0.5)*dp; pos = (0.0, z); g = gamma_fluid(z); vrel = np.array(field(0.0, z))
        b = morris_bulk(pos, field)[0]; w = flux_wall(z, vrel[0])
        # ghost-lattice Morris truth: odd-extension ghosts through the same Morris pair operator
        px, py = pos; vi = np.array(field(px, py)); d = np.stack([px - Xf, py + Yf], 1); r = np.hypot(d[:,0], d[:,1]); m = r < H
        vg = -np.array(field(Xf, Yf)).T
        gl = (2.0*dp*dp*((vi[None]-vg)*(dW(r)/np.maximum(r, 1e-300))[:, None])*m[:, None]).sum(0)[0]
        print("  %-6.2f %-12.5f %-12.5f %-12.5f %-12.5f" % (z, b, b/g, (b + w)/g, b + gl))
sh = lambda x, y: (a*np.asarray(y)*np.ones_like(np.asarray(x, dtype=float)), 0*np.asarray(y)*np.ones_like(np.asarray(x, dtype=float)))
po = lambda x, y: (np.asarray(y)*(Y - np.asarray(y))*np.ones_like(np.asarray(x, dtype=float)), 0*np.asarray(y)*np.ones_like(np.asarray(x, dtype=float)))
run_chiron("Couette   u = a y        ", sh, 0.0)
run_chiron("Poiseuille u = y (Y - y) ", po, -2.0)

# ---- the combination that fits our solver: pairwise (Monaghan) bulk, nu_eff = 1/8, + nu_eff * flux term, with / without 1/gamma
def run_mix(name, field, exact):
    print("\n%s (solver bulk = pairwise, nu_eff = 1/8): target nu lap u_x = %.4f" % (name, nu*exact))
    print("  %-6s %-13s %-15s %-15s %-15s" % ("z", "pairwise bulk", "+ nu*flux", "(+ nu*flux)/gam", "gamma"))
    for k in range(4):
        z = (k+0.5)*dp; pos = (0.0, z); g = gamma_fluid(z); vrel = np.array(field(0.0, z))
        b = bulk_pairwise(None, pos, field)[0]; w = nu*flux_wall(z, vrel[0])
        print("  %-6.2f %-13.5f %-15.5f %-15.5f %-15.4f" % (z, b, b + w, (b + w)/g, g))
run_mix("Couette   ", sh, 0.0)
run_mix("Poiseuille", po, -2.0)
