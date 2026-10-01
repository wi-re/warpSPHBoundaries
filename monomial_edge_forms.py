# UNVERIFIED: per-monomial edge forms + closed-form primitives I_m, J_m, S_{j,m}; direct far-field form for y^alpha moments.
# The brute-force dblquad reference is very slow (discontinuity at r=R) and timed out; replace with a polar reference
# (integrate around x with a radial breakpoint at R) or check symbolically in Maple.
import numpy as np, math
from scipy.integrate import dblquad, quad
from math import comb

# ---- closed-form 1D primitives along an edge: r = sqrt(s^2+z^2) ----
def I(m, s, z):               # odd antiderivative of r^m ds, integer m
    r = math.hypot(s, z)
    if m == 0:  return s
    if m == -1: return math.asinh(s/abs(z))
    if m == -2: return math.atan(s/z)/z
    if m > 0:   return (s*r**m + m*z*z*I(m-2, s, z))/(m+1)
    return ((m+3)*I(m+2, s, z) - s*r**(m+2))/((m+2)*z*z)   # downward
def J(m, s, z):               # antiderivative of s r^m ds
    r = math.hypot(s, z)
    return math.log(r) if m == -2 else r**(m+2)/(m+2)
def S(j, m, s, z):            # antiderivative of s^j r^m ds via s^2 = r^2 - z^2
    if j % 2 == 0:
        h = j//2; return sum(comb(h,i)*(-z*z)**(h-i)*I(m+2*i, s, z) for i in range(h+1))
    h = (j-1)//2; return sum(comb(h,i)*(-z*z)**(h-i)*J(m+2*i, s, z) for i in range(h+1))

def edges(T, x, R):           # yields (n, t, z, s_lo, s_hi) for chords inside radius R
    a, b, c = T; o = np.sign(np.cross(np.r_[b-a,0], np.r_[c-a,0])[2])
    for p, q in ((a,b),(b,c),(c,a)):
        t = (q-p)/np.linalg.norm(q-p); n = o*np.array([t[1], -t[0]])
        z = n@(p-x); s0 = t@(p-x); s1 = t@(q-x)
        if abs(z) >= R: continue
        L = math.sqrt(R*R-z*z); lo, hi = max(s0,-L), min(s1,L)
        if lo < hi: yield n, t, z, lo, hi

def inside(T, x):
    a,b,c = T; d = [np.cross(np.r_[q-p,0], np.r_[x-p,0])[2] for p,q in ((a,b),(b,c),(c,a))]
    return all(v>0 for v in d) or all(v<0 for v in d)

# ---- general monomial term:  ∫_T P(x'-x) r^n 1[r<=R] dA, P = y_x^a y_y^b, k=a+b, d=2 ----
def mono(T, x, n, R, a=0, b=0):
    k = a+b; e = n+2+k
    sph = quad(lambda th: math.cos(th)**a*math.sin(th)**b, 0, 2*math.pi)[0]
    val = inside(T,x)*sph*R**e/e
    for nv, t, z, lo, hi in edges(T, x, R):
        c = np.polynomial.polynomial.polymul(                       # P(z n + s t) as poly in s
              np.polynomial.polynomial.polypow([z*nv[0], t[0]], a),
              np.polynomial.polynomial.polypow([z*nv[1], t[1]], b))
        for j, cj in enumerate(c):
            if cj == 0: continue
            F = lambda s: S(j, n, s, z) - R**e*S(j, -(2+k), s, z)
            val += z*cj*(F(hi)-F(lo))/e
    return val
def mono_grad(T, x, n, R):    # ∇_x ∫_T r^n 1[r<=R] dA
    g = np.zeros(2)
    for nv, t, z, lo, hi in edges(T, x, R): g -= nv*(I(n,hi,z)-I(n,lo,z))
    return g

# ---- reference: brute-force area quadrature ----
def area(T, x, f):
    a,b,c = T; Jd = abs(np.cross(np.r_[b-a,0], np.r_[c-a,0])[2])
    return dblquad(lambda v,u: f(a+u*(b-a)+v*(c-a)-x), 0, 1, 0, lambda u:1-u, epsabs=1e-11, epsrel=1e-11, )[0]*Jd

rng = np.random.default_rng(3); worst = {}
for _ in range(4):
    T = [rng.uniform(-1,1,2) for _ in range(3)]; x = rng.uniform(-1,1,2); R = rng.uniform(.4,1.2)
    for (n,a,b) in [(0,0,0),(3,0,0),(6,0,0),(2,1,0),(3,2,1),(5,0,2)]:
        ref = area(T,x,lambda y: y[0]**a*y[1]**b*np.linalg.norm(y)**n*(np.linalg.norm(y)<=R))
        worst[f"value n={n} P=x^{a}y^{b}"] = max(worst.get(f"value n={n} P=x^{a}y^{b}",0), abs(ref-mono(T,x,n,R,a,b)))
    for n in (3,):
        h = 1e-6; num = np.array([(area(T,x+h*e,lambda y: np.linalg.norm(y)**n*(np.linalg.norm(y)<=R)) -
                                   area(T,x-h*e,lambda y: np.linalg.norm(y)**n*(np.linalg.norm(y)<=R)))/(2*h) for e in np.eye(2)])
        worst[f"grad n={n} (FD ref)"] = max(worst.get(f"grad n={n} (FD ref)",0), np.abs(num-mono_grad(T,x,n,R)).max())
for k,v in worst.items(): print(f"{k:28s} {v:.1e}")

# ---- cubic spline 2D as sum of truncated monomials ----
C = 40/(7*math.pi)            # 2D cubic spline norm, support 1
terms = [(C*cf, n, 1.0) for n,cf in enumerate([2,-6,6,-2])] + \
        [(-4*C*cf, n, .5) for n,cf in enumerate([1/8,-3/4,3/2,-1])]   # 2(1-q)^3 - 8(1/2-q)^3 ... (Price form)
def ws(r):
    q=r; return C*(6*(q**3-q**2)+1 if q<=.5 else 2*(1-q)**3 if q<=1 else 0)
full = sum(cf*mono([np.zeros(2)+v for v in [np.array([-5.,-5]),np.array([5.,-5]),np.array([0,5.])]],np.zeros(2),n,R) for cf,n,R in terms)
T = [np.array([-.3,-.2]),np.array([.6,-.1]),np.array([.1,.7])]; x=np.array([.05,.1])
print("cubic spline: full-support integral =", full, " single triangle err =",
      abs(sum(cf*mono(T,x,n,R) for cf,n,R in terms) - area(T,x,lambda y: ws(np.linalg.norm(y)))))
