# VERIFIED (exploratory, not yet in the paper): closed-form wall Laplacian for a position-dependent free-slip MIRROR ghost field.
#  solid part  int_S (v_ghost - v) lap W  with  v_ghost(x+y) = R v(x + y_m),  y_m = R y + 2 d n,  v Taylor-expanded at the particle to order p,
#  moments  int_S y^a lap W = sum_e int_chord [ y^a W' z/r - W n.grad y^a ] ds + int_S W lap y^a   (Green's 2nd identity; edge integrals + lower value moments).
#  Test field: cubic polynomial with the free-slip symmetry (v_x even, v_y odd about the wall).  p=3 reproduces lap v to 1e-12; p=0 is the solver's constant mirror.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, sympy as sp
from scipy.integrate import quad
import paper_edge_identities_k6 as E
from numpy.polynomial import Polynomial as Poly

SOLID = np.array([[-3,-3],[3,-3],[3,0],[-3,0]],float)          # solid y<0, CCW
FLUID = np.array([[-3,0],[3,0],[3,3],[-3,3]],float)           # fluid y>0, CCW
d = 0.3; x = np.array([0.0, d])
X,Y = sp.symbols('X Y')
vx = 1 + 2*X + sp.Rational(1,2)*X**2 + sp.Rational(7,10)*Y**2 + sp.Rational(3,10)*X*Y**2     # even in Y
vy = sp.Rational(2,5)*Y + sp.Rational(3,5)*X*Y + sp.Rational(1,5)*X**2*Y + sp.Rational(3,10)*Y**3  # odd in Y
v = sp.Matrix([vx,vy])
lap_exact = np.array([float((sp.diff(c,X,2)+sp.diff(c,Y,2)).subs({X:x[0],Y:x[1]})) for c in v])
y1,y2 = sp.symbols('y1 y2')
def monos(expr):
    P = sp.Poly(sp.expand(expr), y1, y2); return {m: float(c) for m,c in P.terms()}

def lapW_blocks(blocks, extra):                                   # blocks of r^extra * r*(W'' + W'/r) = r^(extra+1) W'' + r^extra W'
    dW=[(p.deriv(),rc) for p,rc in blocks]; ddW=[(p.deriv().deriv(),rc) for p,rc in blocks]
    sh=lambda bl,k:[(p*Poly([0]*k+[1]),rc) for p,rc in bl]
    return sh(ddW,extra+1)+sh(dW,extra)

def moment_lapW_ray(P, alpha, blocks):                            # reference: int_P y^alpha lap W
    return E.ray(P, x, alpha, lapW_blocks(blocks, sum(alpha)), n=1_000_000)

def moment_lapW_green(P, alpha, blocks):                          # closed-form route: Green's 2nd identity
    # int_S y^a lap W = sum_e int_chord [ y^a W' z/r - W n.grad(y^a) ] ds + int_S W lap(y^a)
    a1,a2 = alpha; tot = 0.0
    dW=[(p.deriv(),rc) for p,rc in blocks]
    for a,b,t,n in E.edges(P):
        z,s0,s1 = n@(a-x), t@(a-x), t@(b-x); c = E.chord(z,s0,s1,1.0)
        if not c: continue
        def f(s):
            yv = z*n + s*t; r = np.hypot(s,z)
            ya = yv[0]**a1*yv[1]**a2
            gr = np.array([a1*yv[0]**max(a1-1,0)*yv[1]**a2 if a1 else 0.0, a2*yv[0]**a1*yv[1]**max(a2-1,0) if a2 else 0.0])
            return ya*E.evalb(dW,r)*z/r - E.evalb(blocks,r)*(n@gr)
        tot += quad(f,*c,epsabs=1e-14,epsrel=1e-12,limit=200)[0]
    if a1>=2: tot += a1*(a1-1)*E.mom(P,x,(a1-2,a2),blocks)
    if a2>=2: tot += a2*(a2-1)*E.mom(P,x,(a1,a2-2),blocks)
    return tot

for kn in ["w4","cubic"]:
    B = E.KERN[kn]
    print(kn, "Green-2nd-identity moments of lap W over the solid vs ray quadrature:")
    for al in [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3)]:
        g_, r_ = moment_lapW_green(SOLID,al,B), moment_lapW_ray(SOLID,al,B)
        print(f"   alpha={al}: closed {g_:+.10f} ray {r_:+.10f} diff {abs(g_-r_):.1e}")
    # free-slip polynomial test
    n_in = np.array([0.0,-1.0]); R = np.eye(2)-2*np.outer(n_in,n_in)                # wall normal into the wall, mirror matrix
    vx0 = np.array([float(c.subs({X:x[0],Y:x[1]})) for c in v])
    # fluid part, exact polynomial (reference quadrature over the fluid polygon)
    dvf = [sp.expand(c.subs({X:x[0]+y1,Y:x[1]+y2}) - float(c.subs({X:x[0],Y:x[1]}))) for c in v]
    fluid = np.array([sum(cf*E.ray(FLUID,x,m,lapW_blocks(B,sum(m)),n=1_000_000) for m,cf in monos(e).items()) for e in dvf])
    print("   fluid part", fluid)
    # solid part with ghost field = mirror, Taylor-truncated at order p about x
    for p in [0,1,2,3]:
        # y_m = R y + 2 d n   (relative coordinates, wall at y.n = d)
        zz1,zz2 = sp.symbols('zz1 zz2')                                   # displacement of the MIRROR point from the particle
        vz = [sp.expand(c.subs({X:x[0]+zz1,Y:x[1]+zz2},simultaneous=True)) for c in v]
        def trunc(e):
            Pp = sp.Poly(e,zz1,zz2); return sum(c*zz1**m[0]*zz2**m[1] for m,c in Pp.terms() if sum(m)<=p)
        ym = R@sp.Matrix([y1,y2]) + 2*d*sp.Matrix(n_in)                   # mirror point of x+y, relative to x
        vg = (R*sp.Matrix([trunc(e).subs({zz1:ym[0],zz2:ym[1]},simultaneous=True) for e in vz])).applyfunc(sp.expand)
        dvg = [sp.expand(vg[i]-float(vx0[i])) for i in range(2)]
        wall_exact_ref = np.array([sum(cf*moment_lapW_ray(SOLID,m,B) for m,cf in monos(e).items()) for e in dvg])
        wall = np.array([sum(cf*moment_lapW_green(SOLID,m,B) for m,cf in monos(e).items()) for e in dvg])
        tot = fluid+wall
        print(f"   ghost Taylor order p={p}: |wall(closed) - wall(ray)| {np.max(np.abs(wall-wall_exact_ref)):.1e};  total {tot}  vs  lap v {lap_exact}   error {np.max(np.abs(tot-lap_exact)):.2e}")
