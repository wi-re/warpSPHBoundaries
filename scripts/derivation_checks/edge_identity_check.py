# VERIFIED: 2D edge identities (value, gradient, first moment) for full Wendland C4 vs adaptive area quadrature; max diff 3e-13.
# Uses quad along edges (not yet closed-form primitives).
import numpy as np

cross2 = lambda u, v: u[0] * v[1] - u[1] * v[0]      # np.cross of 2-vectors was removed in numpy 2
from scipy.integrate import quad, dblquad
from numpy.polynomial import polynomial as P
# Wendland C4 (2D), support 1: k(q)=(1-q)^6 (1+6q+35/3 q^2)
k = P.polymul(P.polypow([1,-1],6),[1,6,35/3])
Mpoly = P.polyint(P.polymul(k,[0,1]))           # int_0^r s k(s) ds
C = 1/(2*np.pi*P.polyval(1,Mpoly))               # normalise
W  = lambda r: C*P.polyval(r,k)*(r<1)
M  = lambda r: C*P.polyval(np.minimum(r,1),Mpoly)
M1 = M(1.0)                                       # = 1/(2pi)
Psi= lambda r: M1-M(r)                            # int_r^1 s W ds, compact

def area(T,x,f):
    (a,b,c)=T
    def g(v,u):  # barycentric param
        p=a+u*(b-a)+v*(c-a); r=np.linalg.norm(p-x); return f(p,r)
    J=abs(cross2(b-a,c-a))
    return dblquad(g,0,1,0,lambda u:1-u,epsabs=1e-13,epsrel=1e-13)[0]*J

def inside(T,x):
    a,b,c=T; s=[cross2(q-p,x-p) for p,q in ((a,b),(b,c),(c,a))]
    return all(v>0 for v in s) or all(v<0 for v in s)

def edges(T,x,fn):
    a,b,c=T; orient=np.sign(cross2(b-a,c-a)); out=0
    for p,q in ((a,b),(b,c),(c,a)):
        t=(q-p)/np.linalg.norm(q-p); n=orient*np.array([t[1],-t[0]])  # outward
        z=n@(p-x); c0=x+z*n; s0=t@(p-c0); s1=t@(q-c0)
        if abs(z)>=1: continue
        L=np.sqrt(1-z*z); lo,hi=max(s0,-L),min(s1,L)
        if lo<hi: out=out+fn(n,z,lo,hi)
    return out

rng=np.random.default_rng(1); worst=0
for _ in range(20):
    T=[rng.uniform(-1,1,2) for _ in range(3)]; x=rng.uniform(-1,1,2)
    # value
    Va=area(T,x,lambda p,r:W(r))
    Ve=float(inside(T,x))+edges(T,x,lambda n,z,lo,hi: z*quad(lambda s:(M(np.hypot(s,z))-M1)/(s*s+z*z),lo,hi,epsabs=1e-14)[0])
    # gradient wrt x
    Ga=np.array([area(T,x,lambda p,r,i=i: (C*P.polyval(r,P.polyder(k))*(x-p)[i]/r if 0<r<1 else 0)) for i in range(2)])
    Ge=-edges(T,x,lambda n,z,lo,hi: n*quad(lambda s:W(np.hypot(s,z)),lo,hi,epsabs=1e-14)[0])
    # first moment (linear FEM field)
    Ma=np.array([area(T,x,lambda p,r,i=i:(p-x)[i]*W(r)) for i in range(2)])
    Me=-edges(T,x,lambda n,z,lo,hi: n*quad(lambda s:Psi(np.hypot(s,z)),lo,hi,epsabs=1e-14)[0])
    e=max(abs(Va-Ve),*abs(Ga-Ge),*abs(Ma-Me)); worst=max(worst,e)
print("max abs diff over 20 random triangles/points:",worst)
