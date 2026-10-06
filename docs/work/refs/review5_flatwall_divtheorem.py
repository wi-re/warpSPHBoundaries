# independent: flat wall, Delta-lambda(z) = -z * int W'(r)/r dx'  (divergence theorem on the wall line), no warpSPHBoundaries code
import numpy as np
from scipy.integrate import quad
def W2(q): return 7/np.pi*(1-q)**4*(1+4*q)
def dW2(q): return 7/np.pi*(-20*q*(1-q)**3)         # dW/dq (H=1)
def W4(q): return 9/np.pi*(1-q)**6*(1+6*q+35/3*q*q)
def dW4(q): return 9/np.pi*(-(56/3)*q*(1-q)**5*(1+5*q))
for name,dW in (("C2",dW2),("C4",dW4)):
    print(name,[round(-z*quad(lambda x: dW(np.hypot(x,z))/np.hypot(x,z),-np.sqrt(1-z*z),np.sqrt(1-z*z),epsabs=1e-13,epsrel=1e-13)[0],7) for z in (0.02,0.1,0.3,0.5,0.7,0.9)])
# normalisation check
print("norm", quad(lambda r:2*np.pi*r*W2(r),0,1)[0], quad(lambda r:2*np.pi*r*W4(r),0,1)[0])
