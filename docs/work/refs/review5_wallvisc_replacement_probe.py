import math, sys, torch
sys.path.insert(0, "../tests/edge")
from scipy import integrate
import test_deltasph as T
F64 = torch.float64
sim, info = T.small_tank("cuda:0")
sim.cfg.ddt = False; sim.cfg.delta = 0.0; sim.cfg.viscosityExact = True
dp, bed, H = sim.dx, info["bed"], sim.H
x = torch.tensor([[0.0, bed + 0.5*dp]], dtype=F64, device="cuda:0"); rho = torch.ones(1, dtype=F64, device="cuda:0")
sim.Hvec = torch.full((1,), H, dtype=F64, device="cuda:0"); sim.kinds = torch.zeros(1, dtype=torch.int32, device="cuda:0")
got = {}
for name, v in (("into", (0.0, -1.0)), ("along", (1.0, 0.0))):
    vel = torch.tensor([v], dtype=F64, device="cuda:0")
    sim.cfg.wallViscosity = True;  a_on = sim.rhs(x, vel, rho)[0]
    sim.cfg.wallViscosity = False; a_off = sim.rhs(x, vel, rho)[0]
    got[name] = (a_on - a_off)[0].cpu().numpy()
# independent value: flat wall, Delta-lambda = -z * int_{-L}^{L} W'(r)/r dx'   (divergence theorem on the wall line), W = 7/(pi H^2) (1-q)^4 (1+4q)
z = 0.5*dp
dW = lambda r: 7/(math.pi*H**2)*(-20*(r/H)*(1-r/H)**3)/H          # dW/dr
Lc = math.sqrt(H*H - z*z)
B = -z*integrate.quad(lambda xx: dW(math.hypot(xx, z))/math.hypot(xx, z), -Lc, Lc, epsabs=1e-13, epsrel=1e-13)[0]
fac = sim.cfg.alpha*sim.cfg.c0*H/sim.xi
pred_y = -2*(fac/8)*sim.cfg.wallMass*(+1.0)*B*(-1.0)      # u_n = +1 into the wall, n = (0,-1): acc = -2 nu_eff wallMass u_n Delta-lambda n
print("H %.4f z %.4f B %.6f fac %.6f   predicted a_y = %.6f   solver a_into = %s   a_along = %s" % (H, z, B, fac, pred_y, got["into"], got["along"]))
print("relative error %.2e" % (abs(got["into"][1]-pred_y)/abs(pred_y)))
