# ===========================================================================
# 14_tier3_2d.mpl -- docs/derivations/tier3-curvature-2d.md : closest-point (tubular) expansion of the 2D value integral.
#
# Solid DISK of radius R = 1/kappa (kappa > 0 convex), centre at the origin, evaluation point x = (0, D), D = R + d (d = distance from the
# surface, outside).  Solid points: polar (rho', theta), rho' = R - s (s = depth into the solid), arc length t = R theta.
#   area element   rho' d rho' d theta = (1 - kappa s) ds dt                          (J = 1 - kappa s)
#   |y|^2 = Dd^2 + rho'^2 - 2 Dd rho' cos(theta) = (d+s)^2 + 2 (1+kappa d)(1-kappa s) (1-cos(kappa t))/kappa^2
# Checks: J, the exact |y|^2 identity, its kappa-series (coefficients of kappa^1, kappa^2, kappa^3) and the Taylor series of Phi(|y|^2)
# for a generic Phi (the integrands of F1, F2 in the derivation file).
# ===========================================================================
restart:
chk := proc(label, ok) printf("%-76s %s\n", label, `if`(ok, "PASS", "FAIL")); end proc:

R := 1/kappa: Dd := R + d: rhop := R - s: th := kappa*t:      # (D is protected in Maple: the distance to the centre is Dd)
# area element: polar area element rho' d rho' d theta, with d rho' = -ds, d theta = kappa dt  ->  rho' kappa ds dt = (1 - kappa s) ds dt
J := simplify(rhop*kappa):
chk("area element J = rho' * kappa = 1 - kappa s", evalb(simplify(J - (1 - kappa*s)) = 0));

y2 := Dd^2 + rhop^2 - 2*Dd*rhop*cos(th):
y2form := (d+s)^2 + 2*(1+kappa*d)*(1-kappa*s)*(1-cos(kappa*t))/kappa^2:
chk("|y|^2 = (d+s)^2 + 2(1+kappa d)(1-kappa s)(1-cos kappa t)/kappa^2  (exact)", evalb(simplify(expand(y2 - y2form)) = 0));

ser := series(y2form, kappa=0, 7):
c1 := coeff(convert(ser, polynom), kappa, 1): c2 := coeff(convert(ser, polynom), kappa, 2): c3 := coeff(convert(ser, polynom), kappa, 3):
c0 := coeff(convert(ser, polynom), kappa, 0):
chk("kappa^0:  (d+s)^2 + t^2", evalb(expand(c0 - ((d+s)^2 + t^2)) = 0));
chk("kappa^1:  t^2 (d - s)", evalb(expand(c1 - t^2*(d-s)) = 0));
chk("kappa^2:  -d s t^2 - t^4/12", evalb(expand(c2 - (-d*s*t^2 - t^4/12)) = 0));
chk("kappa^3:  t^4 (s - d)/12", evalb(expand(c3 - t^4*(s-d)/12) = 0));


# Taylor series of Phi(sigma0 + delta) in kappa, generic Phi: integrand of  lambda = int (1 - kappa s) Phi(|y|^2) dt ds
sigma := (d+s)^2 + t^2 + kappa*t^2*(d-s) + kappa^2*(-d*s*t^2 - t^4/12):
integrand := (1 - kappa*s)*Phi(sigma):
sr := convert(series(integrand, kappa=0, 3), polynom):
s0 := eval(coeff(sr, kappa, 0)): s1 := coeff(sr, kappa, 1): s2 := coeff(sr, kappa, 2):
sig0 := (d+s)^2 + t^2:
F1int := D(Phi)(sig0)*t^2*(d-s) - s*Phi(sig0):
F2int := D(Phi)(sig0)*(-d*s*t^2 - t^4/12) + (1/2)*(D@@2)(Phi)(sig0)*t^4*(d-s)^2 - s*D(Phi)(sig0)*t^2*(d-s):
chk("F1 integrand  Phi'(s0) t^2 (d-s) - s Phi(s0)", evalb(simplify(s1 - F1int) = 0));
chk("F2 integrand  Phi'(-d s t^2 - t^4/12) + Phi''/2 t^4 (d-s)^2 - s Phi' t^2 (d-s)", evalb(simplify(s2 - F2int) = 0));
chk("F0 integrand Phi(s0)", evalb(simplify(s0 - Phi(sig0)) = 0));

# half-plane variables: y1 = t, y2 = d + s  (s = y2 - d, d - s = 2d - y2)
print("14_tier3_2d: complete");
