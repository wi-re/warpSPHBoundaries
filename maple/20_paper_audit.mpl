# Symbolic audit of the claims in paper/ that maple/01..14 do not already cover.
# Run: ~/maple2026/bin/maple -q maple/20_paper_audit.mpl   (every line must end in PASS)
interface(quiet = true):
chk := proc(name, ok) printf("%s: %s\n", name, `if`(evalb(ok), "PASS", "FAIL")); end proc:

W := table([w2 = (1-q)^4*(1+4*q), w4 = (1-q)^6*(1+6*q+35/3*q^2),
            w6 = (1-q)^8*(1+8*q+25*q^2+32*q^3)]):
C2 := table([cubic = 80/(7*Pi), w2 = 7/Pi, w4 = 9/Pi, w6 = 78/(7*Pi)]):
C3 := table([cubic = 16/Pi, w2 = 21/(2*Pi), w4 = 495/(32*Pi), w6 = 1365/(64*Pi)]):
cub := (1-q)^3: cubc := -4*(1/2-q)^3:   # cubic = cub + cubc on [0,1/2], cub on [1/2,1]

# ---------------------------------------------------------------- kernels
for k in [w2, w4, w6] do
  chk(cat("normalisation 2D ", k), simplify(2*Pi*C2[k]*int(W[k]*q, q = 0..1) - 1) = 0):
  chk(cat("normalisation 3D ", k), simplify(4*Pi*C3[k]*int(W[k]*q^2, q = 0..1) - 1) = 0):
end do:
chk("normalisation 2D cubic", simplify(2*Pi*C2[cubic]*(int(cub*q, q = 0..1) + int(cubc*q, q = 0..1/2)) - 1) = 0):
chk("normalisation 3D cubic", simplify(4*Pi*C3[cubic]*(int(cub*q^2, q = 0..1) + int(cubc*q^2, q = 0..1/2)) - 1) = 0):
chk("cubic on [0,1/2] is 1/2-3q^2+3q^3", expand(cub + cubc - (1/2 - 3*q^2 + 3*q^3)) = 0):
chk("cubic indicator weights 8/7, -1/7",
    simplify(2*Pi*C2[cubic]*int(q*cub, q = 0..1) - 8/7) = 0 and simplify(2*Pi*C2[cubic]*int(q*cubc, q = 0..1/2) + 1/7) = 0):

# ---------------------------------------------------------------- 3D planar
lam3B := -(8/15)*(2*d^6 - 9*d^5 + 15*d^4 - 10*d^3 + 3*d - 1):
lam3A := (192*d^6 - 288*d^5 + 160*d^3 - 84*d + 30)/60:
chk("3D cubic branch B", simplify(int(C3[cubic]*cub*2*Pi*q*(q-d), q = d..1) - lam3B) = 0):
chk("3D cubic branch A", simplify(int(C3[cubic]*cub*2*Pi*q*(q-d), q = d..1)
    + int(C3[cubic]*cubc*2*Pi*q*(q-d), q = d..1/2) - lam3A) = 0):
chk("3D cubic join at d=1/2", eval(lam3A, d = 1/2) = eval(lam3B, d = 1/2)):
chk("3D cubic lambda(0)=1/2, lambda(1)=0", eval(lam3A, d = 0) = 1/2 and eval(lam3B, d = 1) = 0):
for k in [[w2, 8], [w4, 11], [w6, 14]] do
  L := expand(int(C3[k[1]]*W[k[1]]*2*Pi*q*(q-d), q = d..1)):
  chk(cat("3D ", k[1], ": degree ", k[2], ", lambda(0)=1/2, lambda(1)=0"),
      degree(L, d) = k[2] and eval(L, d = 0) = 1/2 and eval(L, d = 1) = 0):
end do:

# ---------------------------------------------------------------- 2D planar by parts
chk("d/dq arccos(d/q) = d/(q sqrt(q^2-d^2))",
    evalb((simplify(diff(arccos(d/q), q) - d/(q*sqrt(q^2-d^2))) assuming 0 < d, d < q) = 0)):
for m from 1 to 6 do
  A := int(q^m/sqrt(q^2-d^2), q) assuming 0 < d, d < q:
  chk(cat("J_", m, " has an elementary antiderivative"),
      evalb((simplify(diff(A, q) - q^m/sqrt(q^2-d^2)) assuming 0 < d, d < q) = 0)):
end do:
# d lambda_2/dd must equal -int_d^1 2 C2 W q/sqrt(q^2-d^2) dq (boundary term vanishes: arccos(1)=0)
for k in [w2, w4, w6] do
  v0 := int(t*eval(W[k], q = t), t = 0..q):
  chk(cat("2D ", k, ": v0'(q) = q W"), simplify(diff(v0, q) - q*W[k]) = 0):
  chk(cat("2D ", k, ": v0(1) = 1/(2 Pi C2)"), simplify(eval(v0, q = 1) - 1/(2*Pi*C2[k])) = 0):
  lam := 2*C2[k]*(eval(v0, q = 1)*arccos(d) - d*int(expand(v0/q)/sqrt(q^2-d^2), q = d..1)):
  chk(cat("2D ", k, ": lambda(0) = 1/2"), simplify(limit(lam, d = 0, right) - 1/2) = 0):
  Digits := 40:
  df := evalf(eval(diff(lam, d), d = 3/10)):
  rhsv := evalf(-Int(2*C2[k]*eval(W[k], q = q)*q/sqrt(q^2 - (3/10)^2), q = 3/10..1)):
  chk(cat("2D ", k, ": d lambda/dd = -int 2 C2 W q/sqrt(q^2-d^2) (40 digits)"), evalb(abs(df - rhsv) < 10^(-30))):
  Digits := 15:
end do:
Digits := 40:
join := evalf((64*Pi - 294*sqrt(3) + 249*ln(2+sqrt(3)))/(168*Pi)):
lc := evalf(Int(C2[cubic]*cub*2*q*arccos(1/(2*q)), q = 1/2..1)):   # d=1/2: only the outer piece
chk("2D cubic join value at d=1/2", evalb(abs(lc - join) < 10^(-35))):
Digits := 15:

# ---------------------------------------------------------------- sphere
DD := R + d:
Acap := 2*Pi*q*(q - (q^2 + DD^2 - R^2)/(2*DD)):
chk("cap area = Pi q/D (R^2-D^2+2Dq-q^2)", simplify(Acap - Pi*q/DD*(R^2 - DD^2 + 2*DD*q - q^2)) = 0):
chk("cap vanishes at q=D-R and q=D+R", simplify(eval(Acap, q = DD-R)) = 0 and simplify(eval(Acap, q = DD+R)) = 0):
chk("A*D/(Pi q) is linear in R", degree(expand(simplify(Acap*DD/(Pi*q))), R) = 1):
chk("R->infinity: cap area -> planar 2 Pi q (q-d)", simplify(limit(Acap, R = infinity) - 2*Pi*q*(q-d)) = 0):

# ---------------------------------------------------------------- divergence identities (generic M)
g2 := simplify(2*Mg(r)/r^2 + r*diff(Mg(r)/r^2, r)):
chk("2D: div(y M/r^2) = M'/r", simplify(g2 - diff(Mg(r), r)/r) = 0):
g3 := simplify(3*Mg(r)/r^3 + r*diff(Mg(r)/r^3, r)):
chk("3D: div(y M3/r^3) = M3'/r^2", simplify(g3 - diff(Mg(r), r)/r^2) = 0):
chk("div(y/r^2)=0 (2D), div(y/r^3)=0 (3D)", simplify(2/r^2 + r*diff(1/r^2, r)) = 0 and simplify(3/r^3 + r*diff(1/r^3, r)) = 0):
# moment form (b): div(y P F) with P homogeneous of degree k=3, n=4
P := y1^2*y2: rr := sqrt(y1^2 + y2^2): ee := 4 + 2 + 3:
dv := proc(Fx) diff(y1*P*Fx, y1) + diff(y2*P*Fx, y2) end proc:
chk("moment (b) inside: div(y P F) = P r^n", simplify(dv(rr^4/ee) - P*rr^4) = 0):
chk("moment (b) outside: div = 0", simplify(dv(R0^ee*rr^(-5)/ee)) = 0):
chk("circle integrals 2 Pi (a-1)!!(b-1)!!/(a+b)!!",
    simplify(int(cos(t)^2*sin(t)^2, t = 0..2*Pi) - 2*Pi/8) = 0 and
    simplify(int(cos(t)^4*sin(t)^2, t = 0..2*Pi) - 2*Pi*3/48) = 0 and
    simplify(int(cos(t)^2, t = 0..2*Pi) - Pi) = 0):
# recursion (a): grad of Phi(r) = -int_r^R t f is y f
fn := r^5:
Phi := -int(t*eval(fn, r = t), t = r..R0):
chk("recursion (a): d_i Phi = y_i f", simplify(diff(Phi, r)/r - fn) = 0):
# truncated block
chk("block: (M-M(R))/r^2 = (r^n - R^(n+2)/r^2)/(n+2)",
    simplify((r^7/7 - R0^7/7)/r^2 - (r^5 - R0^7/r^2)/7) = 0):

# ---------------------------------------------------------------- Laplacian / viscosity constants
for k in [[w2, (1-q)^3], [w4, 6*(1-q)^5 - 5*(1-q)^6]] do
  Wk := C2[k[1]]*eval(W[k[1]], q = r):
  L := diff(Wk, r)/r:
  chk(cat(k[1], ": W'' + W'/r = 2L + r L'"), simplify(diff(Wk, r$2) + diff(Wk, r)/r - (2*L + r*diff(L, r))) = 0):
  chk(cat(k[1], ": int_disk r W' dA = -2"), simplify(2*Pi*int(r^2*diff(Wk, r), r = 0..1) + 2) = 0):
  shape := eval(k[2], q = r):
  Pl := simplify(Pi*L/(Pi*C2[k[1]]*shape)):
  Cl := 1/(2*int(r*shape, r = 0..1)):
  printf("   %a: P = %a, C_l = %a, f = P c/C_l = %a\n", k[1], Pl, Cl, simplify(Pl*Pi*C2[k[1]]/Cl)):
  chk(cat(k[1], ": L = const * registered shape"), type(Pl, numeric)):
end do:

# ---------------------------------------------------------------- tier 3 geometry
Rr := 1/kap:
dist2 := (d+s)^2 + (1+kap*d)*(1-kap*s)*2*(1-cos(kap*t))/kap^2:
exact2 := (Rr + d)^2 + (Rr - s)^2 - 2*(Rr + d)*(Rr - s)*cos(kap*t):
chk("|y|^2 tubular identity", simplify(dist2 - exact2) = 0):
ser := convert(series(dist2, kap = 0, 8), polynom):
chk("|y|^2 series coefficients (k^1, k^2)",
    expand(coeff(ser, kap, 1) - t^2*(d - s)) = 0 and expand(coeff(ser, kap, 2) - (-d*s*t^2 - t^4/12)) = 0):
chk("area element J = 1 - kap s", simplify((Rr - s)/Rr - (1 - kap*s)) = 0):
a1 := t^2*(d - s): b1 := -d*s*t^2 - t^4/12:
T := expand((1 - kap*s)*(Ph0 + Pd1*(kap*a1 + kap^2*b1) + Pd2/2*(kap*a1 + kap^2*b1)^2)):
chk("F1 integrand", simplify(coeff(T, kap, 1) - (Pd1*t^2*(d - s) - s*Ph0)) = 0):
chk("F2 integrand", simplify(coeff(T, kap, 2) - (Pd1*(-d*s*t^2 - t^4/12) + Pd2/2*t^4*(d-s)^2 - s*Pd1*t^2*(d-s))) = 0):

# ---------------------------------------------------------------- tier 4
fm := x^6*y^2 + x^3*y^5 + x^2 + y^4:
lapf := g -> diff(g, x, x) + diff(g, y, y):
mean := int(int(eval(fm, [x = rho*cos(ph), y = rho*sin(ph)])*rho, rho = 0..a), ph = 0..2*Pi)/(Pi*a^2):
lk := fm: ser4 := 0:
for k from 0 to 4 do
  ser4 := ser4 + (a^2/4)^k*eval(lk, [x = 0, y = 0])/(k!*(k+1)!): lk := lapf(lk):
end do:
chk("disk mean-value series", simplify(mean - ser4) = 0):
dn := proc(g, v, n) local i, h; h := g; for i to n do h := diff(h, v) end do; h end proc:
gz := z^9 + 3*z^4 + z:
mean_s := int(eval(gz, z = z0 + u), u = -a..a)/(2*a):
chk("strip series sum a^2k f^(2k)/(2k+1)!",
    simplify(mean_s - add(a^(2*k)*eval(dn(gz, z, 2*k), z = z0)/(2*k+1)!, k = 0..5)) = 0):
chk("shell theorem (f = r^3)", evalb((simplify(2*Pi*int(sin(th)*(D0^2 + rh^2 - 2*D0*rh*cos(th))^(3/2), th = 0..Pi)
    - 2*Pi/(D0*rh)*int(r^4, r = D0 - rh..D0 + rh)) assuming 0 < rh, rh < D0) = 0)):

# ---------------------------------------------------------------- second pass (paper revision)
# planar 2D: J_m recursion (Eq. Jrec) and the explicit w2 closed form, 40 digits at d = 3/10
Jm := proc(m) option remember;
  if m = 0 then ln((1+sqrt(1-d^2))/d) elif m = 1 then sqrt(1-d^2) else (sqrt(1-d^2) + (m-1)*d^2*Jm(m-2))/m end if end proc:
Digits := 40:
for m from 0 to 6 do
  chk(cat("J_", m, " recursion value at d=3/10 (40 digits)"),
      evalb(abs(evalf(eval(Jm(m), d = 3/10)) - evalf(Int(q^m/sqrt(q^2 - (3/10)^2), q = 3/10..1))) < 10^(-30))):
end do:
lam2w2 := 14/Pi*(arccos(d)/14 - d*(Jm(1)/2 - 5*Jm(3)/2 + 4*Jm(4) - 5*Jm(5)/2 + 4*Jm(6)/7)):
chk("2D w2 explicit closed form = arc form (40 digits, d=3/10)",
    evalb(abs(evalf(eval(lam2w2, d = 3/10)) - evalf(Int(C2[w2]*W[w2]*2*q*arccos(3/10/q), q = 3/10..1))) < 10^(-30))):
chk("2D w2: v0 polynomial and 2 C2 v0(1) = 1/Pi",
    simplify(int(q*W[w2], q = 0..r) - (r^2/2 - 5*r^4/2 + 4*r^5 - 5*r^6/2 + 4*r^7/7)) = 0 and
    simplify(2*C2[w2]*(1/2 - 5/2 + 4 - 5/2 + 4/7) - 1/Pi) = 0):
chk("2D cubic lambda(1/2) = 0.03744 (4 digits)",
    evalb(abs(evalf((64*Pi - 294*sqrt(3) + 249*ln(2 + sqrt(3)))/(168*Pi)) - 0.03744216) < 10^(-7))):
Digits := 15:

# edge primitives: odd-j reduction and even-j identity
rr := sqrt(s^2 + z^2):
for jj in [1, 3, 5] do
  for mm in [0, 1, 2, 3] do
    pp := (jj - 1)/2:
    F := add(binomial(pp, i)*(-z^2)^(pp - i)*rr^(mm + 2*i + 2)/(mm + 2*i + 2), i = 0..pp):
    chk(cat("S_{", jj, ",", mm, "} odd-j formula"), simplify(diff(F, s) - s^jj*rr^mm) = 0):
  end do:
end do:
for jj in [2, 4, 6] do
  pp := jj/2:
  chk(cat("S_{", jj, ",m} even-j: s^j r^m = sum C(p,i)(-z^2)^(p-i) r^(m+2i)"),
      simplify(add(binomial(pp, i)*(-z^2)^(pp - i)*rr^(2*i), i = 0..pp) - s^jj) = 0):
end do:
chk("vertex limit: arctan(inf) - arctan(-cot(alpha/2)) = Pi - alpha/2",
    evalb((simplify(arctan(infinity) - arctan(-cot(al/2))) assuming 0 < al, al < Pi) = Pi - al/2)):

# half-plane value identity at 40 digits (w2, d = 3/10): lambda = -d int (M(r) - M(1))/r^2 ds over the chord
Digits := 40:
M2 := eval(C2[w2]*int(t*eval(W[w2], q = t), t = 0..rr0), rr0 = r):
Mw := unapply(C2[w2]*int(t*eval(W[w2], q = t), t = 0..r), r):
dd := 3/10: ell := sqrt(1 - dd^2):
vi := -dd*evalf(Int((Mw(sqrt(s^2 + dd^2)) - Mw(1))/(s^2 + dd^2), s = -ell..ell)):
chk("value identity, half plane, w2, d=3/10 (40 digits)",
    evalb(abs(vi - evalf(Int(C2[w2]*W[w2]*2*q*arccos(dd/q), q = dd..1))) < 10^(-30))):
Digits := 15:

# curvature: regularity of phi(sigma) = W(sqrt(sigma)) at sigma = 0
for k in [[w2, 15], [cubic, 9/4]] do
  Wk := `if`(k[1] = cubic, C2[cubic]*(cub + cubc), C2[k[1]]*W[k[1]]):
  Wr := eval(Wk, q = r):
  p2 := diff(diff(Wr, r)/r, r)/(4*r):
  chk(cat(k[1], ": r*phi'' -> const at 0 (singular, phi'' ~ c/r)"), simplify(limit(r*p2/C2[k[1]], r = 0, right) - k[2]) = 0):
end do:
for k in [w4, w6] do
  Wr := eval(C2[k]*W[k], q = r):
  p2 := diff(diff(Wr, r)/r, r)/(4*r):
  chk(cat(k, ": phi'' bounded at 0"), evalb(abs(evalf(limit(p2, r = 0, right))) < 10^6)):
end do:

# Laplacian constants: normalised shapes and the factor f = P C2 / C_l
chk("C_l (w2): 2 Pi C_l int q (1-q)^3 = 1 for C_l = 10/Pi", simplify(2*Pi*(10/Pi)*int(q*(1-q)^3, q = 0..1) - 1) = 0):
chk("C_l (w4): 2 Pi C_l int q (1+5q)(1-q)^5 = 1 for C_l = 28/(3 Pi)", simplify(2*Pi*(28/(3*Pi))*int(q*(1+5*q)*(1-q)^5, q = 0..1) - 1) = 0):
chk("w2 W'/q = -20 (1-q)^3 and w4 W'/q = -(56/3)(1+5q)(1-q)^5",
    simplify(diff(W[w2], q)/q + 20*(1-q)^3) = 0 and simplify(diff(W[w4], q)/q + (56/3)*(1+5*q)*(1-q)^5) = 0):
chk("f = P C2/C_l = -14 (w2), -18 (w4)", simplify(-20*7/10 + 14) = 0 and simplify(-(56/3)*9/(28/3) + 18) = 0):

# pairwise bulk coefficient (Prop. nueff), n = 2: int y1^2 y2^2 W'/r^3 dA = -1/4  =>  a = fac/4 for v = (y^2, 0), (fac/8)|lap v|
for k in [w2, w4, w6] do
  Wk := C2[k]*eval(W[k], q = r):
  chk(cat(k, ": int y1^2 y2^2 W'/r^3 dA = -1/4"),
      simplify(int(cos(t)^2*sin(t)^2, t = 0..2*Pi)*int(r^4*diff(Wk, r)/r^3*r, r = 0..1) + 1/4) = 0):
  chk(cat(k, ": Morris form int y1^2 W'/r dA = -1 (so lap y1^2 = 2 = -2 int)"),
      simplify(int(cos(t)^2, t = 0..2*Pi)*int(r^2*diff(Wk, r)/r*r, r = 0..1) + 1) = 0):
end do:
chk("isotropic 4th moment n=2: int cos^4 = 2 Pi 3/8, int cos^2 sin^2 = 2 Pi/8 (=> 1/(n(n+2)) = 1/8)",
    simplify(int(cos(t)^4, t = 0..2*Pi) - 2*Pi*3/8) = 0 and simplify(int(cos(t)^2*sin(t)^2, t = 0..2*Pi) - 2*Pi/8) = 0):

# polygon vs disk (Prop. ngon): area deficit/excess and mean radial deviations, a = Pi/N
chk("inscribed N-gon area / (Pi R^2) = 1 - (2/3) a^2 + O(a^4)",
    simplify(coeff(convert(series(sin(2*aa)/(2*aa), aa = 0, 6), polynom), aa, 2) + 2/3) = 0):
chk("circumscribed N-gon area / (Pi R^2) = 1 + (1/3) a^2 + O(a^4)",
    simplify(coeff(convert(series(tan(aa)/aa, aa = 0, 6), polynom), aa, 2) - 1/3) = 0):
din := int(R*(convert(series(cos(aa)/cos(ph), ph = 0, 8), polynom) - 1), ph = -aa..aa)/(2*aa):
dci := int(R*(convert(series(1/cos(ph), ph = 0, 8), polynom) - 1), ph = -aa..aa)/(2*aa):
chk("mean radial deviation: inscribed -R a^2/3, circumscribed R a^2/6 (leading order)",
    simplify(coeff(convert(series(din, aa = 0, 5), polynom), aa, 2)/R + 1/3) = 0 and
    simplify(coeff(convert(series(dci, aa = 0, 5), polynom), aa, 2)/R - 1/6) = 0):

# Pizzetti by Darboux: circle mean of a polynomial = sum rho^(2k) lap^k f / (4^k k!^2)
cm := int(eval(fm, [x = rho*cos(ph), y = rho*sin(ph)]), ph = 0..2*Pi)/(2*Pi):
lk := fm: sum4 := 0:
for k from 0 to 4 do
  sum4 := sum4 + rho^(2*k)*eval(lk, [x = 0, y = 0])/(4^k*(k!)^2): lk := lapf(lk):
end do:
chk("circle mean series (Pizzetti)", simplify(cm - sum4) = 0):
chk("Darboux: u'' + u'/rho of rho^(2k) is (2k)^2 rho^(2k-2)", simplify(diff(rho^(2*kk), rho$2) + diff(rho^(2*kk), rho)/rho - (2*kk)^2*rho^(2*kk-2)) = 0):

printf("%s\n", "20_paper_audit: complete"):
