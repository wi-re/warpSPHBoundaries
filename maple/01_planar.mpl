# ===========================================================================
# 01_planar.mpl — planar boundary integrals lambda(d), 2D + 3D, all kernels.
# Exact arithmetic; run from the repo root:  maple -q maple/01_planar.mpl
#
#   3D:  lambda_3(d) = Int_d^1 C3 W(q) 2 Pi q (q-d) dq     (spherical cap shell)
#   2D:  lambda_2(d) = Int_d^1 C2 W(q) 2 q arccos(d/q) dq  (chord-arc shell)
#
# Method
#   3D Wendland : monomial basis F(n,p,x) -> one polynomial in d (deg 2n).
#   3D cubic    : split at the knot q=1/2 -> two polynomials (paper eq. 18).
#   2D Wendland : by parts with u = arccos(d/q), dv = 2 C2 q W(q) dq,
#       v0(q) = Int_0^q t W(t) dt   (SINGLE t: the 2-D shell carries one q)
#       lambda_2 = 2 C2 [ v0(1) arccos(d) - d Int_d^1 v0(q)/(q sqrt(q^2-d^2)) dq ]
#     the residual closes to J_m-type terms: polynomials in d times
#     sqrt(1-d^2), ln(d), ln(1+sqrt(1-d^2)).
#   2D cubic    : same by-parts per kernel piece; branch A (d in [0,1/2])
#     gains arccos(2d), sqrt(1-4d^2), ln(1+sqrt(1-4d^2)) terms; branch B
#     (d in [1/2,1]) has the Wendland shape with W = (1-q)^3.
#
# In-script validation: 50-digit quadrature, exact boundary values,
# branch joins.  Output: results/symbolic/planar_export.txt
# ===========================================================================
restart:
assume(q > 0, d > 0):

# --- kernel table -------------------------------------------------------------
C2t := table([cubic = 80/(7*Pi), w2 = 7/Pi, w4 = 9/Pi, w6 = 78/(7*Pi)]):
C3t := table([cubic = 16/Pi, w2 = 21/(2*Pi), w4 = 495/(32*Pi), w6 = 1365/(64*Pi)]):
Wcs := t -> (1-t)^3 - 4*(1/2-t)^3:
W2  := t -> (1-t)^4*(1+4*t):
W4  := t -> (1-t)^6*(1+6*t+35/3*t^2):
W6  := t -> (1-t)^8*(1+8*t+25*t^2+32*t^3):

# ===========================================================================
# Section 1: 3-D planar, cubic spline (paper eq. 18)
# ===========================================================================
print("=== 1. 3D planar cubic spline ===");
c3v := C3t[cubic]:
fex := c3v*((1-q)^3 - 4*(1/2-q)^3)*2*Pi*q*(q-d):
gex := c3v*(1-q)^3*2*Pi*q*(q-d):
LA3 := simplify(int(fex, q=d..1/2) + int(gex, q=1/2..1)):
LB3 := simplify(int(gex, q=d..1)):
# paper's published closed form
PA := (192*d^6 - 288*d^5 + 160*d^3 - 84*d + 30)/60:
PB := -(8/15)*(2*d^6 - 9*d^5 + 15*d^4 - 10*d^3 + 3*d - 1):
printf("matches paper eq. 18:  b1 %s   b2 %s\n",
   `if`(is(simplify(LA3 - PA) = 0), "PASS", "FAIL"),
   `if`(is(simplify(LB3 - PB) = 0), "PASS", "FAIL"));
printf("b1(0)=%a (1/2)   join b1(1/2)-b2(1/2)=%a (0)   b2(1)=%a (0)\n",
   subs(d=0, LA3), simplify(subs(d=1/2, LA3) - subs(d=1/2, LB3)), subs(d=1, LB3));
kpw := t -> piecewise(t <= 1/2, Wcs(t), (1-t)^3):
q3cs := dv -> evalf(int(c3v*kpw(q)*2*Pi*q*(q-dv), q=dv..1), 50):
for dv in [1/10, 1/4, 3/10, 7/10, 9/10] do
  sv := `if`(dv <= 1/2, subs(d=dv, LA3), subs(d=dv, LB3)):
  printf("  d=%5.2f  closed=%.15f  numeric=%.15f  diff=%.3e\n",
     evalf(dv), evalf(sv, 20), evalf(q3cs(dv), 20),
     abs(evalf(sv-q3cs(dv), 25)));
end do:

# ===========================================================================
# Section 2: 3-D planar, Wendland (monomial basis)
# ===========================================================================
print("");
print("=== 2. 3D planar Wendland ===");
F := (n,p,x) -> local j;
   add(binomial(n,j)*(-1)^j*x^(p+j+1)/(p+j+1), j=0..n):
polyval := (P, x) -> local i; add(P[i][2]*x^P[i][1], i=1..numelems(P)):
planar_mono := (n, P, C3v2) -> local i;
   2*Pi*C3v2*add(P[i][2]*((F(n,P[i][1]+2,1) - F(n,P[i][1]+2,d))
                   - d*(F(n,P[i][1]+1,1) - F(n,P[i][1]+1,d))),
                   i=1..numelems(P)):
kerns := [
   ["w2", 4, [[0,1], [1,4]],            21/(2*Pi)],
   ["w4", 6, [[0,1], [1,6], [2,35/3]],  495/(32*Pi)],
   ["w6", 8, [[0,1], [1,8], [2,25], [3,32]], 1365/(64*Pi)]
]:
Lp3 := table():
for kv in kerns do
  knm := kv[1]: n := kv[2]: P := kv[3]: C3v2 := kv[4]:
  Lp := simplify(expand(planar_mono(n, P, C3v2))):
  Lp3[knm] := Lp:
  Lpd := simplify(int(C3v2*(1-q)^n*polyval(P,q)*2*Pi*q*(q-d), q=d..1)):
  printf("%s: mono == direct int: %s   deg(d)=%d   lambda(0)=%a  lambda(1)=%a\n",
     knm, `if`(is(simplify(Lp - Lpd) = 0), "PASS", "FAIL"),
     degree(Lp, d), subs(d=0, Lp), subs(d=1, Lp));
  for dv in [1/10, 3/10, 7/10, 9/10] do
    sv := subs(d=dv, Lp):
    nv := evalf(int(C3v2*(1-q)^n*polyval(P,q)*2*Pi*q*(q-dv), q=dv..1), 50):
    printf("  d=%5.2f  diff=%.3e\n", evalf(dv), abs(evalf(sv-nv, 25)));
  end do:
end do:

# ===========================================================================
# Section 3: 2-D planar, cubic spline (branches A/B, join at d = 1/2)
# ===========================================================================
print("");
print("=== 3. 2D planar cubic spline ===");
c2v := C2t[cubic]:
wA := int(t*Wcs(t), t=0..q):
wB := int(t*(1-t)^3, t=0..q):
printf("wA' check: %s   wB' check: %s\n",
   `if`(is(simplify(diff(wA, q) - q*Wcs(q)) = 0), "PASS", "FAIL"),
   `if`(is(simplify(diff(wB, q) - q*(1-q)^3) = 0), "PASS", "FAIL"));
rA  := int(wA/(q*sqrt(q^2-d^2)), q=d..1/2):
rB  := int(wB/(q*sqrt(q^2-d^2)), q=1/2..1):
rB2 := int(wB/(q*sqrt(q^2-d^2)), q=d..1):
printf("rA/rB/rB2 closed: %s %s %s\n",
   `if`(has(rA, Int), "NO", "yes"), `if`(has(rB, Int), "NO", "yes"),
   `if`(has(rB2, Int), "NO", "yes"));
LA2 := 2*c2v*(subs(q=1/2, wA)*arccos(2*d)
              - d*rA
              + subs(q=1, wB)*arccos(d) - subs(q=1/2, wB)*arccos(2*d)
              - d*rB):
LB2 := 2*c2v*(subs(q=1, wB)*arccos(d) - d*rB2):
q2csA := dv -> evalf(int(c2v*Wcs(q)*2*q*arccos(dv/q), q=dv..1/2)
             + int(c2v*(1-q)^3*2*q*arccos(dv/q), q=1/2..1), 50):
q2csB := dv -> evalf(int(c2v*(1-q)^3*2*q*arccos(dv/q), q=dv..1), 50):
printf("branch A (d in [0,1/2]):\n");
for dv in [1/20, 1/10, 1/4, 49/100] do
  printf("  d=%5.3f  closed=%.15f  numeric=%.15f  diff=%.3e\n",
     evalf(dv), evalf(subs(d=dv, LA2), 20), evalf(q2csA(dv), 20),
     abs(evalf(subs(d=dv, LA2) - q2csA(dv), 25)));
end do:
printf("branch B (d in [1/2,1]):\n");
for dv in [51/100, 7/10, 99/100] do
  printf("  d=%5.3f  closed=%.15f  numeric=%.15f  diff=%.3e\n",
     evalf(dv), evalf(subs(d=dv, LB2), 20), evalf(q2csB(dv), 20),
     abs(evalf(subs(d=dv, LB2) - q2csB(dv), 25)));
end do:
printf("A(0)=%a (1/2)   A(1/2)-B(1/2)=%a (0)   B(1)=%a (0)\n",
   simplify(limit(LA2, d=0, right)),
   simplify(subs(d=1/2, LA2) - subs(d=1/2, LB2)),
   simplify(limit(LB2, d=1, left)));
printf("join value A(1/2) = %a\n", simplify(subs(d=1/2, LA2)));

# ===========================================================================
# Section 4: 2-D planar, Wendland (by-parts, v0 = Int_0^q t W(t) dt)
# ===========================================================================
print("");
print("=== 4. 2D planar Wendland ===");
Lp2 := table():
run2d := proc(knm, C2v2, Wv)
  local v0, Jrest, L2, dv, nv;
  global Lp2;
  v0 := int(t*Wv(t), t=0..q):
  if not is(simplify(diff(v0, q) - q*Wv(q)) = 0) then
    print("v0' FAIL:", knm);
  end if:
  Jrest := int(v0/(q*sqrt(q^2-d^2)), q=d..1):
  if has(Jrest, Int) then
    print("Jrest NOT CLOSED:", knm);
  end if:
  L2 := 2*C2v2*(subs(q=1, v0)*arccos(d) - d*Jrest):
  Lp2[knm] := L2:
  printf("%s: v0' PASS  Jrest closed  len=%a\n", knm, length(L2));
  for dv in [1/10, 1/3, 1/2, 2/3, 9/10] do
    nv := evalf(int(2*C2v2*Wv(q)*q*arccos(dv/q), q=dv..1), 50):
    printf("  d=%5.3f  diff=%.3e\n", evalf(dv),
       abs(evalf(subs(d=dv, L2) - nv, 25)));
  end do:
  printf("  L2(0)=%a (1/2)   L2(1)=%a (0)\n",
     simplify(limit(L2, d=0, right)), simplify(limit(L2, d=1, left)));
end proc:
run2d("w2", C2t[w2], W2):
run2d("w4", C2t[w4], W4):
run2d("w6", C2t[w6], W6):

print("");
print("01_planar: derivations complete; writing export");

# --- export -------------------------------------------------------------------
# NOTE: \n escapes live INSIDE fprintf format strings (interpreted by fprintf).
# This section is maintained via bash heredoc because the editor/tool layer
# mangles backslash-n in string literals.
EXPORT := "results/symbolic/planar_export.txt":
f := open(EXPORT, WRITE):
fprintf(f, "# planar_export.txt - generated by maple/01_planar.mpl\n"):
fprintf(f, "# unit support h=1, particle outside the boundary, d in [0,1]\n"):
fprintf(f, "# polynomials: coefficients in DESCENDING powers of d (rational)\n"):
fprintf(f, "# expressions: single-line Maple %%a form; translate Pi, arccos, ln, ^\n"):
fprintf(f, "kernel cubic C2=80/7/Pi C3=16/Pi\n"):
fprintf(f, "kernel w2 n=4 P=0:1 1:4 C2=7/Pi C3=21/2/Pi\n"):
fprintf(f, "kernel w4 n=6 P=0:1 1:6 2:35/3 C2=9/Pi C3=495/32/Pi\n"):
fprintf(f, "kernel w6 n=8 P=0:1 1:8 2:25 3:32 C2=78/7/Pi C3=1365/64/Pi\n"):
# 3D: cubic spline branches (paper eq. 18) and Wendland polynomials
fprintf(f, "planar3d cubic b1 deg=%d:", degree(LA3, d)):
for p from degree(LA3, d) to 0 by -1 do
  fprintf(f, " %a", coeff(LA3, d, p)):
end do:
fprintf(f, "\n"):
fprintf(f, "planar3d cubic b2 deg=%d:", degree(LB3, d)):
for p from degree(LB3, d) to 0 by -1 do
  fprintf(f, " %a", coeff(LB3, d, p)):
end do:
fprintf(f, "\n"):
for kv in kerns do
  Lp := Lp3[kv[1]]:
  fprintf(f, "planar3d %s deg=%d:", kv[1], degree(Lp, d)):
  for p from degree(Lp, d) to 0 by -1 do
    fprintf(f, " %a", coeff(Lp, d, p)):
  end do:
  fprintf(f, "\n"):
end do:
# 2D: cubic spline branches A/B (join at d=1/2) and Wendland expressions
fprintf(f, "planar2d cubic A: %a\n", LA2):
fprintf(f, "planar2d cubic B: %a\n", LB2):
fprintf(f, "planar2d cubic join: %a\n", simplify(subs(d=1/2, LA2))):
for kv in kerns do
  fprintf(f, "planar2d %s: %a\n", kv[1], Lp2[kv[1]]):
end do:
close(f):
print("export written to ", EXPORT);
print("01_planar: complete");
