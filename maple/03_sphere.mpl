# ===========================================================================
# 03_sphere.mpl — solid-sphere (curved) boundary integrals, Wendland kernels.
# Exact arithmetic; run from the repo root:  maple -q maple/03_sphere.mpl
#
# Geometry: solid ball of radius R, particle OUTSIDE at distance d from the
# surface (D = R + d from the center), unit support h = 1, d in [0,1].
# The radius-q shell intersects the ball in the spherical cap
#   A(q) = Pi*q/D * (R^2 - D^2 + 2*D*q - q^2),   q in [d, 2R+d]
# (for q > 2R+d the shell engulfs the whole obstacle and A(q) = 0 again).
#   lambda_sph(R,d) = Int_d^{qmax} C3 W(q) A(q) dq,
#   qmax = 1          (branch 1: 2R + d >= 1, support truncated by q = 1)
#        2R + d       (branch 2: 2R + d <  1, support truncated by the ball)
#
# Result: both branches are RATIONAL with denominator (R+d):
#   lambda_sph = N(R,d) / (R+d),  N a polynomial in R and d
#   (branch 1: deg_R <= 1; the shell area is linear in R).
# The planar limit R -> infinity reproduces the planar result exactly, and
# the two branches join exactly on 2R + d = 1.
#
# Kernels: warpSPHCore kernelFunctions/wendland{2,4,6}.py (3-D shapes):
#   w2: (1-q)^4 (1+4q)                      C3 = 21/(2 Pi)
#   w4: (1-q)^6 (1+6q+35/3 q^2)             C3 = 495/(32 Pi)
#   w6: (1-q)^8 (1+8q+25q^2+32q^3)          C3 = 1365/(64 Pi)
#
# Output: src/warpSPHBoundaries/data/symbolic/sphere_export.txt
# ===========================================================================
restart:

# --- monomial basis -----------------------------------------------------------
F := (n,p,x) -> local j;
   add(binomial(n,j)*(-1)^j*x^(p+j+1)/(p+j+1), j=0..n):
bok := true:
for n in [4, 6, 8] do
  for p from 0 to 10 do
    if not (is(simplify(diff(F(n,p,x), x) - x^p*(1-x)^n) = 0)
        and is(simplify(F(n,p,1) - F(n,p,d) - int(q^p*(1-q)^n, q=d..1)) = 0)) then
      bok := false:
      print("BASIS FAIL n=", n, " p=", p);
    end if:
  end do:
end do:
printf("F(n,p,x) verified (derivative + definite), n in {4,6,8}, p in 0..10: %s\n",
   `if`(bok, "PASS", "FAIL"));

# --- kernel data (warpSPHCore, 3-D) -------------------------------------------
kerns := [
   ["w2", 4, [[0,1], [1,4]],            21/(2*Pi)],
   ["w4", 6, [[0,1], [1,6], [2,35/3]],  495/(32*Pi)],
   ["w6", 8, [[0,1], [1,8], [2,25], [3,32]], 1365/(64*Pi)]
]:
polyval := (P, x) -> local i; add(P[i][2]*x^P[i][1], i=1..numelems(P)):

# --- spherical shell integral, symbolic upper limit u ---------------------------
Dexpr := R + d:
Aexpr := Pi*q/Dexpr*(R^2 - Dexpr^2 + 2*Dexpr*q - q^2):
Wj := (n, P, j, x) -> local i; add(P[i][2]*F(n, P[i][1]+j, x), i=1..numelems(P)):
sph_gen := (n, P, C3v, u) ->
   Pi*C3v/Dexpr*((R^2 - Dexpr^2)*(Wj(n,P,1,u) - Wj(n,P,1,d))
                 + 2*Dexpr*(Wj(n,P,2,u) - Wj(n,P,2,d))
                 -      (Wj(n,P,3,u) - Wj(n,P,3,d))):
sph_b1 := (n, P, C3v) -> simplify(sph_gen(n, P, C3v, 1)):
sph_b2 := (n, P, C3v) -> simplify(sph_gen(n, P, C3v, 2*R + d)):

# planar reference (from 01_planar.mpl) for the R -> infinity check
planar_mono := (n, P, C3v2) -> local i;
   2*Pi*C3v2*add(P[i][2]*((F(n,P[i][1]+2,1) - F(n,P[i][1]+2,d))
                   - d*(F(n,P[i][1]+1,1) - F(n,P[i][1]+1,d))),
                   i=1..numelems(P)):

print("");
print("=== sphere: identities, limits, joins, 50-digit quadrature ===");
for kv in kerns do
  knm := kv[1]: n := kv[2]: P := kv[3]: C3v := kv[4]:
  Lgd := sph_gen(n, P, C3v, u):
  Lgd_direct := simplify(int(C3v*(1-q)^n*polyval(P,q)*Aexpr, q=d..u)):
  printf("%s: mono(u) == direct int(q=d..u): %s\n",
     knm, `if`(is(simplify(Lgd - Lgd_direct) = 0), "PASS", "FAIL"));
  limok := is(simplify(limit(sph_b1(n, P, C3v), R = infinity)
                      - planar_mono(n, P, C3v)) = 0):
  printf("   b1: limit(R->inf) == planar: %s\n", `if`(limok, "PASS", "FAIL"));
  joinok := is(simplify(subs(R = (1-d)/2, sph_b1(n,P,C3v))
                        - subs(R = (1-d)/2, sph_b2(n,P,C3v))) = 0):
  printf("   join: b2(R=(1-d)/2) == b1(R=(1-d)/2): %s\n",
     `if`(joinok, "PASS", "FAIL"));
  quad_ok := true:
  for Rv in [1/2, 1, 2, 5, 20] do
    for dv in [1/10, 3/10, 7/10] do
      sv := subs({R=Rv, d=dv}, sph_b1(n, P, C3v)):
      Dv := Rv + dv:
      nv := evalf(int(C3v*(1-q)^n*polyval(P,q)*Pi*q/Dv
                      *(Rv^2 - Dv^2 + 2*Dv*q - q^2), q=dv..1), 50):
      if abs(evalf(sv-nv, 30)) > 1e-20 then
        quad_ok := false:
        printf("   B1 QUAD FAIL: %s R=%a d=%a diff=%.3e\n",
           knm, Rv, dv, abs(evalf(sv-nv, 30)));
      end if:
    end do:
  end do:
  printf("   b1 50-digit quadrature (R in {1/2,1,2,5,20} x d in {1/10,3/10,7/10}): %s\n",
     `if`(quad_ok, "PASS", "FAIL"));
  quad_ok := true:
  for Rv in [1/8, 1/4, 3/10, 1/3] do
    for dv in [1/10, 1/5, 1/3, 1/2, 3/5] do
      if 2*Rv + dv >= 1 then continue; end if:
      sv := subs({R=Rv, d=dv}, sph_b2(n, P, C3v)):
      Dv := Rv + dv:
      nv := evalf(int(C3v*(1-q)^n*polyval(P,q)*Pi*q/Dv
                      *(Rv^2 - Dv^2 + 2*Dv*q - q^2), q=dv..(2*Rv+dv)), 50):
      if abs(evalf(sv-nv, 30)) > 1e-20 then
        quad_ok := false:
        printf("   B2 QUAD FAIL: %s R=%a d=%a diff=%.3e\n",
           knm, Rv, dv, abs(evalf(sv-nv, 30)));
      end if:
    end do:
  end do:
  printf("   b2 50-digit quadrature (R in {1/8,1/4,3/10,1/3}, 2R+d<1): %s\n",
     `if`(quad_ok, "PASS", "FAIL"));
end do:

print("");
print("03_sphere: derivations complete; writing export");

# --- export -------------------------------------------------------------------
# NOTE: \n escapes live INSIDE fprintf format strings (interpreted by fprintf).
# This section is maintained via bash heredoc because the editor/tool layer
# mangles backslash-n in string literals.
EXPORT := "src/warpSPHBoundaries/data/symbolic/sphere_export.txt":
f := open(EXPORT, WRITE):
fprintf(f, "# sphere_export.txt - generated by maple/03_sphere.mpl\n"):
fprintf(f, "# solid ball radius R, particle outside, unit support h=1, d in [0,1]\n"):
fprintf(f, "# branch 1: 2R+d >= 1 (qmax = 1);  branch 2: 2R+d < 1 (qmax = 2R+d)\n"):
fprintf(f, "# lambda_sph(R,d) = N(R,d)/(R+d);  N = sum c R^i d^j, entries i:j:c\n"):
for kv in kerns do
  knm := kv[1]: n := kv[2]: P := kv[3]: C3v := kv[4]:
  fprintf(f, "kernel %s n=%d P=", knm, n):
  for i from 1 to numelems(P) do
    fprintf(f, "%d:%a ", P[i][1], P[i][2]):
  end do:
  fprintf(f, "C3=%a\n", C3v):
  Ls1 := expand(sph_b1(n, P, C3v)):
  numv1 := expand(normal(Ls1*(R + d))):
  dr1 := degree(numv1, R):
  if dr1 = false or not is(simplify(Ls1 - numv1/(R+d)) = 0) then
    print("EXPORT B1 CHECK FAIL:", knm);
  else
    fprintf(f, "sph b1 %s:", knm):
    for i from dr1 to 0 by -1 do
      cd := simplify(collect(expand(coeff(numv1, R, i)), d)):
      if cd = 0 then continue; end if:
      for p from degree(cd, d) to 0 by -1 do
        c := coeff(cd, d, p):
        if c = 0 then continue; end if:
        fprintf(f, " %d:%d:%a", i, p, c):
      end do:
    end do:
    fprintf(f, "\n"):
  end if:
  Ls2 := expand(sph_b2(n, P, C3v)):
  numv2 := expand(normal(Ls2*(R + d))):
  dr2 := degree(numv2, R):
  if dr2 = false or not is(simplify(Ls2 - numv2/(R+d)) = 0) then
    print("EXPORT B2 CHECK FAIL:", knm);
  else
    fprintf(f, "sph b2 %s:", knm):
    for i from dr2 to 0 by -1 do
      cd := simplify(collect(expand(coeff(numv2, R, i)), d)):
      if cd = 0 then continue; end if:
      for p from degree(cd, d) to 0 by -1 do
        c := coeff(cd, d, p):
        if c = 0 then continue; end if:
        fprintf(f, " %d:%d:%a", i, p, c):
      end do:
    end do:
    fprintf(f, "\n"):
  end if:
end do:
close(f):
print("export written to ", EXPORT);
print("03_sphere: complete");
