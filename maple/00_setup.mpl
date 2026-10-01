# ===========================================================================
# 00_setup.mpl — shared conventions, kernel table, harness sanity checks.
#
# Conventions (paper eqs. 3-4; warpSPHCore kernels/kernel.py):
#   W(r,h) = C_n / h^n * W_hat(q),  q = r/h,  h = support radius (support on
#   [0,1]).  C_n is chosen so the kernel integrates to 1 over R^n:
#     2D:  2*Pi*C2*Int_0^1 W(q) q   dq = 1
#     3D:  4*Pi*C3*Int_0^1 W(q) q^2 dq = 1
#
# Kernels (warpSPHCore kernelFunctions + scripts/kernels/kernel_specs.yaml;
# shapes are the same in 2D and 3D, only C_n differs):
#   cubic: W(q) = (1-q)^3 - 4*(1/2-q)_+^3      C2 = 80/(7Pi)  C3 = 16/Pi
#   w2:    (1-q)^4  (1 + 4q)                    C2 = 7/Pi      C3 = 21/(2Pi)
#   w4:    (1-q)^6  (1 + 6q + 35/3 q^2)         C2 = 9/Pi      C3 = 495/(32Pi)
#   w6:    (1-q)^8  (1 + 8q + 25q^2 + 32q^3)    C2 = 78/(7Pi)  C3 = 1365/(64Pi)
#
# The Wendland family is a single polynomial in q (no midway knot), so the
# monomial basis
#   F(n,p,x) = Int_0^x t^p (1-t)^n dt
#            = sum_{j=0..n} (-1)^j binomial(n,j) x^(p+j+1) / (p+j+1)
# closes every boundary integral for the Wendland kernels.
# ===========================================================================
restart:

C2t := table([cubic = 80/(7*Pi), w2 = 7/Pi, w4 = 9/Pi, w6 = 78/(7*Pi)]):
C3t := table([cubic = 16/Pi, w2 = 21/(2*Pi), w4 = 495/(32*Pi), w6 = 1365/(64*Pi)]):

Wcs := q -> (1-q)^3 - 4*(1/2-q)^3:   # valid for q <= 1/2
W2  := q -> (1-q)^4*(1+4*q):
W4  := q -> (1-q)^6*(1+6*q+35/3*q^2):
W6  := q -> (1-q)^8*(1+8*q+25*q^2+32*q^3):

# --- kernel normalization -----------------------------------------------------
print("=== kernel normalization (expect 1 everywhere) ===");
n2cs := simplify(2*Pi*C2t[cubic]*
   (int(q*((1-q)^3-4*(1/2-q)^3), q=0..1/2) + int(q*(1-q)^3, q=1/2..1))):
n3cs := simplify(4*Pi*C3t[cubic]*
   (int(q^2*((1-q)^3-4*(1/2-q)^3), q=0..1/2) + int(q^2*(1-q)^3, q=1/2..1))):
printf("cubic  2D: %a  %s    3D: %a  %s\n",
   n2cs, `if`(n2cs = 1, "PASS", "FAIL"),
   n3cs, `if`(n3cs = 1, "PASS", "FAIL"));
for pair in [[w2, W2], [w4, W4], [w6, W6]] do
  a := simplify(2*Pi*C2t[pair[1]]*int(pair[2](q)*q, q=0..1)):
  b := simplify(4*Pi*C3t[pair[1]]*int(pair[2](q)*q^2, q=0..1)):
  printf("%a  2D: %a  %s    3D: %a  %s\n", pair[1],
     a, `if`(a = 1, "PASS", "FAIL"),
     b, `if`(b = 1, "PASS", "FAIL"));
end do:

# --- monomial basis F(n,p,x) ---------------------------------------------------
F := (n,p,x) -> local j;
   add(binomial(n,j)*(-1)^j*x^(p+j+1)/(p+j+1), j=0..n):
print("");
print("=== monomial basis F(n,p,x): verify ===");
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
printf("derivative + definite, n in {4,6,8}, p in 0..10: %s\n",
   `if`(bok, "PASS", "FAIL"));
for t in [[4, 0, 1/3], [6, 2, 2/5], [8, 5, 7/10]] do
  fv := F(t[1], t[2], t[3]):
  nv := evalf(int(q^t[2]*(1-q)^t[1], q=0..t[3]), 50):
  printf("F(%a,%a,%a) = %a   50-dig quad diff = %.2e\n",
     t[1], t[2], t[3], fv, abs(evalf(fv-nv, 25)));
end do:

print("");
print("00_setup: complete");
