# ===========================================================================
# 13_halfplane.mpl -- HALF-PLANE cross-check between the two tracks:
#   the edge value / gradient / first-moment identity with ONE infinite edge at
#   distance d  must equal  lambda_2(d), -d lambda_2/dd  (docs/derivation.md s.3,
#   results/symbolic/planar_export.txt) for all four kernels.
#
#   x outside the solid {y > d}:  n = (0,-1), z = -d, indicator 0, chord [-L,L], L = sqrt(R^2-d^2)
#   value   = sum_blocks sum_n c_n/(n+2) [ -2 d I_n(L) + 2 R^(n+2) arccos(d/R) ]        (R = 1, 1/2)
#   -dvalue/dd = 2 sum_blocks sum_n c_n I_n(L)
#   m_(0,1) = - int_chord Phi ds   (compact-potential recursion k=1)  vs  2 int_d^1 W r sqrt(r^2-d^2) dr
#
# Each comparison is attempted SYMBOLICALLY in d (0<d<1, resp. 0<d<1/2 and 1/2<d<1 for
# the cubic) and ALSO checked numerically at 50 digits as a fallback.
# ===========================================================================
restart:
Digits := 50:

chk := proc(label, ok)
  printf("%-72s %s\n", label, `if`(ok, "PASS", "FAIL"));
end proc:

# ---- primitives (verified in 10_edge_primitives.mpl) ---------------------------
Ip := proc(m::integer) option remember;
  if m = 0 then s
  elif m = -1 then arcsinh(s/abs(z))
  elif m = -2 then arctan(s/z)/z
  else (s*(s^2+z^2)^(m/2) + m*z^2*Ip(m-2))/(m+1) end if end proc:
# I_n(L) at z = d (|z| = d; I_n even in z) with L = sqrt(R^2-d^2)
IL := proc(n, Rv) subs([s = sqrt(Rv^2 - d^2), z = d], Ip(n)) end proc:

# ---- kernels as blocks: [R, [c_0, c_1, ...]] including C2 (pi-free coefficients below carry 1/Pi) ----
poly_coeffs := proc(pol) local k; [seq(coeff(expand(pol), q, k), k = 0 .. degree(expand(pol), q))] end proc:
C2 := table([cubic = 80/7, w2 = 7, w4 = 9, w6 = 78/7]):       # C2 = value/Pi
blocks := table([
  cubic = [ [1, poly_coeffs(C2[cubic]*(1-q)^3)], [1/2, poly_coeffs(C2[cubic]*(-4)*(1/2-q)^3)] ],
  w2 = [ [1, poly_coeffs(C2[w2]*(1-q)^4*(1+4*q))] ],
  w4 = [ [1, poly_coeffs(C2[w4]*(1-q)^6*(1+6*q+35/3*q^2))] ],
  w6 = [ [1, poly_coeffs(C2[w6]*(1-q)^8*(1+8*q+25*q^2+32*q^3))] ] ]):

# edge value for x outside at distance d:  (1/Pi) sum_blocks [d < R]  ...
edge_value := proc(kname, dmax_branch) local tot, b, n, Rv, cs;
  tot := 0;
  for b in blocks[kname] do
    Rv := b[1]; cs := b[2];
    if evalb(dmax_branch < Rv) then   # block active (d < R): branch chosen by a sample d
      for n from 0 to nops(cs)-1 do
        if cs[n+1] <> 0 then
          tot := tot + cs[n+1]/(n+2)*( -2*d*IL(n, Rv) + 2*Rv^(n+2)*arccos(d/Rv) );
        end if;
      end do;
    end if;
  end do;
  tot/Pi
end proc:
edge_grad := proc(kname, dmax_branch) local tot, b, n, Rv, cs;
  tot := 0;
  for b in blocks[kname] do
    Rv := b[1]; cs := b[2];
    if evalb(dmax_branch < Rv) then
      for n from 0 to nops(cs)-1 do
        if cs[n+1] <> 0 then tot := tot + cs[n+1]*2*IL(n, Rv); end if;
      end do;
    end if;
  end do;
  tot/Pi
end proc:
# first moment m_(0,1):  - int_chord Phi ds, Phi = (r^(n+2) - R^(n+2))/(n+2) per monomial
edge_m01 := proc(kname, dmax_branch) local tot, b, n, Rv, cs;
  tot := 0;
  for b in blocks[kname] do
    Rv := b[1]; cs := b[2];
    if evalb(dmax_branch < Rv) then
      for n from 0 to nops(cs)-1 do
        if cs[n+1] <> 0 then
          tot := tot - cs[n+1]/(n+2)*( 2*IL(n+2, Rv) - 2*Rv^(n+2)*sqrt(Rv^2 - d^2) );
        end if;
      end do;
    end if;
  end do;
  tot/Pi
end proc:

# ---- load the closed forms lambda_2 from the PLAN track's export ------------------
file := "results/symbolic/planar_export.txt":
lam := table([]):
fd := fopen(file, READ, TEXT):
line := readline(fd):
while line <> 0 do
  if searchtext("planar2d ", line) = 1 then
    pos := searchtext(": ", line);
    key := substring(line, 10 .. pos-1);                  # "cubic A", "cubic B", "cubic join", "w2", ...
    expr := substring(line, pos+2 .. length(line));
    lam[key] := parse(expr);
  end if;
  line := readline(fd);
end do:
fclose(fd):
printf("loaded planar2d keys: %a\n", [indices(lam, 'nolist')]);

# kernel radial profile pieces for the direct 1D integrals (independent of the blocks)
Wsrc := table([
  w2 = (r -> 7/Pi*(1-r)^4*(1+4*r)),
  w4 = (r -> 9/Pi*(1-r)^6*(1+6*r+35/3*r^2)),
  w6 = (r -> 78/(7*Pi)*(1-r)^8*(1+8*r+25*r^2+32*r^3)) ]):

# helper: compare two expressions symbolically (after ln-conversion) and numerically
cmp_sym_num := proc(A, B, dpts)
  local diffexpr, sym, dp, worst, f;
  diffexpr := convert(A - B, ln);
  f := proc() simplify(diffexpr) assuming d > 0, d < 1 end proc;
  sym := "unsimplified":
  try
    sym := timelimit(120, f());
    sym := evalb(sym = 0);
  catch "time expired":
    sym := "time expired";
  catch:
    sym := "error";
  end try:
  worst := 0;
  for dp in dpts do worst := max(worst, abs(evalf(subs(d = dp, A - B)))) end do;
  [sym, worst]
end proc:

dpts_all := [1/1000, 1/20, 1/10, 3/10, 1/2 - 1/1000, 1/2 + 1/1000, 7/10, 9/10, 999/1000]:
dpts_lo  := [1/1000, 1/20, 1/10, 3/10, 1/2 - 1/1000]:
dpts_hi  := [1/2 + 1/1000, 7/10, 9/10, 999/1000]:

# ------------------------------------------------------------ value vs lambda_2 -------------
for k in [w2, w4, w6] do
  A := edge_value(k, 1/2):                  # d < 1: R = 1 block only (sample branch 1/2 < 1)
  res := cmp_sym_num(A, lam[sprintf("%a", k)], dpts_all):
  chk(sprintf("%a: edge value (half-plane) = lambda_2(d): symbolic %a, worst numeric %.2e", k, res[1], res[2]), evalb(res[2] < 1e-45)):
end do:
# cubic: branch A (d < 1/2): blocks R=1 and R=1/2 active; branch B (d > 1/2): R=1 only
A := edge_value(cubic, 1/4):
res := cmp_sym_num(A, lam["cubic A"], dpts_lo):
chk(sprintf("cubic A (d<1/2): edge value = lambda_2: symbolic %a, worst numeric %.2e", res[1], res[2]), evalb(res[2] < 1e-45));
A := edge_value(cubic, 3/4):
res := cmp_sym_num(A, lam["cubic B"], dpts_hi):
chk(sprintf("cubic B (d>1/2): edge value = lambda_2: symbolic %a, worst numeric %.2e", res[1], res[2]), evalb(res[2] < 1e-45));

# ------------------------------------------------------------ gradient = -d lambda_2/dd ---------
for k in [w2, w4, w6] do
  G := edge_grad(k, 1/2):
  dl := -diff(lam[sprintf("%a", k)], d):
  res := cmp_sym_num(G, dl, dpts_all):
  chk(sprintf("%a: edge gradient = -dlambda_2/dd: symbolic %a, worst numeric %.2e", k, res[1], res[2]), evalb(res[2] < 1e-45));
end do:
G := edge_grad(cubic, 1/4): dl := -diff(lam["cubic A"], d):
res := cmp_sym_num(G, dl, dpts_lo):
chk(sprintf("cubic A: edge gradient = -dlambda_2/dd: symbolic %a, worst numeric %.2e", res[1], res[2]), evalb(res[2] < 1e-45));
G := edge_grad(cubic, 3/4): dl := -diff(lam["cubic B"], d):
res := cmp_sym_num(G, dl, dpts_hi):
chk(sprintf("cubic B: edge gradient = -dlambda_2/dd: symbolic %a, worst numeric %.2e", res[1], res[2]), evalb(res[2] < 1e-45));
# the join d = 1/2 is continuous for lambda_2 AND for the gradient (cubic spline is C^1 across the knot in d)
j1 := evalf(subs(d = 1/2, edge_value(cubic, 1/4))): j2 := evalf(subs(d = 1/2, edge_value(cubic, 3/4))):
chk("cubic: value branches A and B agree at d = 1/2 (and equal the export join value)",
    evalb(abs(j1 - j2) < 1e-45 and abs(j1 - evalf(lam["cubic join"])) < 1e-45));

# ------------------------------------------------------------ first moment m_(0,1) -------------
# direct reference (independent of the edge machinery): 2 int_d^1 W(r) r sqrt(r^2-d^2) dr
for k in [w2, w4, w6] do
  ref := 2*int(Wsrc[k](r)*r*sqrt(r^2 - d^2), r = d .. 1) assuming d > 0, d < 1:
  M := edge_m01(k, 1/2):
  res := cmp_sym_num(M, ref, dpts_all):
  chk(sprintf("%a: edge m_(0,1) = 2 int W r sqrt(r^2-d^2): symbolic %a, worst numeric %.2e", k, res[1], res[2]), evalb(res[2] < 1e-45));
end do:
Wc_in := r -> 80/(7*Pi)*(1/2 - 3*r^2 + 3*r^3):
Wc_out := r -> 80/(7*Pi)*(1-r)^3:
refA := 2*(int(Wc_in(r)*r*sqrt(r^2-d^2), r = d .. 1/2) + int(Wc_out(r)*r*sqrt(r^2-d^2), r = 1/2 .. 1)) assuming d > 0, d < 1/2:
refB := 2*int(Wc_out(r)*r*sqrt(r^2-d^2), r = d .. 1) assuming d > 1/2, d < 1:
res := cmp_sym_num(edge_m01(cubic, 1/4), refA, dpts_lo):
chk(sprintf("cubic A: edge m_(0,1) = direct radial integral: symbolic %a, worst numeric %.2e", res[1], res[2]), evalb(res[2] < 1e-45));
res := cmp_sym_num(edge_m01(cubic, 3/4), refB, dpts_hi):
chk(sprintf("cubic B: edge m_(0,1) = direct radial integral: symbolic %a, worst numeric %.2e", res[1], res[2]), evalb(res[2] < 1e-45));

print("13_halfplane: complete");
