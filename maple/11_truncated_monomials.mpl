# ===========================================================================
# 11_truncated_monomials.mpl -- docs/derivations/truncated-monomials.md
#   (HANDOFF.md section 4, task 12.1 second half)
#
#  A. kernel -> truncated monomials: pointwise + normalisation for cubic, w2, w4, w6
#  B. per-block value formula (indicator + [z I_n - R^(n+2) atan(s/z)]/(n+2)):
#       B1. divergence step  div(y M(r)/r^2) = r^n  (r<R), far field div-free
#       B2. half-plane, SYMBOLIC in d and R (both signs of z), n = 0..12
#       B3. wedge with x at a vertex vs polar reference, all clip cases (40 digits)
#  C. per-block gradient:  -d/dd (half-plane value) = 2 I_n(L)  (symbolic, n=0..12)
#
# Gotcha: derivatives/substitutions are done before `assuming` (it renames vars).
# ===========================================================================
restart:

chk := proc(label, ok)
  printf("%-66s %s\n", label, `if`(ok, "PASS", "FAIL"));
end proc:

r2 := s^2 + z^2:
Ip := proc(m::integer)           # odd antiderivative of r^m (m >= -2), verified in 10_*
  option remember;
  if m = 0 then s
  elif m = -1 then arcsinh(s/abs(z))
  elif m = -2 then arctan(s/z)/z
  else (s*r2^(m/2) + m*z^2*Ip(m-2))/(m+1)
  end if
end proc:
# numeric/symbolic evaluation of I_n at (sv, zv)
IE := proc(n, sv, zv) subs([s=sv, z=zv], Ip(n)) end proc:

# ------------------------------------------------------------------ A. kernels --
# each block = [coefficient list in q^0..q^deg, R]
C2t := table([cubic = 80/(7*Pi), w2 = 7/Pi, w4 = 9/Pi, w6 = 78/(7*Pi)]):
What := table([
  cubic = (q -> (1-q)^3 - 4*(1/2-q)^3),            # valid for q <= 1/2
  w2 = (q -> (1-q)^4*(1+4*q)),
  w4 = (q -> (1-q)^6*(1+6*q+35/3*q^2)),
  w6 = (q -> (1-q)^8*(1+8*q+25*q^2+32*q^3)) ]):
# blocks: (outer piece, R=1) and, for the cubic, (piece_in - piece_out, R=1/2)
poly_blocks := table([
  cubic = [ [ (1-q)^3, 1 ], [ -4*(1/2-q)^3, 1/2 ] ],
  w2 = [ [ (1-q)^4*(1+4*q), 1 ] ],
  w4 = [ [ (1-q)^6*(1+6*q+35/3*q^2), 1 ] ],
  w6 = [ [ (1-q)^8*(1+8*q+25*q^2+32*q^3), 1 ] ] ]):

# pointwise: sum_j q_j 1[q<=R_j] = W_hat on [0,1/2], = outer piece on (1/2,1]
ok := true:
# cubic: inner region q<=1/2 : both blocks; outer region: only block 1
cin  := expand(add(b[1], b in poly_blocks[cubic]) - What[cubic](q)):
cout := expand(poly_blocks[cubic][1][1] - (1-q)^3):
ok := evalb(cin = 0 and cout = 0):
# the piece_in of the paper is (1-q)^3-4(1/2-q)^3 = 1/2 - 3q^2 + 3q^3 (HANDOFF/derivation.md remark)
ok := ok and evalb(expand(What[cubic](q) - (1/2 - 3*q^2 + 3*q^3)) = 0):
chk("cubic: (1-q)^3 + (-4)(1/2-q)^3 1[q<=1/2] = W_hat, 1/2-3q^2+3q^3", ok);
for k in [w2, w4, w6] do
  e := expand(poly_blocks[k][1][1] - What[k](q)):
  chk(sprintf("%a: single block R=1 equals W_hat pointwise", k), evalb(e = 0));
end do:
# HANDOFF.md form: W = sigma [2(1-q)^3 - 8(1/2-q)^3 1[q<=1/2]], sigma = 40/(7 Pi) = C2/2
sg := 40/(7*Pi):
e := expand(sg*(2*(1-q)^3 + (-8)*(1/2-q)^3) - C2t[cubic]*(poly_blocks[cubic][1][1] + poly_blocks[cubic][2][1])):
chk("cubic: HANDOFF form sigma[2(1-q)^3 - 8(1/2-q)^3] = C2*(blocks)", evalb(simplify(e) = 0));
# The (unverified) script monomial_edge_forms.py uses -4*sigma*(1/2-q)^3 for the R=1/2 block:
e_bug := expand(sg*(2*(1-q)^3 + (-4)*(1/2-q)^3) - C2t[cubic]*(poly_blocks[cubic][1][1] + poly_blocks[cubic][2][1])):
chk("cubic: script's  -4*sigma*(1/2-q)^3  is NOT the kernel (expect nonzero)", evalb(simplify(e_bug) <> 0));

# normalisation from blocks:  2 pi C2 sum_j int_0^{R_j} q q_j(q) dq = 1
for k in [cubic, w2, w4, w6] do
  tot := 2*Pi*C2t[k]*add(int(q*b[1], q=0..b[2]), b in poly_blocks[k]):
  chk(sprintf("%a: 2D normalisation from blocks = 1", k), evalb(simplify(tot) = 1));
end do:
# Indicator weights per support radius: w_j = 2 pi C2 M_j(R_j); they sum to 1.
printf("cubic indicator weights per radius R=1, R=1/2: %a, %a\n",
  simplify(2*Pi*C2t[cubic]*int(q*poly_blocks[cubic][1][1], q=0..1)),
  simplify(2*Pi*C2t[cubic]*int(q*poly_blocks[cubic][2][1], q=0..1/2)));

# ------------------------------------------------------- B1. divergence step --
rr := sqrt(x^2 + y^2):
ok := true:
for n from 0 to 12 do
  Min := rr^(n+2)/(n+2):                     # M(r), r <= R
  F := [x*Min/rr^2, y*Min/rr^2]:
  dv := diff(F[1], x) + diff(F[2], y):
  e := simplify(dv - rr^n) assuming x::real, y::real, x^2+y^2 > 0;
  # far field (r > R): M = R^(n+2)/(n+2) constant: flux y/r^2 is divergence free
  G := [x/rr^2, y/rr^2]:
  e2 := simplify(diff(G[1], x) + diff(G[2], y)) assuming x::real, y::real, x^2+y^2 > 0;
  if e <> 0 or e2 <> 0 then ok := false; print("FAIL div", n, e, e2); end if:
end do:
chk("div(y M(r)/r^2) = r^n inside, div(y/r^2) = 0 outside, n=0..12", ok);

# --------------------------------------- B2. half-plane, symbolic in d and R --
# the atan jump over the chord: atan(L/d) - atan(-L/d) = 2 atan(L/d) = 2 arccos(d/R)
dd := diff(arctan(sqrt(R^2-d^2)/d) - arccos(d/R), R):
ddz := simplify(dd) assuming d::positive, R > d:
lim0 := limit(arctan(sqrt(R^2-d^2)/d), R=d, left) assuming d::positive:
chk("atan(L/d) = arccos(d/R): derivative in R is 0 and equality at R=d", evalb(ddz = 0 and lim0 = 0));
# x at the origin, solid {y > d}: edge line at distance d, outward normal (0,-1)
# => z = n.(p-x) = -d (x OUTSIDE).  chord s in [-L, L], L = sqrt(R^2-d^2).
# value = 0*indicator + (1/(n+2)) [ z (I_n(L)-I_n(-L)) - R^(n+2) (atan(L/z) - atan(-L/z)) ]
# polar reference: V(R) = int_d^R r^n 2 r arccos(d/r) dr   (so dV/dR = 2 R^(n+1) arccos(d/R))
ok := true: okE := true:
for n from 0 to 12 do
  In := Ip(n):
  # z = -d: z*I_n(L)-I_n(-L) with I_n odd in s -> 2 z I_n(L) ; atan jump -> -2 atan(L/z)... keep literal
  hi := sqrt(R^2 - d^2): lo := -hi:
  # raw edge formula, z = -d
  Eraw := (1/(n+2))*( (-d)*(subs([s=hi, z=-d], In) - subs([s=lo, z=-d], In))
                      - R^(n+2)*(-2*arccos(d/R)) ):
  dE := diff(Eraw, R):
  e := simplify(dE - 2*R^(n+1)*arccos(d/R)) assuming d::positive, R > d:
  # value at the lower limit R = d
  E0 := simplify(subs(R=d, Eraw)) assuming d::positive:
  if e <> 0 or E0 <> 0 then ok := false; print("FAIL halfplane", n, e, E0); end if:
end do:
chk("half-plane (x outside, z=-d): d/dR edge formula = polar, E(R=d)=0, n=0..12", ok);

# x INSIDE the solid at depth d (z = +d, indicator 1):  value = 2 pi R^(n+2)/(n+2) + edge(z=+d)
# polar: = full disk - half-plane at distance d  (the solid contains x: the removed part is {y < -d}...)
ok := true:
for n from 0 to 12 do
  In := Ip(n):
  hi := sqrt(R^2 - d^2): lo := -hi:
  Ein := 2*Pi*R^(n+2)/(n+2) + (1/(n+2))*( d*(subs([s=hi, z=d], In) - subs([s=lo, z=d], In))
                      - R^(n+2)*(2*arccos(d/R)) ):
  Eout := (1/(n+2))*( (-d)*(subs([s=hi, z=-d], In) - subs([s=lo, z=-d], In))
                      - R^(n+2)*(-2*arccos(d/R)) ):
  e := simplify(Ein + Eout - 2*Pi*R^(n+2)/(n+2)) assuming d::positive, R > d:
  if e <> 0 then ok := false; print("FAIL inside", n, e); end if:
end do:
chk("half-plane x inside (z=+d, indicator 1): value + value(outside) = full disk", ok);

# ------------------------------------------------ B3. wedge at a vertex, polar --
# T = triangle (x, A, B), x the vertex at the origin; far edge AB at distance z>0,
# edge ends at s0 < s1.  Edges through x have z=0 and contribute 0 (atan(s/z):=0),
# indicator = (phi1-phi0)/(2 pi).  Polar reference with radial breakpoint at R.
Eval_block := proc(n, R, z, s0, s1)   # edge formula for the far edge only (+ vertex indicator)
  local L, lo, hi, ang, edge;
  ang := arctan(s1/z) - arctan(s0/z);
  edge := 0;
  if z < R then
    L := sqrt(R^2 - z^2);
    lo := `if`(evalf(s0) > evalf(-L), s0, -L); hi := `if`(evalf(s1) < evalf(L), s1, L);
    if evalf(lo) < evalf(hi) then
      edge := (1/(n+2))*( z*(IE(n, hi, z) - IE(n, lo, z)) - R^(n+2)*(arctan(hi/z) - arctan(lo/z)) );
    end if;
  end if;
  ang/(2*Pi)*2*Pi*R^(n+2)/(n+2) + edge
end proc:
Polar_block := proc(n, R, z, s0, s1)
  local p0, p1, pL, pieces, a, b, tot, f, cuts;
  p0 := arctan(s0/z); p1 := arctan(s1/z);
  if z >= R then return (p1 - p0)*R^(n+2)/(n+2) end if;
  pL := arccos(z/R);
  cuts := sort(select(c -> evalf(p0) < evalf(c) and evalf(c) < evalf(p1), [-pL, pL]), (u,v) -> evalf(u) < evalf(v));
  cuts := [p0, op(cuts), p1];
  tot := 0;
  for a from 1 to nops(cuts)-1 do
    b := (cuts[a] + cuts[a+1])/2;
    if evalb(abs(evalf(b)) < evalf(pL)) then   # inside the R-disk: rho = z sec(phi) < R
      tot := tot + Int((z/cos(phi))^(n+2)/(n+2), phi=cuts[a]..cuts[a+1]);
    else
      tot := tot + (cuts[a+1] - cuts[a])*R^(n+2)/(n+2);
    end if;
  end do;
  tot
end proc:
worst := 0: cnt := 0:
for n in [0,1,2,3,4,5,6,8,11] do
  for Rv in [1, 1/2] do
    for zv in [3/10, 1/20, 7/10, 3/2] do
      for ss in [[-2,2],[-1/2,2],[-2,1/2],[1/10,1/2],[1/5,3],[-3,-2/5],[6/5,3],[-1/3,1/3],[-9/10,1/10]] do
        a := Eval_block(n, Rv, zv, ss[1], ss[2]):
        b := Polar_block(n, Rv, zv, ss[1], ss[2]):
        df := abs(evalf(a - b, 40)):
        cnt := cnt + 1:
        if df > worst then worst := df; end if:
      end do:
    end do:
  end do:
end do:
printf("wedge-at-vertex: %d configurations, worst |edge - polar| = %.3e\n", cnt, worst);
chk("wedge at vertex vs polar reference (all clip cases, 40 digits)", evalb(worst < 1e-30 and cnt > 100));

# ------------------------------------------------------- C. gradient per block --
# half-plane: x moves toward the wall: d_x[y] V = -dV/dd; edge identity -n * int_chord r^n ds
# with n=(0,-1) gives +2 I_n(L).
ok := true:
for n from 0 to 12 do
  In := Ip(n):
  hi := sqrt(R^2 - d^2): lo := -hi:
  Eraw := (1/(n+2))*( (-d)*(subs([s=hi, z=-d], In) - subs([s=lo, z=-d], In))
                      - R^(n+2)*(-2*arccos(d/R)) ):
  dEd := diff(Eraw, d):
  gradedge := 2*subs([s=hi, z=d], In):
  e := simplify(-dEd - gradedge) assuming d::positive, R > d:
  if e <> 0 then ok := false; print("FAIL grad", n, e); end if:
end do:
chk("half-plane per-block gradient: -dV/dd = 2 I_n(L), n=0..12 (R>d symbolic)", ok);

print("11_truncated_monomials: complete");
