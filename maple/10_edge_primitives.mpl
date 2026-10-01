# ===========================================================================
# 10_edge_primitives.mpl -- verify the closed-form edge primitives of
# docs/derivations/edge-primitives.md  (HANDOFF.md section 5, task 12.1).
#
#   r = sqrt(s^2+z^2)
#   I_m(s)   = int r^m ds          (odd antiderivative, I_m(0)=0)
#   J_m(s)   = int s r^m ds = r^(m+2)/(m+2),  J_{-2} = ln r
#   S_{j,m}  = int s^j r^m ds      (via s^2 = r^2 - z^2)
#
# Checks: d/ds = integrand (m = -12..12, incl. the downward range m < -2 used
# only by the far-field form (b)), recurrence, oddness, closed forms for even m
# (polynomial) and odd m (asinh tail), S_{j,m} for j = 0..7, z -> 0 behaviour.
#
# Gotcha: `diff(F, s) assuming s::real` returns 0 (the assumption renames s),
# so derivatives are always taken OUTSIDE `assuming` (helper dchk below).
# ===========================================================================
restart:

r2 := s^2 + z^2:

# --- I_m: base cases + recurrence (upward for m >= 1, downward for m <= -3) --
Ip := proc(m::integer)
  option remember;
  if m = 0 then s
  elif m = -1 then arcsinh(s/abs(z))      # |z|: I_m is even in z
  elif m = -2 then arctan(s/z)/z
  elif m > 0 then (s*r2^(m/2) + m*z^2*Ip(m-2))/(m+1)
  else ((m+3)*Ip(m+2) - s*r2^((m+2)/2))/((m+2)*z^2)
  end if
end proc:

Jp := proc(m::integer)
  if m = -2 then ln(r2)/2 else r2^((m+2)/2)/(m+2) end if
end proc:

Sp := proc(j::nonnegint, m::integer)
  local h, i;
  if j mod 2 = 0 then
    h := j/2;
    add(binomial(h,i)*(-z^2)^(h-i)*Ip(m+2*i), i=0..h)
  else
    h := (j-1)/2;
    add(binomial(h,i)*(-z^2)^(h-i)*Jp(m+2*i), i=0..h)
  end if
end proc:

# d/ds F == target ?  (derivative first, simplification under assumptions after)
dchk := proc(F, target)
  local d1, e;
  d1 := diff(F, s);
  e := simplify(d1 - target) assuming s::real, z::positive;
  evalb(e = 0)
end proc:

chk := proc(label, ok)
  printf("%-62s %s\n", label, `if`(ok, "PASS", "FAIL"));
end proc:

# --- 1. d/ds I_m = r^m ----------------------------------------------------------
ok := true:
for m from -12 to 12 do
  if not dchk(Ip(m), r2^(m/2)) then ok := false; print("FAIL I_m", m); end if:
end do:
chk("d/ds I_m = r^m,  m = -12..12  (z>0)", ok);

# z<0: the formulas use |z| in asinh and 1/z in I_{-2}; I_m is EVEN in z for every
# m (only z^2, |z|, and 1/z*arctan(s/z) occur); z*I_{-2} = arctan(s/z) is z-odd.
ok := true:
for m from -12 to 12 do
  Ineg := subs(z=-w, Ip(m)):          # raw formula evaluated at negative z = -w
  d1 := diff(Ineg, s):
  e := simplify(d1 - (s^2+w^2)^(m/2)) assuming s::real, w::positive:
  if e <> 0 then ok := false; print("FAIL neg z", m, e); end if:
end do:
chk("d/ds I_m = r^m,  z = -w < 0 with asinh(s/|z|), m=-12..12", ok);

# --- 2. recurrence (m+1) I_m = s r^m + m z^2 I_{m-2}  ---------------------------
# a) identity of antiderivatives: derivatives agree (m = -1..12)
ok := true:
for m from -1 to 12 do
  if not dchk((m+1)*Ip(m) - s*r2^(m/2) - m*z^2*Ip(m-2), 0) then ok := false; print("FAIL rec", m); end if:
end do:
chk("recurrence (m+1)I_m = s r^m + m z^2 I_{m-2}: d/ds, m=-1..12", ok);
# b) exact equality (constant fixed by oddness): both sides vanish at s=0
ok := true:
for m from 1 to 12 do
  e := simplify(subs(s=0, (m+1)*Ip(m) - s*r2^(m/2) - m*z^2*Ip(m-2))) assuming z::positive;
  if e <> 0 then ok := false; end if:
end do:
chk("recurrence holds with constant 0 (all terms vanish at s=0)", ok);

# Gotcha: subs/diff must happen before `assuming` (it renames s, z).
# --- 3. oddness I_m(-s) = -I_m(s) -----------------------------------------------
ok := true:
for m from -10 to 12 do
  sm := subs(s=-s, Ip(m)) + Ip(m):
  e := simplify(sm) assuming s::real, z::positive:
  if e <> 0 then ok := false; print("FAIL odd", m, e); end if:
end do:
chk("I_m odd in s,  m = -10..12", ok);

# --- 4. closed forms ------------------------------------------------------------
# even m = 2k >= 0: polynomial
ok := true:
for k from 0 to 6 do
  Ipoly := add(binomial(k,i)*z^(2*(k-i))*s^(2*i+1)/(2*i+1), i=0..k):
  e := simplify(Ip(2*k) - Ipoly) assuming s::real, z::positive:
  if e <> 0 or degree(expand(Ip(2*k)), s) = false then ok := false; print("FAIL even", k, e); end if:
end do:
chk("even m=2k: I_m = sum C(k,i) z^(2k-2i) s^(2i+1)/(2i+1)  (k=0..6)", ok);

# odd m >= 1: I_m = c_m z^(m+1) asinh(s/z) + s*r*poly(s^2,z^2),  c_m = m!!/(m+1)!!
ok := true:
for m in [1,3,5,7,9,11] do
  cm := doublefactorial(m)/doublefactorial(m+1):
  tail := expand(Ip(m) - cm*z^(m+1)*arcsinh(s/abs(z))):
  q := simplify(tail/(s*sqrt(r2))) assuming s::positive, z::positive:
  if has(tail, arcsinh) or degree(expand(q), s) = false then
    ok := false; print("FAIL odd closed form", m, tail);
  end if:
end do:
chk("odd m: I_m = m!!/(m+1)!! z^(m+1) asinh(s/z) + s r poly(s^2,z^2)", ok);

# --- 5. J_m ----------------------------------------------------------------------
ok := true:
for m from -2 to 12 do
  if not dchk(Jp(m), s*r2^(m/2)) then ok := false; print("FAIL J_m", m); end if:
end do:
chk("d/ds J_m = s r^m,  m = -2..12   (J_-2 = ln r)", ok);

# --- 6. S_{j,m} = int s^j r^m ds -----------------------------------------------
ok := true: cnt := 0:
for j from 0 to 7 do
  for m from -8 to 10 do
    cnt := cnt + 1:
    if not dchk(Sp(j,m), s^j*r2^(m/2)) then ok := false; print("FAIL S_{j,m}", j, m); end if:
  end do:
end do:
chk(sprintf("d/ds S_{j,m} = s^j r^m, j=0..7, m=-8..10 (%d cases)", cnt), ok);

# definite integrals vs Maple's own numerical Int (40 digits); chords above,
# spanning and below s = 0
ok := true:
for jm in [[0,3],[1,3],[2,2],[2,5],[3,4],[4,-3],[5,-4],[3,-2],[2,-1],[0,-2],[1,-2],[6,8]] do
  j := jm[1]: m := jm[2]:
  F := Sp(j,m):
  for ab in [[1/3, 5/4], [-7/5, 9/10], [-2, -1/2]] do
    got := evalf(eval(F, [s=ab[2], z=3/2]) - eval(F, [s=ab[1], z=3/2]), 40):
    ref := evalf(Int(s^j*(s^2+9/4)^(m/2), s=ab[1]..ab[2]), 40):
    if abs(got - ref) > 1e-30 then ok := false; print("FAIL spot", j, m, ab, got, ref); end if:
  end do:
end do:
chk("S_{j,m} definite integrals vs evalf Int (40 digits)", ok);

# --- 7. z -> 0 --------------------------------------------------------------------
# z^2 I_{-1} -> 0 ; z I_{-2} = arctan(s/z) -> +-Pi/2 (z -> 0+-) ; odd-m I_m regular
l1 := limit(z^2*arcsinh(s/z), z=0, right) assuming s::positive:
l2 := limit(arctan(s/z), z=0, right) assuming s::positive:
l3 := limit(arctan(s/z), z=0, left) assuming s::positive:
ok := evalb(l1 = 0 and l2 = Pi/2 and l3 = -Pi/2):
for m in [1,3,5,7,9,11] do
  lm := limit(Ip(m), z=0, right) assuming s::positive:
  e := simplify(lm - s^(m+1)/(m+1)) assuming s::positive:
  if e <> 0 then ok := false; print("FAIL z->0", m, lm); end if:
end do:
chk("z->0: z^2 I_-1 -> 0, z I_-2 -> +-Pi/2, odd-m I_m -> s^(m+1)/(m+1)", ok);

ok := true:
for m in [0,2,4,6,8,10,12] do
  z0 := subs(z=0, Ip(m)) - s^(m+1)/(m+1):
  e := simplify(z0) assuming s::positive:
  if e <> 0 then ok := false; end if:
end do:
chk("even m: I_m(z=0) = s^(m+1)/(m+1)", ok);

print("10_edge_primitives: complete");
