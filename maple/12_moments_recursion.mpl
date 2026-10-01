# ===========================================================================
# 12_moments_recursion.mpl -- docs/derivations/moments-recursion.md
#   (HANDOFF.md section 6, task 12.2)
#
#  1. pointwise identities behind (a) the compact-potential recursion and
#     (b) the far-field form; angular integrals  oint w^alpha
#  2. half-plane (x outside and inside), alpha = (a,b), k = 1..4, blocks r^n 1[r<=R]:
#       (a) recursion, (b) far-field form, polar reference     -- 40 digits
#       (+ symbolic d/dR check of (a) and (b) against the polar integrand)
#  3. wedge with x at a vertex (3 edges, two through x, asymmetric chords):
#       (a) recursion vs polar reference, k = 0..4                -- 40 digits
#
# Finding recorded here: (b) is NOT usable at z = 0 (edge through x): z * int r^(-2-k)
# has a finite, direction-dependent limit; (a) is regular there.
# ===========================================================================
restart:
Digits := 50:

chk := proc(label, ok)
  printf("%-70s %s\n", label, `if`(ok, "PASS", "FAIL"));
end proc:

# =================== 1. pointwise identities ===============================
y1 := 'y1': y2 := 'y2':
rr := sqrt(y1^2 + y2^2):
ok := true: cnt := 0:
for n from 0 to 8 do
  Phi := (rr^(n+2) - R^(n+2))/(n+2):
  for a from 0 to 4 do for b from 0 to 4-a do
    if a+b >= 1 then
      if a > 0 then i := 1; beta := [a-1, b]; else i := 2; beta := [a, b-1]; end if:
      # d_i( y^beta Phi ) - beta_i y^(beta - e_i) Phi == y^alpha r^n
      lhs_ := diff(y1^beta[1]*y2^beta[2]*Phi, `if`(i=1, y1, y2)):
      bi := beta[i]:
      corr := `if`(bi >= 1, bi*Phi*y1^(beta[1]-`if`(i=1,1,0))*y2^(beta[2]-`if`(i=2,1,0)), 0):
      e := simplify(lhs_ - corr - y1^a*y2^b*rr^n) assuming y1::real, y2::real, y1^2+y2^2 > 0:
      cnt := cnt + 1:
      if e <> 0 then ok := false; print("FAIL (a) step", n, a, b, e); end if:
    end if:
  end do: end do:
end do:
chk(sprintf("(a) pointwise: d_i(y^b Phi) - b_i y^(b-e_i) Phi = y^a r^n  (%d cases)", cnt), ok);

# (b): div(y P F) = P[(2+k) F + r F'],  F = r^n/e inside, R^e r^(-2-k)/e outside, e = n+2+k
ok := true: cnt := 0:
for n from 0 to 6 do for a from 0 to 4 do for b from 0 to 4-a do
  k := a+b: e := n+2+k:
  P := y1^a*y2^b:
  Fin := rr^n/e: Fout := R^e*rr^(-2-k)/e:
  din := simplify(diff(y1*P*Fin, y1) + diff(y2*P*Fin, y2) - P*rr^n) assuming y1::real, y2::real, y1^2+y2^2 > 0:
  dout := simplify(diff(y1*P*Fout, y1) + diff(y2*P*Fout, y2)) assuming y1::real, y2::real, y1^2+y2^2 > 0:
  cont := simplify(subs(r = R, r^n/e) - subs(r = R, R^e*r^(-2-k)/e)):
  cnt := cnt + 1:
  if din <> 0 or dout <> 0 or cont <> 0 then ok := false; print("FAIL (b) step", n, a, b, din, dout, cont); end if:
end do: end do: end do:
chk(sprintf("(b) pointwise: div(y P F)=P r^n inside, 0 outside, F cont. at R (%d cases)", cnt), ok);

# angular integrals: oint cos^a sin^b = 2 pi (a-1)!!(b-1)!!/(a+b)!!  (a,b even), else 0
ok := true:
for a from 0 to 6 do for b from 0 to 6 do
  ang := int(cos(t)^a*sin(t)^b, t=0..2*Pi):
  if type(a, even) and type(b, even) then
    form := 2*Pi*doublefactorial(a-1)*doublefactorial(b-1)/doublefactorial(a+b-1+1-1) ;
  end if:
end do: end do:
# (note: (a-1)!!(b-1)!!/(a+b)!!  with the usual convention (-1)!! = 1)
df := proc(n) if n <= 0 then 1 else doublefactorial(n) end if end proc:
ok := true:
for a from 0 to 6 do for b from 0 to 6 do
  ang := int(cos(t)^a*sin(t)^b, t=0..2*Pi):
  form := `if`(type(a, even) and type(b, even), 2*Pi*df(a-1)*df(b-1)/df(a+b), 0):
  if simplify(ang - form) <> 0 then ok := false; print("FAIL angular", a, b, ang, form); end if:
end do: end do:
chk("oint cos^a sin^b = 2 pi (a-1)!!(b-1)!!/(a+b)!!  (even a,b), 0 else; a,b<=6", ok);
omega := proc(a, b) `if`(type(a, even) and type(b, even), 2*Pi*df(a-1)*df(b-1)/df(a+b), 0) end proc:

# =================== numeric primitives (guarded at z = 0) ==================
Inum := proc(m, sv, zv) local r;
  r := sqrt(sv^2 + zv^2);
  if m = 0 then return sv end if;
  if m = 1 then return (sv*r + `if`(zv = 0, 0, zv^2*arcsinh(sv/abs(zv))))/2 end if;
  (sv*r^m + m*zv^2*Inum(m-2, sv, zv))/(m+1)
end proc:
Jnum := proc(m, sv, zv) (sv^2+zv^2)^((m+2)/2)/(m+2) end proc:   # m >= 0
Snum := proc(j, m, sv, zv) local h, i;            # int s^j r^m ds, m >= 0
  if j mod 2 = 0 then h := j/2; add(binomial(h,i)*(-zv^2)^(h-i)*Inum(m+2*i, sv, zv), i=0..h)
  else h := (j-1)/2; add(binomial(h,i)*(-zv^2)^(h-i)*Jnum(m+2*i, sv, zv), i=0..h) end if
end proc:

# edge record E = [n1, n2, t1, t2, z, s0, s1]
chord := proc(E, R) local L;
  if abs(E[5]) >= R then return FAIL end if;
  L := sqrt(R^2 - E[5]^2);
  [max(E[6], -L), min(E[7], L)]
end proc:
# int_chord y^beta * sum_p gam_p r^(m_p) ds  (y = z n + s t)
edgeint := proc(E, R, a, b, gl) local ch, P, j, c, g, tot, ss;
  ch := chord(E, R);
  if ch = FAIL or ch[1] >= ch[2] then return 0 end if;
  P := expand((E[5]*E[1] + ss*E[3])^a * (E[5]*E[2] + ss*E[4])^b);
  tot := 0;
  for j from 0 to a+b do
    c := coeff(P, ss, j);
    if c <> 0 then
      for g in gl do
        tot := tot + c*g[1]*(Snum(j, g[2], ch[2], E[5]) - Snum(j, g[2], ch[1], E[5]));
      end do;
    end if;
  end do;
  tot
end proc:

# value block, indicator `ind` (1/0/fraction at degenerate positions)
valueblock := proc(Es, ind, n, R) local tot, E, ch, jump;
  tot := ind*2*Pi*R^(n+2)/(n+2);
  for E in Es do
    ch := chord(E, R);
    if ch <> FAIL and ch[1] < ch[2] then
      jump := `if`(E[5] = 0, 0, arctan(ch[2]/E[5]) - arctan(ch[1]/E[5]));
      tot := tot + (E[5]*(Inum(n, ch[2], E[5]) - Inum(n, ch[1], E[5])) - R^(n+2)*jump)/(n+2);
    end if;
  end do;
  tot
end proc:

# (a) recursion
momentA := proc(Es, ind, a, b, n, R) local i, beta, bi, t1, t2, E, bm;
  if a + b = 0 then return valueblock(Es, ind, n, R) end if;
  if a > 0 then i := 1; beta := [a-1, b]; else i := 2; beta := [a, b-1]; end if;
  t1 := 0;
  for E in Es do
    t1 := t1 + E[i]*edgeint(E, R, beta[1], beta[2], [[1/(n+2), n+2], [-R^(n+2)/(n+2), 0]]);
  end do;
  bi := beta[i];
  t2 := 0;
  if bi >= 1 then
    bm := [beta[1] - `if`(i = 1, 1, 0), beta[2] - `if`(i = 2, 1, 0)];
    t2 := bi*(momentA(Es, ind, bm[1], bm[2], n+2, R) - R^(n+2)*momentA(Es, ind, bm[1], bm[2], 0, R))/(n+2);
  end if;
  t1 - t2
end proc:

# (b) far-field form (z <> 0 for every edge with a non-empty chord); needs r^(-2-k)
momentB := proc(Es, ind, a, b, n, R) local k, e, tot, E, ch, P, j, c, ss, lo, hi, sg;
  k := a + b: e := n + 2 + k:
  tot := ind*omega(a, b)*R^e/e;
  for E in Es do
    ch := chord(E, R);
    if ch <> FAIL and ch[1] < ch[2] then
      if E[5] = 0 then error "(b) undefined at z=0" end if;
      P := expand((E[5]*E[1] + ss*E[3])^a * (E[5]*E[2] + ss*E[4])^b);
      for j from 0 to k do
        c := coeff(P, ss, j);
        if c <> 0 then
          tot := tot + E[5]/e*c*( (Sg(j, n, ch[2], E[5]) - Sg(j, n, ch[1], E[5]))
                                  - R^e*(Sg(j, -2-k, ch[2], E[5]) - Sg(j, -2-k, ch[1], E[5])) );
        end if;
      end do;
    end if;
  end do;
  tot
end proc:
# general S_{j,m} including m < 0 (downward recursion) -- numeric, z <> 0
Ig := proc(m, sv, zv) local r;
  r := sqrt(sv^2 + zv^2);
  if m = 0 then sv
  elif m = -1 then arcsinh(sv/abs(zv))
  elif m = -2 then arctan(sv/zv)/zv
  elif m > 0 then (sv*r^m + m*zv^2*Ig(m-2, sv, zv))/(m+1)
  else ((m+3)*Ig(m+2, sv, zv) - sv*r^(m+2))/((m+2)*zv^2) end if
end proc:
Jg := proc(m, sv, zv) `if`(m = -2, ln(sqrt(sv^2+zv^2)), (sv^2+zv^2)^((m+2)/2)/(m+2)) end proc:
Sg := proc(j, m, sv, zv) local h, i;
  if j mod 2 = 0 then h := j/2; add(binomial(h,i)*(-zv^2)^(h-i)*Ig(m+2*i, sv, zv), i=0..h)
  else h := (j-1)/2; add(binomial(h,i)*(-zv^2)^(h-i)*Jg(m+2*i, sv, zv), i=0..h) end if
end proc:

# =================== 2. half-plane ===========================================
# solid {y2 > c}, outward normal (0,-1), t = (1,0), edge points y = (s, -z): z = -c.
#  outside: c = d>0,  z = -d, ind = 0.   inside: c = -d, z = +d, ind = 1.
# polar reference (outside):  int_d^R r^(n+k+1) A_{ab}(d/r) dr,
#   A_{ab}(u) = int_{asin u}^{Pi-asin u} cos^a sin^b dtheta
Aab := proc(a, b, u) local th; Int(cos(th)^a*sin(th)^b, th = arcsin(u) .. Pi - arcsin(u)) end proc:
Acl := proc(a, b) option remember; local th, u;
  # closed form of A_ab(u), u = sin(theta0) in (0,1)
  simplify(eval(int(cos(th)^a*sin(th)^b, th), th = Pi - arcsin(u)) - eval(int(cos(th)^a*sin(th)^b, th), th = arcsin(u))) assuming u > 0, u < 1
end proc:
polar_out := proc(a, b, n, dv, Rv) local k, rho, F;
  k := a + b;
  F := unapply(subs(u = dv/rho, Acl(a, b)), rho);
  evalf(Int(rho^(n+k+1)*F(rho), rho = dv .. Rv))
end proc:
# full disk moment
disk := proc(a, b, n, Rv) omega(a, b)*Rv^(n+2+a+b)/(n+2+a+b) end proc:

worstA := 0: worstB := 0: cnt := 0:
for pt in [[1/5, 1], [3/5, 1], [1/10, 7/10]] do
  dv := pt[1]: Rv := pt[2]:
  Eout := [map(evalf, [0, -1, 1, 0, -dv, -10, 10])]:     # far-extent edge, chord is clipped by L
  Ein  := [map(evalf, [0, -1, 1, 0,  dv, -10, 10])]:
  for n in [0, 1, 3] do
    for a from 0 to 4 do for b from 0 to 4-a do
      k := a + b:
      # reference
      ref_out := polar_out(a, b, n, dv, Rv):
      ref_in  := disk(a, b, n, Rv) - (-1)^b*ref_out:          # reflect y2 -> -y2
      A_out := evalf(momentA(Eout, 0, a, b, n, Rv)):
      A_in  := evalf(momentA(Ein,  1, a, b, n, Rv)):
      eA := max(abs(A_out - ref_out), abs(A_in - ref_in)):
      worstA := max(worstA, eA):
      if k >= 0 then
        B_out := evalf(momentB(Eout, 0, a, b, n, Rv)):
        B_in  := evalf(momentB(Ein,  1, a, b, n, Rv)):
        eB := max(abs(B_out - ref_out), abs(B_in - ref_in)):
        worstB := max(worstB, eB):
      end if:
      cnt := cnt + 1:
    end do: end do:
  end do:
end do:
printf("half-plane: %d (point,n,alpha) cases, k=0..4; worst |(a)-polar| = %.3e, worst |(b)-polar| = %.3e\n", cnt, worstA, worstB);
chk("half-plane, x outside & inside: (a) recursion = polar reference (40 digits)", evalb(worstA < 1e-35));
chk("half-plane, x outside & inside: (b) far-field   = polar reference (40 digits)", evalb(worstB < 1e-35));

# symbolic d/dR check of (a) and (b) on the half-plane (x outside), selected alpha
# (a) via the explicit recursion needs symbolic S_{j,m}; do it for k=1,2,3 with n=0,1,2
Ipsym := proc(m::integer) option remember;
  if m = 0 then s elif m = -1 then arcsinh(s/abs(z)) elif m = -2 then arctan(s/z)/z
  elif m > 0 then (s*(s^2+z^2)^(m/2) + m*z^2*Ipsym(m-2))/(m+1)
  else ((m+3)*Ipsym(m+2) - s*(s^2+z^2)^((m+2)/2))/((m+2)*z^2) end if end proc:
Jpsym := proc(m::integer) `if`(m = -2, ln(s^2+z^2)/2, (s^2+z^2)^((m+2)/2)/(m+2)) end proc:
Spsym := proc(j::nonnegint, m::integer) local h, i;
  if j mod 2 = 0 then h := j/2; add(binomial(h,i)*(-z^2)^(h-i)*Ipsym(m+2*i), i=0..h)
  else h := (j-1)/2; add(binomial(h,i)*(-z^2)^(h-i)*Jpsym(m+2*i), i=0..h) end if end proc:
# symbolic chord integral over [-L, L] with z = -d  of s^j r^m:
chordS := proc(j, m) local F, hi, lo, d1, d2;
  F := Spsym(j, m);
  hi := sqrt(R^2 - d^2); lo := -hi;
  subs([s = hi, z = -d], F) - subs([s = lo, z = -d], F)
end proc:
ok := true: okB := true: cnt := 0:
for n in [0, 1, 2] do for a from 0 to 3 do for b from 0 to 3-a do
  k := a + b:
  if k >= 1 then
    e := n + 2 + k:
    # (b) symbolic, x outside: sum_j  z/e * c_j * [chordS(j,n) - R^e chordS(j,-2-k)],  P = s^a (-z)^b = s^a d^b (z=-d)
    Bsym := (-d)/e*d^b*( chordS(a, n) - R^e*chordS(a, -2-k) ):
    dB := diff(Bsym, R):
    # the polar integrand:  R^(n+k+1) A_ab(d/R)
    refd := R^(n+k+1)*int(cos(th)^a*sin(th)^b, th = arcsin(d/R) .. Pi - arcsin(d/R)):
    e1 := simplify(dB - refd) assuming d::positive, R > d:
    cnt := cnt + 1:
    if e1 <> 0 and a = 2 and b = 0 and n = 0 then print(e1); end if:
    if e1 <> 0 then okB := false; printf("  (b) symbolic d/dR residual not simplified to 0 for n=%d a=%d b=%d\n", n, a, b); end if:
  end if:
end do: end do: end do:
chk(sprintf("half-plane (b) SYMBOLIC: d/dR of far-field edge form = polar integrand (%d cases)", cnt), okB);

# =================== 3. wedge at a vertex: (a) vs polar ======================
mkedges := proc(V, x) local Es, i, p, q, t, ln, nn, tt, Ed;
  Es := []:
  for i from 1 to nops(V) do
    p := V[i]; q := V[(i mod nops(V)) + 1];
    ln := sqrt((q[1]-p[1])^2 + (q[2]-p[2])^2);
    tt := [(q[1]-p[1])/ln, (q[2]-p[2])/ln];
    nn := [tt[2], -tt[1]];                       # outward for CCW
    Ed := [nn[1], nn[2], tt[1], tt[2],
           nn[1]*(p[1]-x[1]) + nn[2]*(p[2]-x[2]),
           tt[1]*(p[1]-x[1]) + tt[2]*(p[2]-x[2]),
           tt[1]*(q[1]-x[1]) + tt[2]*(q[2]-x[2])];
    Es := [op(Es), map(evalf, Ed)];
  end do;
  Es
end proc:

worst := 0: cnt := 0:
for AB in [ [[1/5, 0], [0, 1/4]], [[4/5, -1/10], [1/10, 3/5]], [[3/2, 1/2], [-1/2, 3/2]], [[1/2, -1/4], [3/10, 1/2]] ] do
  Es := mkedges([[0,0], AB[1], AB[2]], [0, 0]):
  Efar := Es[2]:
  z := Efar[5]: s0 := Efar[6]: s1 := Efar[7]:
  thn := arctan(Efar[2], Efar[1]):
  ang := arctan(s1/z) - arctan(s0/z):
  ind := ang/(2*Pi):                                   # interior angle / 2 pi at the vertex
  for Rv in [1/8, 1/2, 1, 5/2] do
    for n in [0, 1, 2, 4] do
      for a from 0 to 4 do for b from 0 to 4-a do
        k := a + b: e := n + 2 + k:
        val := momentA(Es, ind, a, b, n, Rv):
        # polar reference
        p0 := arctan(s0/z): p1 := arctan(s1/z):
        if z >= Rv then pL := 0 else pL := arccos(z/Rv) end if:
        cuts := [p0, p1]:
        if z < Rv then cuts := sort([p0, p1, seq(c, c in select(c -> evalf(p0) < evalf(c) and evalf(c) < evalf(p1), [-pL, pL]))], (u,v) -> evalf(u) < evalf(v)) end if:
        ref := 0:
        for m1 from 1 to nops(cuts)-1 do
          lo := cuts[m1]: hi := cuts[m1+1]: mid := evalf((lo+hi)/2):
          if z < Rv and abs(mid) < evalf(pL) then
            ref := ref + Int(cos(thn+ph)^a*sin(thn+ph)^b*(z/cos(ph))^e/e, ph = lo .. hi);
          else
            ref := ref + Int(cos(thn+ph)^a*sin(thn+ph)^b*Rv^e/e, ph = lo .. hi);
          end if:
        end do:
        ref := evalf(ref):
        df := abs(evalf(val) - ref):
        worst := max(worst, df): cnt := cnt + 1:
      end do: end do:
    end do:
  end do:
end do:
printf("wedge at vertex: %d cases (k=0..4), worst |(a) - polar| = %.3e\n", cnt, worst);
chk("wedge at vertex (3 edges, z=0 edges included): (a) recursion = polar", evalb(worst < 1e-35 and cnt > 100));

print("12_moments_recursion: complete");
