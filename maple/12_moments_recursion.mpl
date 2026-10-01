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
Digits := 40:

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

# angular integrals: oint cos^a sin^b = 2 pi (a-1)!!(b-1)!!/(a+b)!!  (a,b even), 0 else
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

# =================== 2. half-plane, SYMBOLIC in (d, R) =========================
# solid {y2 > c}, outward normal n = (0,-1), t = (1,0); on the edge y = (s, -z), z = -c.
#   outside the solid at distance d:  z = -d, indicator 0       (c = d)
#   inside the solid at depth d:      z = +d, indicator 1       (c = -d)
# chord [-L, L], L = sqrt(R^2 - d^2); atan jump over the chord = 2 arccos(d/R) sign(z).
# Reference: dV/dR = R^(n+k+1) A_ab(d/R), A_ab(u) = int_{asin u}^{Pi-asin u} cos^a sin^b dtheta
# (outside), V(R=d) = 0; inside: V_in = disk - (-1)^b V_out.  Both sides are compared as
# functions of R (derivative) at sample points, 40 digits -- no numerical quadrature needed.
Ipsym := proc(m::integer) option remember;
  if m = 0 then s elif m = -1 then arcsinh(s/abs(z)) elif m = -2 then arctan(s/z)/z
  elif m > 0 then (s*(s^2+z^2)^(m/2) + m*z^2*Ipsym(m-2))/(m+1)
  else ((m+3)*Ipsym(m+2) - s*(s^2+z^2)^((m+2)/2))/((m+2)*z^2) end if end proc:
Jpsym := proc(m::integer) `if`(m = -2, ln(s^2+z^2)/2, (s^2+z^2)^((m+2)/2)/(m+2)) end proc:
Spsym := proc(j::nonnegint, m::integer) local h, i;
  if j mod 2 = 0 then h := j/2; add(binomial(h,i)*(-z^2)^(h-i)*Ipsym(m+2*i), i=0..h)
  else h := (j-1)/2; add(binomial(h,i)*(-z^2)^(h-i)*Jpsym(m+2*i), i=0..h) end if end proc:

Lh := sqrt(R^2 - d^2):
# int_{-L}^{L} s^j r^m ds with z = zz (zz = -d or +d); |z| = d
CS := proc(j, m, zz) local F;
  F := Spsym(j, m);
  subs([s = Lh, z = zz], F) - subs([s = -Lh, z = zz], F)
end proc:

# (a) symbolic recursion on the half-plane
momentAS := proc(a, b, n, zz, ind) local i, beta, bi, t1, t2, bm, sgn;
  if a + b = 0 then
    sgn := `if`(zz = d, 1, -1);
    return ind*2*Pi*R^(n+2)/(n+2) + (zz*CS(0, n, zz) - R^(n+2)*2*arccos(d/R)*sgn)/(n+2);
  end if;
  if a > 0 then i := 1; beta := [a-1, b]; else i := 2; beta := [a, b-1]; end if;
  # edge normal (0,-1): n_1 = 0, n_2 = -1;  y^beta on the edge = s^beta1 (-zz)^beta2
  if i = 1 then t1 := 0;
  else t1 := -(-zz)^beta[2]*( CS(beta[1], n+2, zz) - R^(n+2)*CS(beta[1], 0, zz) )/(n+2) end if;
  bi := beta[i]; t2 := 0;
  if bi >= 1 then
    bm := [beta[1] - `if`(i = 1, 1, 0), beta[2] - `if`(i = 2, 1, 0)];
    t2 := bi*(momentAS(bm[1], bm[2], n+2, zz, ind) - R^(n+2)*momentAS(bm[1], bm[2], 0, zz, ind))/(n+2);
  end if;
  t1 - t2
end proc:
# (b) symbolic
omegaS := proc(a, b) `if`(type(a, even) and type(b, even), 2*Pi*df(a-1)*df(b-1)/df(a+b), 0) end proc:
momentBS := proc(a, b, n, zz, ind) local k, e;
  k := a + b: e := n + 2 + k:
  ind*omegaS(a, b)*R^e/e + zz/e*(-zz)^b*( CS(a, n, zz) - R^e*CS(a, -2-k, zz) )
end proc:

Acl := proc(a, b) option remember; local th, u;
  simplify(eval(int(cos(th)^a*sin(th)^b, th), th = Pi - arcsin(u)) - eval(int(cos(th)^a*sin(th)^b, th), th = arcsin(u))) assuming u > 0, u < 1
end proc:

worstA := 0: worstB := 0: worstAB := 0: worst0 := 0: cnt := 0:
pts := [[1/5, 1], [3/5, 1], [1/10, 7/10]]:
for n in [0, 1, 2, 4] do
  printf("  [half-plane symbolic: n = %d, t = %.1f s]\n", n, time());
  for a from 0 to 4 do for b from 0 to 4-a do
    k := a + b:
    Aab := Acl(a, b):
    for side in [out, inn] do
      zz := `if`(side = out, -d, d): ind := `if`(side = out, 0, 1):
      EA := momentAS(a, b, n, zz, ind):
      EB := `if`(k >= 0, momentBS(a, b, n, zz, ind), 0):
      dA := diff(EA, R): dB := diff(EB, R):
      # polar derivative
      dref := R^(n+k+1)*subs(u = d/R, Aab):
      dref := `if`(side = out, dref, R^(n+k+1)*omegaS(a, b) - (-1)^b*dref):
      for pt in pts do
        sb := [d = pt[1], R = pt[2]]:
        eA := abs(evalf(subs(sb, dA - dref))):
        eB := abs(evalf(subs(sb, dB - dref))):
        eAB := abs(evalf(subs(sb, EA - EB))):
        worstA := max(worstA, eA): worstB := max(worstB, eB): worstAB := max(worstAB, eAB):
        cnt := cnt + 1:
      end do:
      # value at the lower end R = d (L = 0): outside -> 0 ; inside -> disk moment
      v0 := evalf(subs([d = 1/3, R = 1/3], EA)):
      r0 := `if`(side = out, 0, evalf(subs([d = 1/3, R = 1/3], omegaS(a, b)*R^(n+2+k)/(n+2+k)))):
      worst0 := max(worst0, abs(v0 - r0)):
    end do:
  end do: end do:
end do:
printf("half-plane symbolic: %d (n,alpha,side,point) cases; worst |dA/dR-polar| = %.2e, |dB/dR-polar| = %.2e, |a-b| = %.2e, |V(R=d) - ref| = %.2e\n",
       cnt, worstA, worstB, worstAB, worst0);
chk("half-plane x outside/inside, k<=4: d/dR of (a) = polar integrand (40 digits)", evalb(worstA < 1e-30));
chk("half-plane x outside/inside, k<=4: d/dR of (b) = polar integrand (40 digits)", evalb(worstB < 1e-30));
chk("half-plane: (a) = (b) as functions of R (and V(R=d) matches the start value)", evalb(worstAB < 1e-30 and worst0 < 1e-30));

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
for AB in [ [[1/5, 0], [0, 1/4]], [[4/5, -1/10], [1/10, 3/5]], [[3/2, 1/2], [-1/2, 3/2]] ] do
  Es := mkedges([[0,0], AB[1], AB[2]], [0, 0]):
  Efar := Es[2]:
  z := Efar[5]: s0 := Efar[6]: s1 := Efar[7]:
  thn := arctan(Efar[2], Efar[1]):
  ang := arctan(s1/z) - arctan(s0/z):
  ind := ang/(2*Pi):                                   # interior angle / 2 pi at the vertex
  for Rv in [1/8, 1/2, 5/2] do
    for n in [0, 1, 3] do
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
