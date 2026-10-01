# Handoff: exact SPH boundary integrals via edge reductions (2D → 3D, FEM nodal fields)

Purpose: continue in a local code session (with Maple for derivation/verification).
Goal: replace the sector/segment/stub + 2F1 machinery of the 2D exact-triangle paper with
divergence-theorem reductions to **edge line integrals with elementary primitives**, extend to
**higher-order FEM nodal fields**, then to **tetrahedra**.

Status legend: **[V]** verified numerically · **[D]** derived, not verified · **[P]** plan / open

---

## 1. Context (short)

- **Tube Maps** (Nogina & Sellán, SIGGRAPH '26, doi:10.1145/3799902.3811118): second-order
  curvature expansion (tubular coords, $J(s)=1-2Hs+Ks^2$, angular averaging) of the
  Koschier–Bender *smoothed* density integral $\int\gamma(\Phi)W$. Density only; no pressure
  operator (pressure via SPlisHSPlasH host path, i.e. mirroring + grid gradients).
  Their "planar / Winchenbach 2020" baseline is their own $A_0$ term with linear $\gamma$.
  Main-text sign of $K$ in $J$ is inconsistent with their Eq. 16 — re-derive if reused.
- **Winchenbach et al. 2020** (doi:10.1145/3414685.3417829): semi-analytic planar, any quantity,
  pressure via boundary integral with MLS-extrapolated contact-point pressure.
- **Winchenbach & Kolb 2025** (arXiv:2507.21686): exact 2D triangle integrals via decomposition
  + Chebyshev + $_2F_1$. Stated drawbacks: branching (GPU), 2D only, tets "significant challenge".
- Key observation of this session: the same exact results follow from the divergence theorem
  with **only per-edge chord clipping + elementary antiderivatives** (atan, asinh/log, polynomials),
  including interpolants (not just gradients) via an **inside-indicator** term.

## 2. Notation / conventions (2D)

- Evaluation point $x$, $y = x' - x$, $r = |y|$. Normalise support $h=1$ (scale back at the end).
- Kernel normalised: $\int_{\mathbb R^2} W = 1 \Rightarrow 2\pi M(1)=1$, $M(r)=\int_0^r tW(t)\,dt$.
- Edge $e = (p_e,q_e)$, unit tangent $t_e$, **outward** normal $n_e$ (orientation from triangle sign).
- $z_e = n_e\cdot(p_e - x)$ (signed; $>0$ when $x$ is on the inner side of the edge line).
- Along the edge: $y = z_e n_e + s\,t_e$, $r=\sqrt{s^2+z_e^2}$, $s_0=t_e\cdot(p_e-x)$, $s_1=t_e\cdot(q_e-x)$.
- Chord for support radius $R$: if $|z_e|<R$, $L=\sqrt{R^2-z_e^2}$, $s\in[\max(s_0,-L),\min(s_1,L)]$.
- $\mathbb 1[x\in T]$: point-in-triangle test using the **same orientation predicate** as the signs of $z_e$
  (so the atan jump and the indicator jump cancel exactly when $x$ crosses an edge).

## 3. Core 2D identities  [V]

Verified for the full Wendland C4 kernel on 20 random triangles/points vs adaptive area quadrature,
max abs diff $3\times10^{-13}$ (quadrature-limited). Script: `edge_identity_check.py`.

$$\int_T W\,dA = \mathbb 1[x\in T] + \sum_e z_e\int_{\text{chord}}\frac{M(r)-M(1)}{r^2}\,ds$$

$$\nabla_x\!\int_T W(|x-x'|)\,dA' = -\sum_e n_e\int_{\text{chord}} W(r)\,ds$$

$$\int_T y\,W\,dA = -\sum_e n_e\int_{\text{chord}}\Psi(r)\,ds,\qquad \Psi(r)=\int_r^1 tW(t)\,dt\ \ (\text{compact})$$

Why: $\nabla\cdot\big(y\,M(r)/r^2\big)=W$. Beyond the support the flux $M(1)\,y/r^2$ is
divergence-free (non-compact!) → distant edges cancel on a closed boundary, leaving the indicator.
Gradient-type integrands have compact potentials → purely local edge integrals.

## 4. Arbitrary polynomial kernels: truncated monomials  [D]

Write any piecewise polynomial kernel as
$$W(r)=\sum_j q_j(r)\,\mathbb 1[r\le R_j],\qquad q_{\text{last}}=\text{outer piece},\ q_j=\text{piece}_j-\text{piece}_{j+1}.$$
Then everything is linear in **building blocks** $f_{n,R}(r)=r^n\,\mathbb 1[r\le R]$.

- Wendland: single $R=1$.
- Cubic spline (2D, support 1): $W=\sigma\,[\,2(1-q)^3_+ - 8(\tfrac12-q)^3_+\,]$, $\sigma=40/(7\pi)$
  → monomials with $R=1$ and $R=\tfrac12$.  (Check: for $q\le\tfrac12$ gives $1-6q^2+6q^3$.)

Per building block (value, $\alpha=0$):
$$\int_T r^n\mathbb 1[r\le R] = \mathbb 1[x\in T]\frac{2\pi R^{n+2}}{n+2}
 + \sum_e\frac{1}{n+2}\Big[z_e\,I_n(s) - R^{n+2}\arctan(s/z_e)\Big]_{\text{chord}}$$
Summed over a normalised kernel: indicator weights sum to $\mathbb 1[x\in T]$, and the atan
coefficient per edge sums to $M(R)$ → **one atan per edge per support radius**, not per monomial.

Gradient per block (exact per truncated term; circle jumps cancel between pieces of a continuous kernel):
$$\nabla_x\int_T r^n\mathbb 1[r\le R] = -\sum_e n_e\,[I_n(s)]_{\text{chord}}$$

## 5. Edge primitives (closed form)  [D]

With $r=\sqrt{s^2+z^2}$, use **odd** antiderivatives (so chords spanning $s=0$ add, not cancel):

- $I_m(s)=\int r^m ds$:
  $I_0=s$, $I_{-1}=\operatorname{asinh}(s/|z|)$, $I_{-2}=\arctan(s/z)/z$ (only ever appears as $z\,I_{-2}=\arctan(s/z)$),
  recurrence $(m+1)I_m = s\,r^m + m z^2 I_{m-2}$.
  Even $m\ge0$ → polynomial in $s$ (can expand $(s^2+z^2)^{m/2}$ binomially). Odd $m\ge1$ → ends in $I_{-1}$.
- $J_m(s)=\int s\,r^m ds = r^{m+2}/(m+2)$; $J_{-2}=\log r$.
- $\int s^j r^m ds$: substitute $s^2=r^2-z^2$ → linear combination of $I$ (j even) or $J$ (j odd).

Numerics: $z\to0$ — $I_{-1}$ diverges like $\log|z|$ but only enters multiplied by $z^2$ → guard exact $z=0$.
Short chords with both ends same sign → difference of nearby values (use series or short Gauss there).
Downward recurrence for $m<-2$ divides by $z^2$ — **avoid** (see §6, recursion route needs $m\ge-2$ only).

## 6. Higher moments $m_\alpha=\int_T y^\alpha f(r)\,dA$, $k=|\alpha|$

**(a) Recommended: compact-potential recursion (any dimension)  [D, k=1 is V]**
Define $\Phi[f](r) = -\int_r^R t\,f(t)\,dt$ (compact, $\Phi(R)=0$, continuous). Then $y_i f = \partial_i\Phi$ and, for $\alpha=\beta+e_i$:
$$\int_T y^{\beta+e_i} f\,dA = \sum_e n_{e,i}\int_{\text{chord}} y^\beta\,\Phi\,ds\;-\;\beta_i\int_T y^{\beta-e_i}\,\Phi\,dA .$$
Degree drops by 2 per step. Odd $k$ → purely edge integrals (compact). Even $k$ → ends in one
value-type integral (§3 form with $f\to$ iterated $\Phi$). For $f=r^n$: $\Phi=(r^{n+2}-R^{n+2})/(n+2)$,
still polynomial in $r$ → only $I_m, J_m$ with $m\ge -2$.

**(b) Alternative: direct far-field form  [D, implemented in `monomial_edge_forms.py`, unverified]**
$$\int_T y^\alpha r^n\mathbb 1[r\le R] = \mathbb 1[x\in T]\Big(\oint_{S^1}\omega^\alpha\Big)\frac{R^{n+2+k}}{n+2+k}
+\sum_e\frac{z_e}{n+2+k}\int_{\text{chord}}(z_en_e+st_e)^\alpha\Big[r^n-R^{n+2+k}r^{-(2+k)}\Big]ds$$
(from $\nabla\cdot(yPF)=P[(d+k)F+rF']$; far field $P\,y\,r^{-(d+k)}$ divergence-free; $\oint\omega^\alpha=0$ for odd $k$).
Needs $r^{-(2+k)}$ → worse conditioning near edges. Use mainly as an independent cross-check of (a).

**Gradients of moments  [D]:**
$\partial_{x_j}\int_T y^\alpha f = -\sum_e n_{e,j}\int_{\text{chord}} y^\alpha f\,ds$ (purely edge).

## 7. FEM nodal values (P1–P3, interior nodes)  [D]

Field on $T$: $A(x')=\sum_i A_i N_i(x')$, Lagrange basis of degree $p$ (P3 bubble node = just another node).
Re-expand each basis function about the evaluation point:
$$N_i(x+y)=\sum_{|\alpha|\le p} B_{\alpha i}(x)\,y^\alpha,\qquad B_{\alpha i}=\partial^\alpha N_i(x)/\alpha!$$
($N_i$ are polynomials in barycentrics, $\lambda_j(x+y)=\lambda_j(x)+\nabla\lambda_j\cdot y$ → cheap symbolic/numeric expansion.)

- Interpolant: $\langle A\rangle(x)=\sum_i A_i\,w_i$, $\;w_i=\sum_\alpha B_{\alpha i}\,m_\alpha$, $m_\alpha=\int_T y^\alpha W$.
- Gradient $\int_T A\,\nabla_x W$: $\sum_i A_i\sum_\alpha B_{\alpha i}\,g_\alpha$ with
  $g_\alpha=\int_T y^\alpha\nabla_xW = -\sum_e n_e\!\int_{\text{chord}} y^\alpha W\,ds + \sum_j\alpha_j e_j\,m_{\alpha-e_j}$.
- Difference / symmetric SPH forms: as in the 2D paper, swap nodal values ($f_i=\rho_i(A_i-A(x))$ etc.);
  weights unchanged.
- Two-way coupling: the same weights ($w_i$, $g_\alpha$-based) distribute the reaction to nodes →
  exact action = reaction between particle and element.
- Variable boundary density $\rho(x')$ nodal: $\rho A$ is degree $2p$ → just higher moments
  (or treat $\rho A$ as its own nodal field, inexact at degree $p$).
- Samples inside the triangle that are not a unisolvent nodal set (quadrature-point data, MLS samples):
  least-squares / L2-project to $P_p$ per element → $B$ via pseudo-inverse; exact w.r.t. the projection.
- **Conditioning risk [P]:** $B_{\alpha i}\sim L_T^{-|\alpha|}$, $m_\alpha\sim h^{|\alpha|}$ → cancellation
  $\sim (h/L_T)^p$ for elements much smaller than $h$. Quantify with mpmath; mitigations: scale $y$ by $h$,
  hybrid quadrature for elements fully inside the support, edge-local coordinates.

## 8. Pressure (fits as special cases)

- 2020 contact-point MLS pressure (per particle per element, constant) → only $m_0$, $g_0$.
- MLS-extrapolated linear pressure $p(x')=p_c+\nabla p\cdot(y+x-x_c)$ → moments $k\le1$, exact over real
  triangles; MLS order $q$ → moments up to $k=q$.
- Nodal pressure (paper's option (a), needed elements ~$h/4$ at P1) → §7 with P2/P3 now possible.

## 9. 3D plan  [P]

- Value: $\int_{\text{tet}} f = \mathbb 1[x\in\text{tet}]\,4\pi M_3(R) + \sum_f z_f\int_{\text{face}}\frac{M_3(r)-M_3(R)}{r^3}dA$,
  $M_3(r)=\int_0^r t^2 f\,dt$; integrand compact on the face disk $\rho\le\sqrt{R^2-z_f^2}$.
- Face integral = 2D value identity in-plane with radial profile $g(\rho)=z_f(M_3(r)-M_3(R))/r^3$,
  $r=\sqrt{\rho^2+z_f^2}$: indicator(foot in face) term + edge terms with $G(\rho)=\int_0^\rho t\,g\,dt=\int_{|z|}^{r} r'g\,dr'$ (elementary).
- New edge primitive family: $\int r^m/(s^2+w_e^2)\,ds$ with $r^2=s^2+w_e^2+z_f^2$ → elementary
  (solid-angle-type $\arctan\!\big(s z/(w r)\big)$ terms). Cross-check against Van Oosterom–Strackee
  triangle solid angle and closed-form polyhedral Newtonian potentials (Waldvogel 1979; Werner & Scheeres 1997).
- Moments: §6(a) recursion is dimension-independent (tet → faces with $y^\beta\Phi n_i$), then in-plane
  recursion; note the in-plane potential $-\int_\rho^{\rho_R} t\,\varphi\,dt = -\int_r^R r'\varphi\,dr'$ has the same form.
- Watch: $z_f\to0$ face-level jump vs indicator; edges/vertices of the tet inside the support.

## 10. Tier structure (roadmap)  [P]

| Tier | Input geometry | Method | Exact? | Valid when |
|---|---|---|---|---|
| 1 | Volume mesh (tris 2D / tets 3D) + fields inside (P0…Pk nodal) | per-element face/edge reductions (§3–7, §9) | yes (polytopes) | any thickness; conditioning for $L_T\ll h$ |
| 2 | Closed boundary mesh (polyline 2D / tri surface 3D), data on boundary only | same identities, interior faces cancel | yes | watertight, non-overlapping |
| 3 | Implicit surface (SDF + curvature) | closest-point expansion: order 0 = planar (2020), 1 = $H$, 2 = $H^2,K$ (Tube Maps), full shape operator for tensor moments | $O((h\kappa)^{n+1})$ | reach $\ge h$ on solid side, single sheet in support, smooth |
| 4 | Skeleton + radius (medial rep.): point → disk/ball, curve → strip (2D) / strand (3D), surface → shell/plate (3D) | expansion about the skeleton, or semi-analytic exact for straight pieces (§11) | approx / semi-analytic | thickness $a \ll h$ (series); any $a$ for straight segments |

Notes:
- Tiers 1 and 2 use the same machinery; the difference is where the data lives (volume vs boundary).
- Tiers 3 and 4 are complementary regimes (thickness $\gg h$ vs $\ll h$) → **gap at thickness $\sim h$**
  where neither expansion holds → fall back to tier 1/2 or semi-analytic forms.

Orthogonal axes (every tier must state what it supports):
- Field order: P0, P1, Pk nodal, MLS contact-point extrapolation of degree $q$.
- Operator: interpolant, gradient, div/curl, moments $m_0,m_1,m_2$ (kernel consistency corrections),
  viscosity/Laplacian (not linear in the field → single-contact-point treatment as in the 2D paper).
- Kernel: compact polynomial (exact), piecewise via truncated monomials, non-polynomial (tiers 3/4 or quadrature only).
- Geometry motion: rigid / deforming (tier 3 must recompute curvature; tiers 1/2 have no precompute),
  differentiability w.r.t. geometry (shape optimisation).

Cross-cutting gaps:
- **Tier selection per particle–object pair**, from $t/h$ (local thickness or reach), $h\kappa$, $L_T/h$, $a/h$.
  With adaptive resolution an object can move between tiers.
- **Continuity at tier switches**: match asymptotics (tier 4 series ↔ tier 1 exact prism as $a\to h$;
  tier 3 order 0 ↔ tier 2 on flat geometry) or blend; otherwise forces jump.
- **Curved exact elements** (between tiers 2 and 3): circular arcs / quadric patches lead to elliptic integrals.
- **Multiple / touching objects**: tiers 1/2 assume disjoint elements; implicit tiers sum per object →
  double counting at contact (fibre on wall, stacked bodies) — needs union handling.
- **Sub-resolution physics**: tiers give the geometric transfer only (§11 caveat).
- **Baselines**: particle boundaries (Akinci), density/volume maps, Tube Maps.

## 11. Fibres / codimension-2 boundaries (tier 4 detail)  [D/P]

**Why surface expansions (Tube Maps, tier 3) fail for $a<h$ / locally closed geometry:**
- Circumferential curvature $1/a$ → $h\kappa=h/a>1$: the distance expansion is not small.
- Surface tubular coords valid only to the reach ($=a$, the centreline); integrating $s$ to $h$ folds:
  $J=(1-s/a)(1-\kappa_2 s)$ hits zero and goes negative; the far side is never seen as a second sheet.
- Their high-curvature fallback is planar = half-space → overestimates by ~½ kernel vs true $O(\pi a^2\!\int W)$.
- Same failure for thin plates and narrow fluid gaps (multiple sheets in the support).
- I believe the Koschier–Bender smoothed integrand itself assumes the solid is ~$h$ thick → check their paper.
- Partial patch: cut $s$ at the local thickness (distance to the medial axis); $(1-s/a)$ on $[0,a]$ is then
  the exact polar Jacobian, but the kernel-distance expansion is still in $h/a$.

**Centreline parametrisation:** $x'=c(\tau)+\rho(\cos\varphi\,N+\sin\varphi\,B)$, Jacobian $\rho(1-\kappa_c\rho\cos\varphi)$
(Weyl tube formula setting). Small parameter is $h\kappa_c$ (fibre bending), usually small even for $a<h$.

**Straight segment, any $a$ (semi-analytic):** axial integral closed form with the same $I_m$ primitives
($z^2\to\zeta^2$, distance within the cross-section plane). The remaining cross-section integral is a radial
function over an offset disk → 2D divergence identity → one periodic integral around the circumference
(elliptic in closed form; trapezoid rule converges exponentially — split at kinks where the support sphere crosses).

**Slender limit $a\ll h$:**
$$\rho_B\approx\rho_0\pi a^2\sum_{\text{seg}}\int_{\text{chord}}\Big[W+\tfrac{a^2}{8}\Delta_\perp W+\tfrac{a^4}{192}\Delta_\perp^2W+\dots\Big]ds$$
(2D-disk mean-value series $\sum_k \Delta^k f\,a^{2k}/(4^k k!(k+1)!)$). Leading term = the edge-chord primitive with
$z$ = distance to the segment line, i.e. the same code as 2D edges. Wendland has no linear term → $\Delta_\perp W$
regular on the axis. Series is asymptotic (kernel smoothness at $r=h$) → truncate at C-order (C4: $a^2$ or $a^4$).
2D strip analog: 1D mean-value series $\sum_k a^{2k}\partial_\perp^{2k}W/(2k+1)!$.

**Curved centreline:** expansion in $\kappa_c$ (and torsion). Tube volume is independent of $\kappa_c$ (Weyl, $a<1/\kappa_c$),
but kernel weighting is not symmetric in $\varphi$ → first-order correction $\propto\kappa_c a^2$, depending on the
direction to $x$ relative to $N$.

**Point primitives:** 3D ball of radius $a$ at distance $d$ — shell theorem gives
$\int_{S^2} f(|x-c-\rho\omega|)\,d\omega = \frac{2\pi}{d\rho}\int_{|d-\rho|}^{d+\rho} r f(r)\,dr$ → **elementary** for polynomial kernels
(clip limits at $h$). 2D disk: no such reduction ($d\varphi$ is not $r\,dr$) → elliptic / periodic quadrature.

**Joints and ends:** polyline joints overlap (inner) / gap (outer) → mitred segments, ball joints, or
curved-segment correction. Ends: flat cap = 2D face problem; hemispherical cap = ball ∩ ball. Gradients are
elementary including segment-endpoint terms.

**Physical caveat:** for $a<h$ the peak density contribution is $O((a/h)^2)$ of $\rho_0$ → pressure-only
no-penetration is weak (leakage, cf. Tube Maps Fig. 15). Use the exact weights as the transfer operator for a
sub-resolution force model (slender-body / Morison drag, or penalty), and the same weights for the reaction on
fibre nodes (exact action = reaction).

## 12. Task list / test plan

1. [ ] Maple: verify primitives $I_m$, $J_m$, $\int s^j r^m$ (odd forms), and the §4 per-monomial value/gradient.
2. [ ] Maple: verify §6(a) recursion for $k=2,3$; cross-check against §6(b).
3. [ ] Implement 2D closed form (float64 first), kernels: Wendland C2/C4/C6, cubic spline (two radii).
4. [ ] Exact tests (no quadrature): mesh covering the whole support → $\int A W$ equals the closed-form
   disk integral for polynomial $A$ (and gradient = known), à la the paper's Fig. 4/5 setup.
5. [ ] Compare against the 2025 2D implementation (C++/PyTorch) on its test cases; accuracy and speed.
6. [ ] Robustness: $x$ on an edge / at a vertex, $z\to0$, tiny chords, tiny elements vs $h$, float32.
7. [ ] Autodiff (PyTorch / Warp) vs analytic gradients.
8. [ ] FEM P1–P3 nodal weights (§7); conditioning study with mpmath.
9. [ ] Branch count / GPU suitability (per edge: one clip + few elementary calls).
10. [ ] 3D derivation (§9) in Maple → tets.
11. [ ] Tier selection metrics + continuity tests at tier switches (§10).
12. [ ] Tier 4: slender series vs semi-analytic straight cylinder vs tier-1 prism mesh as $a/h$ varies (find the crossover).
13. [ ] Tier 4: 3D ball primitive (shell theorem) closed form; 2D disk via periodic quadrature.
14. [ ] Curved-centreline correction ($\kappa_c$ order 1) in Maple.

## 13. Open questions

- Best stable evaluation of odd-$m$ primitives for short chords and large $m$.
- Conditioning of nodal re-expansion for $L_T\ll h$ (§7).
- On-edge / on-vertex conventions (half-plane limit should give exactly ½ of the disk contribution).
- 3D: is the face → edge chain cheaper than a direct tet ∩ ball decomposition in practice?

## 14. Files

- `edge_identity_check.py` — [V] §3 identities vs area quadrature (edge integrals by `quad`).
- `monomial_edge_forms.py` — [D] closed-form primitives + §4/§6(b); brute-force reference timed out
  (discontinuity at $r=R$); swap in a polar reference (radial breakpoint at $R$) or Maple.

## 15. References

- Nogina & Sellán, Tube Maps, SIGGRAPH '26 — https://gatc.cs.columbia.edu/projects/tubemaps.html
- Winchenbach, Akhunov, Kolb, ACM TOG 39(6), 2020 — doi:10.1145/3414685.3417829
- Winchenbach & Kolb, arXiv:2507.21686 (2025)
- Koschier & Bender 2017 (density maps); Bender et al. 2019 (volume maps)
- Violeau & Mayrhofer 2014; Mayrhofer et al. 2015 (exact wall renormalisation via divergence theorem)
- Band et al. 2018 (MLS pressure boundaries)
- Pottmann et al., integral invariants (CAGD 2009) — curvature expansions of ball-neighbourhood integrals
