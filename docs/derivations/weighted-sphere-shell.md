# Weighted sphere shell integration

For unit support, let `d` be the signed outward query-to-sphere distance,
`R > 0` its radius, `c = R + d >= 0` the centre-to-query distance, and `n` the
outward unit centre-to-query axis. Source-minus-query coordinates are `y = r w`.
A shell point lies in the solid precisely when

```
|r w + c n|² <= R²,
t = w.n <= (R² - c² - r²)/(2 c r).
```

For `c > 0`, the implementation evaluates the same cap as
`1 - ((r+d)/r) * ((R/c + 1 + r/c)/2)`. This avoids subtracting `R²-c²`.
For `c = 0` a shell is full for `r < R` and empty for `r > R`; its single
boundary radius has zero volume measure. The integration splits at `|d|`,
`2R+d`, support 1 and cubic's kernel knot 1/2, retaining only interior cuts.
These are exact spherical limits in real arithmetic. Native geometry arithmetic
is explicitly the selected fp32/fp64 precision, separately from host input
validation and rule construction.

With `P = I - n nᵀ`, write `w = n t + sqrt(1-t²) v`, where `v` is a unit vector
in the two-dimensional tangent plane. The polar Gauss rule integrates the
Cartesian polynomial after azimuthal integration. Azimuthal equally spaced
samples integrate its trigonometric modes through the advertised degree in
exact arithmetic. For a request with degree `a` and derivative order `s`,
`a+s <= 11`, so at most 12 azimuth samples and six polar Gauss samples suffice.
Radial Gauss is separate. Default radial order is 24; every prepared rule
compares normalized requested moments to order+8, promotes or rejects.

The four supported SPH kernels have piecewise radial polynomials. For Wendland,
after exact angular integration the radial powers are nonnegative, including
kernel derivatives through order two. Their largest degree is at most
`2*a + m + 3 <= 32` (`m <= 11`), below the 24-point Gauss degree limit 47.
Cubic's outer Hessian includes Laurent radial terms and retains a numerical
radial integration error. These exact-arithmetic degree observations do not
certify floating-point cancellation, arbitrary geometry conditioning, or global
quadrature error. The provider always retains its numerical label and finite
comparison/independent-fixture gates.

The oracle in `sphere_reference.py` uses another calculation. It expands
Cartesian powers into axial factors and tangent moments. For tangent degree
`2k`, Gaussian Wick contractions of `P` divided by `2**k * k!` give the circular
angular average. Expanding `(1-t²)**k` and integrating each power of `t` produces
an exact rational polynomial in the cap, divided by pi. Substituting the cap
`A/r + B*r` produces exact rational Laurent coefficients. The kernel's `1/pi`
normalization cancels the angular pi. Rational powers are summed exactly before
70-digit conversion; possible logarithmic terms receive extra precision for
near-centre conditioning. No native quadrature nodes enter the oracle.

Physical outputs scale once by `h**Request.scale`; requests differentiate only
the kernel. Complete derivatives of translated monomials additionally subtract
`alpha[j] * M_(alpha-e_j)`. The provider integrates volume and does not compute
a sphere surface-area measure, curved-element FEM basis, shape adjoint or
pressure/porosity law. The existing analytic 28-channel sphere baseline remains
a separate supported API.
