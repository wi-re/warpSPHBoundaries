"""Evaluators for the Maple-generated closed forms in results/symbolic/.

The export files are produced by maple/01_planar.mpl and maple/03_sphere.mpl.
Two result types:
  * polynomials in d (3-D planar): stored as coefficient lists, evaluated
    EXACTLY with fractions.Fraction;
  * single-line Maple expressions (2-D planar): translated to mpmath and
    evaluated at arbitrary precision.

Conventions: unit support h=1, particle outside the boundary, d in [0,1].
"""
import re
from fractions import Fraction
from pathlib import Path

import mpmath as mp

REPO_ROOT = Path(__file__).resolve().parents[2]
PLANAR_EXPORT = REPO_ROOT / "results" / "symbolic" / "planar_export.txt"
SPHERE_EXPORT = REPO_ROOT / "results" / "symbolic" / "sphere_export.txt"

_planar_cache = None
_sphere_cache = None


# ------------------------------------------------------------------ parsing --
def _parse_pi_frac(s: str) -> Fraction:
    """'80/7/Pi' -> Fraction(80, 7); left-associative, Pi dropped."""
    toks = [t for t in s.split("/") if t != "Pi"]
    v = Fraction(toks[0])
    for t in toks[1:]:
        v /= Fraction(t)
    return v


def _parse_triples(s: str) -> dict:
    """'i:j:c i:j:c ...' -> {(i, j): Fraction(c)}."""
    out = {}
    for tok in s.split():
        i, j, c = tok.split(":")
        out[(int(i), int(j))] = Fraction(c)
    return out


def _load_planar():
    global _planar_cache
    if _planar_cache is not None:
        return _planar_cache
    data = {"meta": {}, "3d": {}, "2d": {}}
    for raw in PLANAR_EXPORT.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("kernel "):
            parts = line.split()
            k = parts[1]
            data["meta"][k] = {}
            for t in parts[2:]:
                if t.startswith("n="):
                    data["meta"][k]["n"] = int(t[2:])
                elif t.startswith("P="):
                    data["meta"][k]["P"] = t[2:]
                elif t.startswith("C2="):
                    data["meta"][k]["c2"] = _parse_pi_frac(t[3:])
                elif t.startswith("C3="):
                    data["meta"][k]["c3"] = _parse_pi_frac(t[3:])
        elif line.startswith("planar3d "):
            parts = line.split()
            k = parts[1]
            if k == "cubic":
                branch = parts[2]
                coeffs = [Fraction(x) for x in parts[4:]]
            else:
                branch = "all"
                coeffs = [Fraction(x) for x in parts[3:]]
            data["3d"][(k, branch)] = coeffs
        elif line.startswith("planar2d "):
            parts = line.split(maxsplit=3)
            if parts[1] == "cubic":
                data["2d"][("cubic", parts[2].rstrip(":"))] = parts[3].strip()
            else:
                data["2d"][(parts[1].rstrip(":"), "all")] = parts[2].strip()
    _planar_cache = data
    return data


def _load_sphere():
    global _sphere_cache
    if _sphere_cache is not None:
        return _sphere_cache
    data = {"meta": {}, "b1": {}, "b2": {}}
    for raw in SPHERE_EXPORT.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("kernel "):
            parts = line.split()
            data["meta"][parts[1]] = {}
            for t in parts[2:]:
                if t.startswith("n="):
                    data["meta"][parts[1]]["n"] = int(t[2:])
                elif t.startswith("C3="):
                    data["meta"][parts[1]]["c3"] = _parse_pi_frac(t[3:])
        elif line.startswith("sph b1 ") or line.startswith("sph b2 "):
            branch = "b1" if line.startswith("sph b1 ") else "b2"
            rest = line.split(None, 2)[2]      # "w2: 1:8:3/2 ..."
            kname, _, triples = rest.partition(": ")
            data[branch][kname] = _parse_triples(triples)
    _sphere_cache = data
    return data


# -------------------------------------------------------------- evaluators --
_FRAC_RE = re.compile(r"\d+/\d+")


def _eval_maple_expr(expr: str, d: mp.mpf, dps: int = 40) -> mp.mpf:
    """Translate a Maple %a expression (Pi, arccos, ln, ^) and evaluate it.

    Rational literals a/b are mapped to mp.mpf("a/b"): a bare Python a/b
    would be a double-precision float and would pollute the mpf result.
    """
    s = (expr.replace("Pi", "mp.pi")
             .replace("arccos", "mp.acos")
             .replace("ln(", "mp.log(")
             .replace("^(1/2)", "**(mp.mpf(1)/2)")
             .replace("^", "**"))
    s = _FRAC_RE.sub(lambda m: f'mp.mpf("{m.group(0)}")', s)
    with mp.workdps(dps):
        return mp.mpf(eval(s, {"mp": mp, "d": d}))


def _eval_poly(coeffs_desc, x: Fraction) -> Fraction:
    acc = Fraction(0)
    for c in coeffs_desc:
        acc = acc * x + c
    return acc


def planar2d(kernel_name: str, d, dps: int = 40) -> mp.mpf:
    """2-D planar boundary integral lambda_2(d) (unit support, d in [0,1])."""
    data = _load_planar()
    d = mp.mpf(d)
    if not mp.mpf(0) <= d <= mp.mpf(1):
        raise ValueError("d must be in [0, 1]")
    if kernel_name == "cubic":
        branch = "A" if d <= mp.mpf(1) / 2 else "B"
    else:
        branch = "all"
    return _eval_maple_expr(data["2d"][(kernel_name, branch)], d, dps)


def planar3d(kernel_name: str, d) -> Fraction:
    """3-D planar boundary integral lambda_3(d), EXACT rational in d."""
    data = _load_planar()
    d = Fraction(d)
    if not Fraction(0) <= d <= Fraction(1):
        raise ValueError("d must be in [0, 1]")
    if kernel_name == "cubic":
        branch = "b1" if d <= Fraction(1, 2) else "b2"
    else:
        branch = "all"
    return _eval_poly(data["3d"][(kernel_name, branch)], d)


def sphere(kernel_name: str, R, d, branch: str | None = None) -> Fraction:
    """Solid-sphere boundary integral lambda_sph(R,d), EXACT rational.

    branch: None (auto), "b1" (2R+d >= 1) or "b2" (2R+d < 1).
    """
    data = _load_sphere()
    R = Fraction(R)
    d = Fraction(d)
    if branch is None:
        branch = "b1" if 2 * R + d >= 1 else "b2"
    terms = data[branch][kernel_name]
    N = sum(c * R ** i * d ** j for (i, j), c in terms.items())
    return N / (R + d)
