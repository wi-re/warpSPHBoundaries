"""Generate static browser profiles from the exact native rational plans.

Runtime browsers need only this checked-in JSON; no Python/Warp or package/CDN
imports. Requests are assembled in JS; profiles cover degree nine and DS <= 2.
"""
import json
from pathlib import Path
from warpSPHBoundaries.fluxbridge import Request
from warpSPHBoundaries.fluxbridge.conditioned2d import ConditionedPlan
from warpSPHBoundaries.fluxbridge.tet_quadrature import QuadraturePlan

TARGET = Path(__file__).resolve().parents[2]/'src/warpSPHBoundaries/visualizer/static/fluxbridge-profiles.json'


def build():
    out = dict(schema_version=1, source='exact rational ConditionedPlan / QuadraturePlan before fp32 rounding', dimensions={})
    for d, cls in ((2, ConditionedPlan), (3, QuadraturePlan)):
        requests = [Request((k,)+(0,)*(d-1), ds) for k in range(10) for ds in ((), (0,), (0,0))]
        out['dimensions'][str(d)] = {}
        for kernel in ('w2','w4','w6','cubic'):
            p = cls(d, requests, kernel)
            out['dimensions'][str(d)][kernel] = dict(profiles=p.profiles, bounds=p.bounds,
                coefficients=p.coefficients, constants=p.constants,
                inner_scaled=getattr(p,'inner_scaled',[]))
    return out


if __name__ == '__main__':
    TARGET.write_text(json.dumps(build(), separators=(',',':'), allow_nan=False)+'\n')
    print(TARGET)
