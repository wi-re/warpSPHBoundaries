"""Freeze weighted sphere cap references before implementing native quadrature."""
from dataclasses import asdict
from fractions import Fraction as F
import json
from pathlib import Path
import mpmath as mp
from warpSPHBoundaries.fluxbridge import Request, baseline
from warpSPHBoundaries.fluxbridge.contract import monomials, KERNELS
from warpSPHBoundaries.fluxbridge.sphere_reference import evaluate


def main():
    derivatives = [(),(0,),(1,),(2,),(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
    full = [Request(a,ds) for a in monomials(3,9) for ds in derivatives]
    sparse = list(dict.fromkeys([*baseline(3),*[Request(a,ds) for a in
        ((9,0,0),(0,9,0),(0,0,9),(3,3,3),(2,3,4),(4,2,3)) for ds in derivatives]]))
    configurations = [
        ('surface','.5','0',full), ('centre','.05','-.05',sparse),
        ('near-centre','.5','-.499999',sparse), ('interior','.5','-.2',sparse),
        ('exterior','.5','.5',sparse), ('small','.02','.2',sparse),
        ('large','100','.2',sparse), ('full','2','-1',sparse),
        ('near-tangent','.5','.999',sparse), ('tangent','.5','1',sparse),
        ('empty','.5','2',sparse),
    ]
    cases=[];checks=0
    previous=json.loads((Path(__file__).resolve().parents[1]/'tests/fixtures/fluxbridge-radial-v1.json').read_text())['cases']
    with mp.workdps(70):
        normal=(F(2,3),F(-1,3),F(2,3))
        for label,radius,distance,requests in configurations:
            for kernel in KERNELS:
                values=evaluate(distance,radius,requests,kernel,normal)
                old=next((c for c in previous if c['kind']=='sphere' and c['kernel']==kernel
                    and F(c['radius'])==F(radius) and F(c['distance'])==F(distance)),None)
                if old is not None:
                    mapping=dict(zip(requests,values))
                    for r,expected in zip(baseline(3),old['values']):
                        error=abs(mapping[r]-mp.mpf(expected))/(1+abs(mp.mpf(expected)))
                        if error>mp.mpf('1e-55'):raise RuntimeError((label,kernel,r,str(error)))
                        checks+=1
                cases.append(dict(label=label,radius=radius,distance=distance,kernel=kernel,
                    normal=[mp.nstr(mp.mpf(x.numerator)/x.denominator,65) for x in normal],
                    requests=[asdict(r) for r in requests],values=[mp.nstr(v,65) for v in values]))
                print(label,kernel,len(values),flush=True)
    path=Path(__file__).resolve().parents[1]/'tests/fixtures/fluxbridge-sphere-v1.json'
    path.write_text(json.dumps(dict(schema=1,dps=70,reference='rational Cartesian cap moments and radial Laurent antiderivatives',independent_baseline_checks=checks,cases=cases),indent=2)+'\n')
    print(path,len(cases),sum(len(c['values']) for c in cases),'channels',flush=True)


if __name__=='__main__': main()
