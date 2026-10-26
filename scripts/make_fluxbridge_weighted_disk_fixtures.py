"""Freeze 70-digit weighted-circle flux references before the native port."""
from dataclasses import asdict
from fractions import Fraction as F
import json
from pathlib import Path
import mpmath as mp
from warpSPHBoundaries.fluxbridge import Request,baseline,monomials
from warpSPHBoundaries.fluxbridge.contract import KERNELS
from warpSPHBoundaries.fluxbridge.weighted_disk_reference import evaluate


def main():
    root=Path(__file__).resolve().parents[1]
    previous=json.loads((root/'tests/fixtures/fluxbridge-diskbox-v1.json').read_text())['cases']
    derivatives=[(),(0,),(1,),(0,0),(0,1),(1,1)]
    full=[Request(a,ds) for a in monomials(2,9) for ds in derivatives]
    sparse=list(dict.fromkeys([*baseline(2),*[Request(a,ds) for a in [(9,0),(0,9),(4,5),(5,4)] for ds in derivatives]]))
    configurations=[('surface','.5','0',full),('interior','.5','-.2',sparse),
        ('centre','.1','-.1',sparse),('near-centre','.5','-.49999999',sparse),
        ('near-surface','.5','.00000001',sparse),('small','.02','.2',sparse),
        ('large','100','.2',sparse),('full','2','-1',sparse),
        ('near-tangent','.5','.999',sparse),('tangent','.5','1',sparse)]
    cases=[];checks=0
    with mp.workdps(70):
        for label,radius,distance,requests in configurations:
            for kernel in KERNELS:
                values=evaluate(distance,radius,requests,kernel)
                old=next((c for c in previous if c['kind']=='disk' and c['kernel']==kernel
                    and F(c['radius'])==F(radius) and F(c['distance'])==F(distance)),None)
                if old:
                    mapping=dict(zip(requests,values))
                    for r,v in zip(baseline(2),old['values']):
                        defect=abs(mapping[r]-mp.mpf(v))/(1+abs(mp.mpf(v)))
                        if defect>mp.mpf('1e-50'):raise RuntimeError((label,kernel,r,str(defect)))
                        checks+=1
                cases.append(dict(label=label,radius=radius,distance=distance,kernel=kernel,normal=['.6','.8'],
                    requests=[asdict(r) for r in requests],values=[mp.nstr(v,65) for v in values]))
                print(label,kernel,len(values),flush=True)
    path=root/'tests/fixtures/fluxbridge-weighted-disk-v1.json'
    path.write_text(json.dumps(dict(schema=1,dps=70,independent_baseline_checks=checks,
        reference='whole-circle regular flux with exact radial primitives; independent volume-shell baseline checks',cases=cases),indent=2)+'\n')
    print(path,sum(len(c['values']) for c in cases),'channels',flush=True)


if __name__=='__main__':main()
