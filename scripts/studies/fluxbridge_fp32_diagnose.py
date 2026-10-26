"""Separate primitive, coefficient and reduction errors in analytic fp32 plans.

Offline fp64 reconstruction is diagnostic only, never a device fallback.
The independent frozen fixture remains the accuracy reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from warpSPHBoundaries.fluxbridge import Geometry,Evaluator,Request

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    fixture=ROOT/'tests/fixtures/fluxbridge-v1.json'
    cases=json.loads(fixture.read_text())['cases'];rows=[]
    for case_id,case in enumerate(cases):
        requests=[Request(**r) for r in case['requests']]
        geometry=Geometry([case['vertices']],case['dimension'],case['measure'])
        for compact in (False,True):
            evaluators=[Evaluator(geometry,requests,kernel=case['kernel'],precision=p,compact=compact) for p in ('float64','float32')]
            batches=[e.prepare([case['query']],case['h'],exhaustive=True) for e in evaluators]
            for b in batches:b.launch()
            w64,w32=[b.work.numpy()[0].astype(float) for b in batches]
            native=batches[1].numpy()[0];ref=np.array(case['values'],float)
            scales=np.array([case['h']**r.scale for r in requests]);native=native/scales;ref=ref/scales
            arrays=evaluators[0].host_plan.arrays();terms=arrays['terms'];offsets=arrays['offsets']
            nf,nb=len(arrays['f']),len(arrays['b'])
            for column,request in enumerate(requests):
                t=terms[offsets[column]:offsets[column+1]]
                indices=t[:,1].astype(int)+np.where(t[:,0]==1,nf+nb,np.where(t[:,0]==2,nf,0))
                c64=t[:,2];c32=c64.astype(np.float32).astype(float)
                exact=float(c64@w64[indices]);primitive=float(c64@w32[indices]);coefficient=float(c32@w32[indices])
                denominator=1+abs(ref[column]);error=abs(native[column]-ref[column])/denominator
                if error>1e-5:
                    rows.append(dict(case=case_id,dimension=case['dimension'],measure=case['measure'],kernel=case['kernel'],
                        query=case['query'],compact=compact,alpha=request.alpha,derivative=request.derivative,column=column,
                        native=float(native[column]),reference=float(ref[column]),mixed_error=float(error),
                        fp64_reference_error=abs(exact-ref[column])/denominator,
                        primitive_error=abs(primitive-exact)/denominator,
                        coefficient_error=abs(coefficient-primitive)/denominator,
                        reduction_error=abs(native[column]-coefficient)/denominator,
                        signed_term_amplification=float(np.sum(abs(c64*w32[indices]))/denominator)))
    rows.sort(key=lambda r:r['mixed_error'],reverse=True)
    paths=list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*.py'))+[Path(__file__),fixture]
    report=dict(schema=1,device='cpu',precision='float32',diagnostic_reference='offline fp64; not device arithmetic',
        failing_channels=len(rows),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},rows=rows)
    (args.output/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in rows[:12]:print(json.dumps(row))


if __name__=='__main__':main()
