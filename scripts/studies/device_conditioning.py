"""N1/M1 executed Scene and fused CPU/CUDA checkpoint; no localhost required."""
import argparse,json,time
from pathlib import Path

from warpSPHBoundaries.validation.paths import evidence_path
import numpy as np
from warpSPHBoundaries.scene.scene import SurfaceRep
from warpSPHBoundaries.validation.adversarial import Measurement,scene_channels
from warpSPHBoundaries.validation.robustness import save_measurements,collect_findings
from warpSPHBoundaries.validation.audit_report import generate,method_data
from warpSPHBoundaries.validation.report_view import accessible_report


def run(baseline,output,device,dpi=300):
    baseline,output=Path(baseline),Path(output)
    if baseline.resolve()==output.resolve(): raise ValueError('Preserve reference checkpoint')
    output.mkdir(parents=True,exist_ok=True)
    source=json.loads((baseline/'summary.json').read_text())
    records=[]; timings=[]; failures=[]
    for row in source['records']:
        folder=output/row['kernel']/row['case'];folder.mkdir(parents=True,exist_ok=True)
        with np.load(baseline/row['kernel']/row['case']/'data.npz') as a:
            vertices,points,target,uncertainty=a['vertices'],a['points'],a['target'],a['reference_estimate']
        methods=[];diagnostics={}
        for method,fused,stable in [('scene-device',False,False),('fused-device-mono',True,False),('fused-device-cheb',True,True)]:
            start=time.perf_counter()
            values,estimate,status=scene_channels(SurfaceRep.polygon(vertices),points,row['kernel'],device,
                fused=fused,stable=stable,second=True,return_diagnostics=True)
            if stable:
                refined=scene_channels(SurfaceRep.polygon(vertices),points,row['kernel'],device,
                    fused=True,stable=(16,12),second=True)
                # Area rows keep their inner/outer estimate. Edge rows receive
                # an empirical 8x6 versus 16x12 Chebyshev refinement difference.
                estimate=np.where((status!=0)[:,None],estimate,abs(values-refined))
            elapsed=time.perf_counter()-start
            timings.append(dict(case=row['case'],kernel=row['kernel'],method=method,seconds=elapsed,queries=len(points)))
            diagnostics[method+'_status']=status
            failed=int((status<0).sum())
            if failed: failures.append(dict(case=row['case'],kernel=row['kernel'],method=method,failed=failed))
            methods.append(Measurement(method,values,target,estimate=estimate,status=status,
                note='Executed production path; all twelve moments. Static triangle conditioning; empirical device refinement estimate.'))
        methods=save_measurements(folder,target,uncertainty,methods,vertices=vertices,points=points,**diagnostics)
        records.append(dict(row,methods=methods))
        print(device,row['kernel'],row['case'], 'failed',sum(int((s<0).sum()) for s in diagnostics.values()),flush=True)
    summary=dict(source,records=records,device=device,checkpoint='N1/M1 executed device integration',specialized=[],representation=[])
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (output/'independent-oracle-checks.json').write_text((baseline/'independent-oracle-checks.json').read_text())
    (output/'coverage.json').write_text(json.dumps(dict(explicit_limits=[
        'Executed Scene and fixed-adjacency FusedWall paths, 21 original geometries, four base kernels and all twelve moments.',
        '70/45-digit analytic references and independent polar spot checks inherited from the conditioning checkpoint.',
        'Only single closed-triangle SurfaceRep geometry uses the new device area fallback; general polygons retain edge identities.',
        'Static geometry metadata is prepared once on the host; evaluation positions, supports, masks, estimates and status stay on device.',
        'Refinement differences are empirical, not certified bounds. Negative area status produces nonfinite outputs, never an accurate zero.',
        'Chebyshev edge estimates compare 8 nodes/6 panels with 16 nodes/12 panels; area rows retain their separate inner/outer estimate.',
        'Second moments are exposed for explicit SurfaceRep geometry. SDF, lookup and volume representation completion is separate.',
        'Times include adjacency, allocation, synchronization, and initial compilation; these are audit timings, not simulation throughput benchmarks.']),indent=2)+'\n')
    finish(output,summary,device,timings,failures,dpi)


def finish(output,summary,device,timings,failures,dpi=300):
    output=Path(output)
    collect_findings(summary,output)
    stats={name:method_data(summary['records'],name) for name in ['scene-device','fused-device-mono','fused-device-cheb']}
    counts={name:{k:list(d['outcomes'].values()).count(k) for k in ['within','exceeds','nonfinite']} for name,d in stats.items()}
    checkpoint=dict(device=device,counts=counts,worst={k:d['worst'] for k,d in stats.items()},quadrature_failures=failures,timings=timings)
    (output/'checkpoint.json').write_text(json.dumps(checkpoint,indent=2,allow_nan=False)+'\n')
    path=generate(output,dpi)
    text=path.read_text();title,body=text.split('\n',1)
    intro=['## N1/M1 executed device checkpoint','',f'Device: `{device}`. Fresh production evaluations with all twelve moments. No local web server is needed to read this Markdown report or open its figures.','',
           '| Path | Within ε32 | Exceeds | Nonfinite |','|---|---:|---:|---:|']
    for name,c in counts.items():intro.append(f"| {name} | {c['within']} | {c['exceeds']} | {c['nonfinite']} |")
    intro+=['','[Raw checkpoint and audit timings](checkpoint.json). Original geometry and references are preserved.','']
    path.write_text(title+'\n\n'+'\n'.join(intro)+body);accessible_report(path)
    print(json.dumps(dict(counts=counts,failures=failures),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',default=evidence_path('triangle-conditioning'))
    p.add_argument('--output',default=evidence_path('device-conditioning-cpu'))
    p.add_argument('--device',default='cpu');p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args()
    if a.report_only:
        folder=Path(a.output);summary=json.loads((folder/'summary.json').read_text());c=json.loads((folder/'checkpoint.json').read_text())
        finish(folder,summary,c['device'],c['timings'],c['quadrature_failures'],a.dpi)
    else:
        run(a.baseline,a.output,a.device,a.dpi)
