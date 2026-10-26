"""Independent 70-digit regular-potential near-plane acceptance fixtures."""
import json
from pathlib import Path
import mpmath as mp
from warpSPHBoundaries.fluxbridge import Request,baseline
from warpSPHBoundaries.fluxbridge.reference import evaluate


def main():
    tet=[[0,0,0],[2,0,0],[0,2,0],[0,0,2]]
    cases=[('near_face_inside',tet,['1e-16','.2','.2'],1),
        ('near_face_outside',tet,['-1e-16','.2','.2'],1),
        ('near_edge_inside',tet,['1e-16','1e-16','.2'],1),
        ('near_edge_outside',tet,['-1e-16','-1e-16','.2'],1),
        ('near_vertex_inside',tet,['1e-16']*3,1),
        ('near_vertex_outside',tet,['-1e-16']*3,1),
        ('switch_below',tet,['.999e-12','.2','.2'],1),
        ('switch_above',tet,['1.001e-12','.2','.2'],1),
        ('slanted_face_roundoff',tet,['.6666666666666666']*3,1),
        ('wall_grid_roundoff',[[0,'-.2','-.5'],[0,'.2','-.5'],[0,'.2','.5'],[1,'.2','.5']],
            [0,'.19999999999999996',0],'.6')]
    requests=baseline(3)+(Request((3,3,3)),Request((4,2,3),(0,2)),Request((2,4,3),(1,)))
    output=[]
    for label,vertices,query,h in cases:
        for kernel in ('w2','w4','w6','cubic'):
            with mp.workdps(70):values=evaluate(vertices,query,requests,kernel,h=h)
            output.append(dict(label=label,vertices=vertices,query=query,h=h,kernel=kernel,
                requests=[dict(alpha=r.alpha,derivative=r.derivative) for r in requests],
                values=[mp.nstr(v,65) for v in values]))
            print(label,kernel,flush=True)
    target=Path(__file__).resolve().parents[1]/'tests/fixtures/fluxbridge-nearplane-v1.json'
    target.write_text(json.dumps(dict(schema=1,dps=70,reference='independent arbitrary-precision mesh flux',cases=output),indent=2)+'\n')
    print(target,flush=True)


if __name__=='__main__':main()
