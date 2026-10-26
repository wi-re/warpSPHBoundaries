"""Reference fixtures before the analytic native plane/strip port."""
from fractions import Fraction as F
import json
from pathlib import Path
import mpmath as mp
from warpSPHBoundaries.fluxbridge.contract import baseline,monomials,Request,KERNELS
from warpSPHBoundaries.fluxbridge.reference import evaluate as mesh_reference
from warpSPHBoundaries.fluxbridge.planar_reference import evaluate as shell_reference

FACES=((0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
       (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7))


def vertices(d,lo,hi,rotated):
    if d==2:
        n=(F(3,5),F(4,5)) if rotated else (F(0),F(1))
        u=(n[1],-n[0])
        v=[[t*u[j]+z*n[j] for j in range(2)] for t,z in ((-2,lo),(2,lo),(2,hi),(-2,hi))]
    else:
        n=(F(2,3),F(-1,3),F(2,3)) if rotated else (F(0),F(0),F(1))
        u=(F(1,3),F(-2,3),F(-2,3)) if rotated else (F(1),F(0),F(0))
        w=(F(2,3),F(2,3),F(-1,3)) if rotated else (F(0),F(1),F(0))
        v=[[x*u[j]+y*w[j]+z*n[j] for j in range(3)] for z in (lo,hi) for x,y in ((-2,-2),(2,-2),(2,2),(-2,2))]
    return v,n


def main():
    cases=[]
    shapes=[('plane',F(d),None) for d in ('-2','-1','-.5','0','.2','.999','1','2')]
    shapes += [('strip',F(d),F(a)) for d,a in (('-2','.2'),('-.2','.2'),('0','.2'),('.15','.2'),('.8','.2'),('1.2','.2'),('0','.02'))]
    with mp.workdps(70):
        for dimension in (2,3):
            high=tuple(Request(a,ds) for a in monomials(dimension,9) for ds in
                ([()]+[(i,) for i in range(dimension)]+[(i,j) for i in range(dimension) for j in range(i,dimension)]))
            groups=[(kind,d,a,True,baseline(dimension)) for kind,d,a in shapes]
            groups += [('plane',F('.2'),None,False,high),('strip',F('.1'),F('.2'),False,high)]
            for kind,d,a,rotated,req in groups:
                lo=max(F(-2),-d-a) if a is not None else F(-2)
                hi=min(F(2),-d+a) if a is not None else min(F(2),-d)
                for kernel in KERNELS:
                    if hi<=-1 or lo>=1:
                        values=[mp.mpf(0)]*len(req)
                        _,n=vertices(dimension,F(-2),F(2),rotated)
                    else:
                        v,n=vertices(dimension,lo,hi,rotated)
                        if dimension==3 and not rotated:
                            values=shell_reference(lo,hi,req,kernel)
                        else:
                            values=mesh_reference(v,[0]*dimension,req,kernel,faces=FACES if dimension==3 else None)
                    cases.append(dict(kind=kind,dimension=dimension,distance=str(float(d)),half_width=None if a is None else str(float(a)),
                        normal=[float(x) for x in n],kernel=kernel,requests=[dict(alpha=r.alpha,derivative=r.derivative) for r in req],
                        values=[mp.nstr(x,65) for x in values]))
                    print('reference',dimension,kind,float(d),kernel,len(req),flush=True)
    path=Path(__file__).resolve().parents[1]/'tests/fixtures/fluxbridge-planar-v1.json'
    path.write_text(json.dumps(dict(schema=1,dps=70,cases=cases),indent=2)+'\n')
    print(path,len(cases),sum(len(c['values']) for c in cases),flush=True)


if __name__=='__main__':main()
