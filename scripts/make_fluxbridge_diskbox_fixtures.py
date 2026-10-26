"""Establish independent 70-digit disk and box fixtures before native ports."""
import json
from pathlib import Path
import mpmath as mp
from warpSPHBoundaries.fluxbridge.contract import baseline,Request,monomials,KERNELS
from warpSPHBoundaries.fluxbridge.disk_reference import invariants
from warpSPHBoundaries.fluxbridge.radial_reference import select
from warpSPHBoundaries.fluxbridge.reference import evaluate


def main():
    cases=[]
    shapes=[(r,d) for r,d in [
        (".02","-.02"),(".02","0"),(".02",".2"),
        (".1","-.1"),(".1","-.09999999"),(".1","0"),
        (".5","-.5"),(".5","-.49999999"),(".5","-.2"),(".5","-.000000000001"),
        (".5","0"),(".5",".000000000001"),(".5",".5"),(".5",".999"),(".5","1"),
        ("1","-1"),("1","-.5"),("1","0"),("1",".5"),("1","1"),
        ("2","-1"),("2","-.5"),("2","0"),("2",".2"),("100",".2")]]
    with mp.workdps(70):
        n=[mp.mpf(3)/5,mp.mpf(4)/5,mp.mpf(0)]
        for r,d in shapes:
            for kernel in KERNELS:
                inv=invariants(d,r,kernel)
                cases.append(dict(kind="disk",radius=r,distance=d,kernel=kernel,normal=[mp.nstr(v,65) for v in n[:2]],
                    invariants=[mp.nstr(v,65) for v in inv],values=[mp.nstr(v,65) for v in select(inv,n,baseline(2))]))
        vertices=[["-.5","-.7"],[".9","-.7"],[".9",".6"],["-.5",".6"]]
        for query in [[0,0],["-.5",0],["-.5","-.7"],["1.9",0],[".9",".6"],["1.1",".7"]]:
            for kernel in KERNELS:
                req=baseline(2)
                cases.append(dict(kind="box",vertices=vertices,query=query,kernel=kernel,
                    requests=[dict(alpha=r.alpha,derivative=r.derivative) for r in req],
                    values=[mp.nstr(v,65) for v in evaluate(vertices,query,req,kernel)]))
        # Exact rectangular geometry also retains the full polynomial contract.
        derivatives=[(),(0,),(1,),(0,0),(0,1),(1,1)]
        req=tuple(Request(a,ds) for a in monomials(2,9) for ds in derivatives)
        for kernel in KERNELS:
            cases.append(dict(kind="box",vertices=vertices,query=[".1",".2"],kernel=kernel,
                requests=[dict(alpha=r.alpha,derivative=r.derivative) for r in req],
                values=[mp.nstr(v,65) for v in evaluate(vertices,[".1",".2"],req,kernel)]))
    path=Path(__file__).resolve().parents[1]/"tests/fixtures/fluxbridge-diskbox-v1.json"
    path.write_text(json.dumps(dict(schema=1,dps=70,cases=cases),indent=2)+"\n")
    print(path,len(cases),"cases",sum(len(c['values']) for c in cases),flush=True)


if __name__=="__main__":main()
