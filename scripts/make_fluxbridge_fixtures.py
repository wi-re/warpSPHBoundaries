"""Rebuild 70-digit independent fixtures before native Warp evaluation."""
import json
from pathlib import Path
import mpmath as mp
from warpSPHBoundaries.fluxbridge.contract import Request, baseline, monomials
from warpSPHBoundaries.fluxbridge.reference import evaluate


def main():
    cases=[]
    for d, vertices in [(2, [[0,0],[2,0],[0,2]]), (3, [[0,0,0],[2,0,0],[0,2,0],[0,0,2]])]:
        for query in [[0]*d, ["0.2"]*d, ["-0.1"]+["0.2"]*(d-1), [1]+[0]*(d-1), [3]*d]:
            for kernel in ("w2","w4","w6","cubic"):
                requests=baseline(d)+(Request((9,)+(0,)*(d-1)),Request((3,)*d),)
                with mp.workdps(70):
                    values=evaluate(vertices,query,requests,kernel)
                    cases.append(dict(dimension=d,vertices=vertices,query=query,kernel=kernel,measure="volume",h=1,
                        requests=[dict(alpha=r.alpha,derivative=r.derivative) for r in requests],
                        values=[mp.nstr(v,65) for v in values]))
    vertices=[[0,0,"0.25"],[2,0,"0.25"],[0,2,"0.25"]]
    for kernel in ("w2","w4","w6","cubic"):
        with mp.workdps(70):
            requests=baseline(3)
            values=evaluate(vertices,["0.2","0.2",0],requests,kernel,measure="surface")
            cases.append(dict(dimension=3,vertices=vertices,query=["0.2","0.2",0],kernel=kernel,measure="surface",h=1,
                requests=[dict(alpha=r.alpha,derivative=r.derivative) for r in requests],values=[mp.nstr(v,65) for v in values]))
    # Full advertised volume degree/derivative coverage, beyond viewer channels.
    for d in (2,3):
        vertices=[[0]*d]+[[2*int(i==j) for j in range(d)] for i in range(d)]
        derivatives=[()]+[(j,) for j in range(d)]+[(i,j) for i in range(d) for j in range(i,d)]
        requests=tuple(Request(a,ds) for a in monomials(d,9) for ds in derivatives)
        for kernel in ("w2","w4","w6","cubic"):
            with mp.workdps(70):
                values=evaluate(vertices,["0.2"]*d,requests,kernel)
                cases.append(dict(dimension=d,vertices=vertices,query=["0.2"]*d,kernel=kernel,measure="volume",h=1,
                    requests=[dict(alpha=r.alpha,derivative=r.derivative) for r in requests],values=[mp.nstr(v,65) for v in values]))
    path=Path(__file__).resolve().parents[1]/"tests/fixtures/fluxbridge-v1.json"
    path.write_text(json.dumps(dict(schema=1,dps=70,cases=cases),indent=2)+"\n")
    print(path, len(cases), "cases")

if __name__ == "__main__":
    main()
