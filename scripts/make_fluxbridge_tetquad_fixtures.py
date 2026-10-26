"""Additional 70-digit tetrahedral quadrature gates from the independent flux oracle."""
import json
from pathlib import Path
import mpmath as mp
from warpSPHBoundaries.fluxbridge import Request, baseline
from warpSPHBoundaries.fluxbridge.reference import evaluate


def main():
    tet=[[0,0,0],[2,0,0],[0,2,0],[0,0,2]]
    cases=[
        ("face",tet,[0,".2",".2"],1),
        ("near_face",tet,["-1e-10",".2",".2"],1),
        ("face_tangent",tet,[-1,".4",".4"],1),
        ("near_tangent",tet,["-.999999",".4",".4"],1),
        ("edge_tangent",tet,["-.6","-.8",".2"],1),
        ("vertex_tangent",tet,["-0.3333333333333333333333333333333333333333333333333333333333333333333333",
                               "-0.6666666666666666666666666666666666666666666666666666666666666666666667",
                               "-0.6666666666666666666666666666666666666666666666666666666666666666666667"],1),
        ("disjoint_in_aabb",tet,["1.5","1.5","1.5"],".5"),
        ("full_support",[[-4,-4,-4],[12,-4,-4],[-4,12,-4],[-4,-4,12]],[0,0,0],1),
        ("sliver",[[0,0,0],[2,0,0],[0,2,0],[0,0,".001"]],[".2",".2",".0005"],1),
        ("scalene",[["-.3","-.2","-.1"],["1.7","-.15",".25"],[".125","1.6","-.125"],
                     ["-.1",".1","1.5"]],[".2",".25",".1"],1),
    ]
    requests=baseline(3)+(Request((3,3,3)),Request((4,2,3),(0,2)),Request((2,4,3),(1,)))
    out=[]
    for label,vertices,query,h in cases:
        for kernel in ("w2","w4","w6","cubic"):
            with mp.workdps(70):
                values=evaluate(vertices,query,requests,kernel,h=h)
            out.append(dict(label=label,vertices=vertices,query=query,h=h,kernel=kernel,
                requests=[dict(alpha=r.alpha,derivative=r.derivative) for r in requests],
                values=[mp.nstr(v,65) for v in values]))
            print(label,kernel,flush=True)
    path=Path(__file__).resolve().parents[1]/"tests/fixtures/fluxbridge-tetquad-v1.json"
    path.write_text(json.dumps(dict(schema=1,dps=70,reference="independent arbitrary-precision mesh flux",cases=out),indent=2)+"\n")
    print(path,flush=True)


if __name__=="__main__":main()
