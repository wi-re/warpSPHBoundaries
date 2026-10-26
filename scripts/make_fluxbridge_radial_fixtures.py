"""Generate independent 70-digit fixtures before porting native special forms."""
import json
from pathlib import Path
import mpmath as mp
from warpSPHBoundaries.fluxbridge.radial_reference import invariants,select
from warpSPHBoundaries.fluxbridge.contract import baseline,KERNELS


def main():
    cases=[]
    geometries=[("plane",d,None) for d in ("-2","-1","-0.5","0","0.2","0.5","0.999","1","2")]
    geometries += [("sphere",d,r) for r,d in [
        ("0.05","-0.05"),("0.05","0"),("0.05","0.2"),
        ("0.5","-0.5"),("0.5","-0.499999"),("0.5","-0.2"),("0.5","0"),("0.5","0.5"),
        ("1","-1"),("1","-0.5"),("1","0"),("1","0.5"),("1","0.999"),("1","1"),
        ("2","-1"),("2","-0.5"),("2","0"),("2","0.2"),("100","0.2")]]
    with mp.workdps(70):
        normal=(mp.mpf(2)/3,-mp.mpf(1)/3,mp.mpf(2)/3)
        for kind,d,r in geometries:
            for kernel in KERNELS:
                values=invariants(d,kernel,r)
                cases.append(dict(kind=kind,distance=d,radius=r,kernel=kernel,
                    normal=[mp.nstr(x,65) for x in normal],
                    invariants=[mp.nstr(x,65) for x in values],
                    values=[mp.nstr(x,65) for x in select(values,normal,baseline(3))]))
    path=Path(__file__).resolve().parents[1]/"tests/fixtures/fluxbridge-radial-v1.json"
    path.write_text(json.dumps(dict(schema=1,dps=70,cases=cases),indent=2)+"\n")
    print(path,len(cases),"cases",len(cases)*len(baseline(3)),"channels")


if __name__=="__main__":
    main()
