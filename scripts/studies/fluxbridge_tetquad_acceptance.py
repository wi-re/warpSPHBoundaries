"""Independent tetrahedral accuracy and complete degree-nine material costs."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
from itertools import product
import json
from math import comb
from pathlib import Path
import mpmath as mp
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry,Evaluator,Request,AffineResistanceMaterial
from fluxbridge_shell_acceptance import timed

ROOT=Path(__file__).resolve().parents[2]


def sources():
    paths=list((ROOT/"src/warpSPHBoundaries/fluxbridge").glob("*.py"))+list((ROOT/"tests/fluxbridge").glob("*.py"))
    paths += [Path(__file__),ROOT/"scripts/studies/fluxbridge_shell_acceptance.py",
        ROOT/"scripts/make_fluxbridge_tetquad_fixtures.py",ROOT/"tests/fixtures/fluxbridge-v1.json",
        ROOT/"tests/fixtures/fluxbridge-tetquad-v1.json",ROOT/'tests/fixtures/fluxbridge-nearplane-v1.json',
        ROOT/'scripts/make_fluxbridge_nearplane_fixtures.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def fixture_cases():
    main=json.loads((ROOT/"tests/fixtures/fluxbridge-v1.json").read_text())["cases"]
    extra=json.loads((ROOT/"tests/fixtures/fluxbridge-tetquad-v1.json").read_text())["cases"]
    extra+=json.loads((ROOT/'tests/fixtures/fluxbridge-nearplane-v1.json').read_text())['cases']
    return [c for c in main if c["dimension"]==3 and c["measure"]=="volume"]+extra


def accuracy(device,preparation='host',nodes=16):
    rows=[]
    for precision,budget in (("float64",1e-14),("float32",1e-5)):
        for kernel in ("w2","w4","w6","cubic"):
            errors=[]; convergence=[]; promotions=0; cases=0; samples=0; rejected=[];orders={}
            for c in fixture_cases():
                if c["kernel"]!=kernel:continue
                req=[Request(**r) for r in c["requests"]]
                ev=Evaluator(Geometry([c["vertices"]],3),req,kernel=kernel,precision=precision,
                    method="tet_quadrature",device=device,preparation=preparation,nodes=nodes)
                try:
                    b=ev.prepare([c["query"]],c["h"],exhaustive=True);b.launch()
                except ValueError as error:
                    rejected.append(dict(label=c.get("label",str(c["query"])),reason=str(error)));continue
                value=b.numpy()[0];ref=np.array(c["values"],float)
                scales=np.array([float(c['h'])**r.scale for r in req]);value=value/scales;ref=ref/scales
                errors.extend((abs(value-ref)/(1+abs(ref))).tolist());cases+=1
                convergence.append(b.quadrature["worst_mixed_difference"])
                promotions+=b.quadrature["promotions"];samples=max(samples,b.quadrature["samples"])
                order=str(b.quadrature["nodes"]);orders[order]=orders.get(order,0)+1
            errors=np.array(errors)
            row=dict(precision=precision,kernel=kernel,cases=cases,channels=len(errors),budget=budget,
                worst_mixed_error=float(errors.max()),failing_channels=int(np.sum(~np.isfinite(errors)|(errors>budget))),
                worst_mixed_convergence=max(convergence),promotions=promotions,max_accepted_samples=samples,
                rejected_cases=len(rejected),rejections=rejected,fallback_count=0,requested_nodes=nodes,accepted_orders=orders)
            rows.append(row);print("accuracy",row,flush=True)
            if row["failing_channels"] or rejected:raise RuntimeError("tetrahedral accuracy gate failed")
    return rows


def material_reference(material,kernel):
    """70-digit frozen moments contracted with the actual packed coefficients."""
    main=json.loads((ROOT/"tests/fixtures/fluxbridge-v1.json").read_text())["cases"]
    c=next(c for c in main if c["dimension"]==3 and c["measure"]=="volume" and c["kernel"]==kernel and len(c["requests"])==2200)
    with mp.workdps(70):
        moments={tuple(r["alpha"]):mp.mpf(v) for r,v in zip(c["requests"],c["values"]) if not r["derivative"]}
        delta=[mp.mpf(".2")]*3;packed=[mp.mpf(0)]*int(np.prod(material.field.shape))
        for alpha,coeff in material.field.coefficients[0].items():
            integral=mp.mpf(0)
            for beta in product(*(range(a+1) for a in alpha)):
                weight=mp.mpf(1)
                for j in range(3):weight*=comb(alpha[j],beta[j])*delta[j]**(alpha[j]-beta[j])
                integral+=weight*moments[beta]
            for j in np.flatnonzero(coeff):packed[j]+=mp.mpf(float(coeff[j]))*integral
        return np.array([float(v) for v in packed])


def material_runtime(device,quick=False,preparation='host',nodes=16):
    geometry=Geometry([np.vstack([np.zeros(3),2*np.eye(3)])],3)
    rng=np.random.default_rng(52);factors=rng.normal(size=(1,20,3,3))
    D=factors@factors.swapaxes(-1,-2)+np.eye(3)*.2;controls=rng.normal(size=(1,20,3))
    velocity=rng.normal(size=(1,3));gradient=rng.normal(size=(1,3,3));masses=np.array([.7])
    material,construction=timed(lambda:AffineResistanceMaterial(geometry,D,controls,resistance_degree=3,velocity_degree=3),device,1)
    rows=[];reference_action=None
    for kernel in (("w2",) if quick else ("w2","w6")):
        packed_reference=material_reference(material,kernel)
        for method,precision in (("analytic","float64"),("tet_quadrature","float64"),("tet_quadrature","float32")):
            options=dict(kernel=kernel,method=method,precision=precision,device=device,
                preparation=preparation if method=='tet_quadrature' else 'host',nodes=nodes)
            b,prepare=timed(lambda:material.prepare([[.2]*3],1,**options),device,1)
            real=b.batch.evaluator.api.real
            u=wp.array(velocity,dtype=real,device=device);L=wp.array(gradient,dtype=real,device=device)
            mass=wp.array(masses,dtype=real,device=device)
            apply=lambda:b.apply_device(u,mass,L)
            _,first=timed(apply,device,1)
            _,warm=timed(apply,device,3,events=True)
            result=b.apply(velocity,masses,gradient)
            _,upload_readback=timed(lambda:b.apply(velocity,masses,gradient),device,1)
            _,complete=timed(lambda:material.prepare([[.2]*3],1,**options).apply(velocity,masses,gradient),device,1)
            packed=b.batch.numpy()[0]
            mixed=float(np.max(abs(packed-packed_reference)/(1+abs(packed_reference))))
            action=np.r_[result.fluid_force.ravel(),result.gradient_load.ravel(),result.reaction.ravel(),result.dissipation]
            if method=="analytic":reference_action=action
            difference=float(np.max(abs(action-reference_action)/(1+abs(reference_action))))
            work=float(np.sum(result.fluid_force*velocity)+np.sum(result.gradient_load*gradient)+np.sum(result.reaction*controls))
            work_error=abs(work+result.dissipation)/(1+abs(result.dissipation))
            balance=np.linalg.norm(result.fluid_force.sum(axis=0)+result.reaction.sum(axis=(0,1)))
            from warpSPHBoundaries.mesh3d.fem import expansion
            fem_nodes,_=expansion(geometry.elements[0].tolist(),[0,0,0],3)
            torque=float(np.linalg.norm(result.torque_defect(np.full((1,3),.2),np.array(fem_nodes,float)[None])))
            budget=1e-12 if precision=="float64" else 1e-5
            if method=="tet_quadrature" and max(mixed,work_error,balance,torque)>budget:
                raise RuntimeError(dict(packed_error=mixed,work_error=work_error,force_balance=balance,torque=torque))
            row=dict(method=method,precision=precision,kernel=kernel,queries=1,interactions=b.batch.count,
                resistance_degree=3,velocity_degree=3,test_degree=3,product_degree=material.field.degree,
                requested_moments=len(b.batch.evaluator.host_plan.requests),contracted_outputs=b.batch.columns,
                batch_bytes=b.bytes,field_batch_bytes=b.batch.bytes,material_construction=construction,checked_preparation=prepare,
                first_action=first,warm_action=warm,upload_action_readback=upload_readback,reprepare_and_apply=complete,
                packed_mixed_error_from_70_digit_moments=mixed,mixed_action_difference_from_analytic_fp64=difference,
                normalized_work_defect=work_error,force_balance_norm=float(balance),torque_defect_norm=torque,
                dissipation=result.dissipation,quadrature=getattr(b.batch,"quadrature",None),graph_replay=None)
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:apply()
                before=[a.numpy() for a in (b.force,b.gradient_load,b.reaction,b.power)]
                wp.capture_launch(capture.graph)
                row["graph_bitwise_equal"]=all(np.array_equal(a.numpy(),value) for a,value in zip((b.force,b.gradient_load,b.reaction,b.power),before))
                if not row["graph_bitwise_equal"]:raise RuntimeError("material graph output changed")
                _,row["graph_replay"]=timed(lambda:wp.capture_launch(capture.graph),device,3,events=True)
            rows.append(row)
            print("material",kernel,method,precision,"warm",warm["median_ms"],"complete",complete["median_ms"],"packed error",mixed,flush=True)
    return rows


def compile_only(precision,output,arch,preparation='host',nodes=16):
    import warp._src.build as build
    g=Geometry([np.vstack([np.zeros(3),2*np.eye(3)])],3)
    m=AffineResistanceMaterial(g,[[np.eye(3)]],np.zeros((1,4,3)))
    m.prepare([],1,precision=precision,method="tet_quadrature",preparation=preparation,nodes=nodes)
    rows=[]
    for name in ("tet_quadrature","fields","evaluate","affine")+(("device_panels",) if preparation=='device' else ())+ ("device_search",):
        module=wp.get_module("warpSPHBoundaries.fluxbridge."+name)
        options=dict(module.options);options["block_dim"]=256
        source,extension,_,ltoirs,fatbins=module._run_codegen(options,is_cpu=False)
        path=output/f"{name}-{precision}.{extension}";path.write_text(source)
        ptx=output/f"{name}-{precision}.ptx"
        build.build_cuda(str(path),arch,str(ptx),pch_dir=None,use_precompiled_headers=False,
            fast_math=False,fuse_fp=False,ltoirs=ltoirs,fatbins=fatbins)
        instructions=[s.strip() for s in ptx.read_text().splitlines() if ".f64" in s and not s.strip().startswith((".","//"))]
        if precision=="float32" and name not in ('device_panels','device_search') and instructions:raise RuntimeError("unexpected fp64 evaluation instructions")
        rows.append(dict(module=name,fp64_instructions=len(instructions),ptx_bytes=ptx.stat().st_size))
    (output/f"compile-{precision}.json").write_text(json.dumps(dict(precision=precision,arch=arch,
        compile_passed=True,execution_tested=False,preparation=preparation,geometry_precision='float64',requested_nodes=nodes,modules=rows,source_sha256=sources()),indent=2)+"\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cpu");p.add_argument("--output",required=True,type=Path)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--accuracy-only",action="store_true")
    mode.add_argument("--material-only",action="store_true")
    mode.add_argument("--compile-only",choices=["float32","float64"])
    p.add_argument("--quick",action="store_true");p.add_argument("--arch",type=int,default=120)
    p.add_argument('--preparation',choices=['host','device'],default='host')
    p.add_argument('--nodes',type=int,default=16,choices=range(8,65))
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True);wp.init()
    if args.compile_only:compile_only(args.compile_only,args.output,args.arch,args.preparation,args.nodes);return
    device=wp.get_device(args.device)
    with (open("/tmp/fluxbridge-gpu.lock","w") if device.is_cuda else nullcontext(None)) as lock:
        if lock is not None:fcntl.flock(lock,fcntl.LOCK_EX)
        report=dict(schema=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),warp=wp.__version__,device=str(device),
            device_name=device.name,cuda_runtime_tested=device.is_cuda,preparation=args.preparation,geometry_precision='float64',requested_nodes=args.nodes,source_sha256=sources(),accuracy=[] if args.material_only else accuracy(device,args.preparation,args.nodes),
            material=[] if args.accuracy_only else material_runtime(device,args.quick,args.preparation,args.nodes),
            limitations=["frozen prepared queries and geometry; reprepare after changes",
                "explicit host/device fp64 angular geometry; selected-dtype evaluation is audited separately",
                "two-rule convergence and independent fixtures, not a certified global error bound",
                "signed boundary quadrature; no universal positivity or relative-error guarantee",
                "2 million samples per interaction and 1 GiB profile/tile storage per rule; unresolved inputs rejected",
                "shared-machine timings; CUDA-event intervals span multiple launches",
                "no pressure, porosity or solver reconstruction"])
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(args.output/"report.json")


if __name__=="__main__":main()
