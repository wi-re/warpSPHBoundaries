"""Serialized whole-element CPU/CUDA BVH build, query and readback costs.

Finite regular sparse scenes: test result sets against host and independent
all-box enumeration, and report synchronization/transfers in wall timings.
The material consumer study is fluxbridge_batch_scaling --search device.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import resource
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import Geometry,DeviceSearch,adjacency
from fluxbridge_shell_acceptance import timed

ROOT=Path(__file__).resolve().parents[2]


def source_hashes():
    files=[Path(__file__),ROOT/'scripts/studies/fluxbridge_shell_acceptance.py',ROOT/'tests/fluxbridge/test_device_search.py']
    files+=list((ROOT/'src/warpSPHBoundaries/fluxbridge').glob('*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def scene(count):
    side=int(np.ceil(count**(1/3)))
    lattice=np.array(np.unravel_index(np.arange(count),(side,side,side))).T*4.
    tetra=np.vstack([np.zeros(3),2*np.eye(3)])
    return Geometry([tetra+x for x in lattice],3),lattice


def pairs(a):return np.c_[a.query,a.element,a.shift]


def run(device,elements,counts,repeats):
    # Exclude first module compilation from repeated measurements, report it.
    g,_=scene(1);cold=DeviceSearch(g,device=device)
    _,cold_cost=timed(lambda:cold.prepare([[.2]*3],1),device,1)
    rows=[]
    for count in elements:
        (g,lattice),geometry_cost=timed(lambda:scene(count),device,1)
        search,build=timed(lambda:DeviceSearch(g,device=device),device,repeats)
        for n in counts:
            rng=np.random.default_rng(933);ids=rng.integers(count,size=n)
            q=lattice[ids]+.2+rng.uniform(-.015,.015,(n,3))
            h=rng.uniform(.7,1.,n)
            ref,host=timed(lambda:adjacency(g,q,h),device,repeats)
            actual,cost=timed(lambda:search.prepare(q,h),device,repeats,events=True)
            np.testing.assert_array_equal(pairs(actual),pairs(ref))
            np.testing.assert_array_equal(actual.offsets,ref.offsets)
            # Independent brute force on first 17 probes, no tree or image code.
            for qi in range(min(17,n)):
                delta=np.maximum(np.maximum(g.bounds[:,0]-q[qi],q[qi]-g.bounds[:,1]),0)/h[qi]
                expected=np.flatnonzero(np.einsum('ij,ij->i',delta,delta)<=1)
                np.testing.assert_array_equal(actual.element[actual.offsets[qi]:actual.offsets[qi+1]],expected)
            row=dict(elements=count,queries=n,interactions=len(actual.query),status='passed',
                geometry_construction=geometry_cost,static_bvh_build_and_upload=build,
                host_search=host,device_search_including_upload_scan_sort_readback=cost,
                static_array_bytes=search.array_bytes,query_metrics=search.last_metrics,
                host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            rows.append(row)
            print(count,n,len(actual.query),'host',round(host['median_ms'],3),'device',round(cost['median_ms'],3),flush=True)
            del ref,actual;gc.collect()
        del search,g;gc.collect()
    return cold_cost,rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0')
    p.add_argument('--elements',default='64,1024,16384');p.add_argument('--counts',default='128,4096,16384')
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();elements=[int(v) for v in args.elements.split(',')];counts=[int(v) for v in args.counts.split(',')]
    if min(elements+counts)<1 or max(elements)>65536 or max(counts)>65536 or not 1<=args.repeats<=20:p.error('invalid resource/repetition bounds')
    args.output.mkdir(parents=True,exist_ok=True);wp.init();device=wp.get_device(args.device)
    with (open('/tmp/fluxbridge-gpu.lock','w') if device.is_cuda else nullcontext(None)) as lock:
        if lock:fcntl.flock(lock,fcntl.LOCK_EX)
        cold,rows=run(device,elements,counts,args.repeats)
        report=dict(schema=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),device=str(device),device_name=device.name,
            cuda_runtime_tested=device.is_cuda,warp=wp.__version__,source_sha256=source_hashes(),first_query_including_compile=cold,
            rows=rows,limitations=['regular separated tetrahedra, roughly one interaction per query; not arbitrary overlap or periodic scaling',
                'fp64 filtering and periodic enumeration, outward-rounded fp32 BVH; padded boundary extras are conservative',
                'wall intervals include uploads, scan/sort, allocation, host synchronization and CSR readback',
                'CUDA-event interval spans multiple launches and host synchronization, not isolated GPU traversal throughput',
                'reported static arrays exclude opaque Warp BVH storage and scan/sort allocator scratch',
                'query array bytes are logical owned storage, not allocator reservation or comparison memory',
                'current consumers reupload returned CSR; complete material costs are measured separately'])
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(args.output/'report.json',flush=True)


if __name__=='__main__':main()
