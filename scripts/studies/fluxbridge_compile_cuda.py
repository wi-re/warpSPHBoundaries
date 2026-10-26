"""Compile paired-fp32 kernels to PTX without claiming GPU execution.

This optional compiler audit pins Warp's internal code-generation API at the
installed version. It is not a runtime dependency or a substitute for CUDA
hardware acceptance. PTX is retained outside the source tree.
"""
import argparse
import json
from pathlib import Path
import warp as wp
import warp._src.build as build
from warpSPHBoundaries.fluxbridge import paired


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arch',type=int,default=80)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    wp.init()
    module=wp.get_module(paired.__name__)
    options=dict(module.options)
    options['block_dim']=256
    source,extension,_,ltoirs,fatbins=module._run_codegen(options,is_cpu=False)
    path=args.output/('paired.'+extension)
    path.write_text(source)
    ptx=args.output/'paired.ptx'
    build.build_cuda(str(path),args.arch,str(ptx),pch_dir=None,use_precompiled_headers=False,
                     fast_math=False,fuse_fp=False,ltoirs=ltoirs,fatbins=fatbins)
    assembly=ptx.read_text()
    fp64_lines=[line.strip() for line in assembly.splitlines() if '.f64' in line]
    fp64_instructions=[line for line in fp64_lines if not line.startswith(('.', '//'))]
    report=dict(warp=wp.__version__,arch=args.arch,compile_passed=True,execution_tested=False,
                cuda_device_available=wp.is_cuda_available(),ptx_bytes=ptx.stat().st_size,
                fp64_instructions=bool(fp64_instructions),fp64_instruction_lines=fp64_instructions,fp64_declarations=len(fp64_lines)-len(fp64_instructions),source=str(path),ptx=str(ptx))
    (args.output/'compile-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['fp64_instructions']:
        raise RuntimeError('Unexpected fp64 operations in paired-fp32 PTX')

if __name__=='__main__':
    main()
