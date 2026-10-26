"""Summarize exported Nsight CSV/SQLite and unprofiled scaling measurements.

Uses instruction counts, not estimated FLOPs from source. Missing metrics remain
null or fail a required field check; unavailable counters never become zeros.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np


def number(row, name, required=False):
    value = row.get(name)
    if value in (None, '', 'n/a', 'N/A'):
        if required:
            raise ValueError(f'missing required metric {name}')
        return None
    return float(value.replace(',', ''))


def csv_rows(path):
    # Raw exports have one units row after the header.
    return list(csv.DictReader(path.open()))[1:]


def flops(row, precision):
    ops = ('fadd', 'fmul', 'ffma') if precision == 32 else ('dadd', 'dmul', 'dfma')
    return sum(number(row, f'smsp__sass_thread_inst_executed_op_{op}_pred_on.sum', True)
               * (2 if op.endswith('fma') else 1) for op in ops)


def kernel(row, peak32, peak64):
    seconds = number(row, 'gpu__time_duration.sum', True) * 1e-9
    clock = number(row, 'gpc__cycles_elapsed.avg.per_second', True)
    f32, f64 = flops(row, 32), flops(row, 64)
    return dict(name=row['Kernel Name'], duration_ms=seconds*1e3,
        fp32_flops=f32, fp64_flops=f64, fp32_tflops_s=f32/seconds/1e12,
        fp64_gflops_s=f64/seconds/1e9,
        fp32_pct_of_sustained_fma_peak=100*f32/seconds/(2*peak32*clock),
        fp64_pct_of_sustained_fma_peak=100*f64/seconds/(2*peak64*clock),
        achieved_occupancy_pct=number(row, 'sm__warps_active.avg.pct_of_peak_sustained_active', True),
        sm_throughput_pct=number(row, 'sm__throughput.avg.pct_of_peak_sustained_elapsed', True),
        memory_throughput_pct=number(row, 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', True),
        dram_throughput_pct=number(row, 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed', True),
        fp64_pipe_active_pct=number(row, 'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed'),
        fma_pipe_active_pct=number(row, 'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed'),
        registers_per_thread=number(row, 'launch__registers_per_thread', True),
        grid_blocks=number(row, 'launch__grid_size', True),
        eligible_warps_per_scheduler_cycle=number(row, 'smsp__warps_eligible.avg.per_cycle_active'),
        stall_samples={k: number(row, k) for k in row if k.startswith('smsp__warpidsamp_warps_issue_stalled_')})


def timeline(path):
    with sqlite3.connect(path) as c:
        api = c.execute('select s.value,r.start,r.end from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where s.value like "%Profiler%"').fetchall()
        start = next(x[2] for x in api if x[0] == 'cuProfilerStart')
        end = next(x[1] for x in api if x[0] == 'cuProfilerStop')
        intervals = []
        for table in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
            if c.execute('select count(*) from sqlite_master where name=?', (table,)).fetchone()[0]:
                intervals += c.execute(f'select start,end from {table}').fetchall()
        merged = []
        for lo, hi in sorted(intervals):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        busy = sum(hi-lo for lo, hi in merged)
        kernels = [dict(name=name, calls=count, duration_ms=duration) for name, count, duration in c.execute(
            'select s.value,count(*),sum(k.end-k.start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id group by s.value order by sum(k.end-k.start) desc')]
        return dict(profile_region_ms=(end-start)/1e6, cuda_busy_union_ms=busy/1e6,
            cuda_busy_pct=100*busy/(end-start), kernels=kernels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    peak = csv_rows(a.evidence/'ncu-peak.csv')[0]
    peak32 = number(peak, 'sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained', True)
    peak64 = number(peak, 'sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained', True)
    result = dict(schema=1, evidence_directory=str(a.evidence),
        sustained_fp32_fma_instructions_per_cycle=peak32,
        sustained_fp64_fma_instructions_per_cycle=peak64,
        definitions=dict(flops='predicated-on FADD + FMUL + 2*FFMA, and corresponding double instructions; excludes division, sqrt, transcendental, integer and address operations',
            occupancy='Nsight achieved active warps / maximum active warps',
            sm_throughput='Nsight SM throughput, busiest contributing pipeline; not fp32 peak utilization',
            memory_throughput='Nsight maximum contributing memory-system throughput; not DRAM bandwidth alone',
            fp32_peak='Nsight sustained FMA instructions/cycle * 2 * measured GPC clock for that kernel',
            timing='Unprofiled scaling timings and instrumented profiler/replay timings are separate',
            scene='16 separated volume elements, one selected interior particle-element interaction per query; W6, fp32 numerical, 2D order16 and 3D explicit order8'),
        profiles={}, memory={}, timelines={}, scaling={}, slopes={}, evidence_sha256={})
    sources = [Path(__file__)]
    for path in sorted(a.evidence.glob('ncu-*.csv')):
        sources.append(path)
        rows = csv_rows(path)
        if path.stem == 'ncu-peak':
            continue
        if 'memory' in path.stem:
            names = ['dram__bytes.sum.per_second', 'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
                'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum', 'derived__local_spilling_requests',
                'derived__local_spilling_requests_pct', 'l1tex__t_sector_hit_rate.pct', 'lts__t_sector_hit_rate.pct',
                'gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed']
            result['memory'][path.stem] = [dict(name=r['Kernel Name'], **{k: number(r, k) for k in names}) for r in rows]
        else:
            result['profiles'][path.stem] = [kernel(r, peak32, peak64) for r in rows]
    for path in sorted(a.evidence.glob('nsys-*.sqlite')):
        sources.append(path)
        result['timelines'][path.stem] = timeline(path)
    for d in (2, 3):
        path = a.evidence/f'scaling-{d}d/report.json'
        sources.append(path)
        r = json.loads(path.read_text())
        result['scaling'][str(d)] = r
        terms = sorted({x['term'] for x in r['rows']})
        slopes = {}
        for term in terms:
            slopes[term] = {}
            for key in ('warm', 'complete'):
                low = 32768 if d == 2 else 512
                rows = [x for x in r['rows'] if x['term'] == term and x['queries'] >= low and key in x]
                if len(rows) >= 2:
                    slope = float(np.polyfit(np.log([x['queries'] for x in rows]), np.log([x[key]['median_ms'] for x in rows]), 1)[0])
                    slopes[term][key] = dict(exponent=slope, count_min=rows[0]['queries'], count_max=rows[-1]['queries'], points=len(rows))
        result['slopes'][str(d)] = slopes
    # Relate the measured instruction total to the actual quadrature expansion.
    # This is workload accounting, not an operation lower bound or accuracy test.
    from warpSPHBoundaries.fluxbridge.contract import baseline
    from warpSPHBoundaries.fluxbridge.tet_quadrature import QuadraturePlan
    plan = QuadraturePlan(3, baseline(3), 'w6')
    row = next(x for x in result['scaling']['3']['rows'] if x['term'] == 'baseline' and x['queries'] == 512)
    n = row['queries']
    rule = row['quadrature']
    stages = result['profiles']['ncu-3d-baseline-512-warm']
    result['work_expansion_3d_baseline'] = dict(queries=n, outputs=len(plan.requests),
        radial_profiles=plan.profiles, coefficient_lengths=[list(map(len, ps)) for ps in plan.coefficients],
        angular_samples_per_pair=rule['samples']/n,
        stored_radial_arguments_per_pair=rule['cached_radii']/n,
        radial_profile_evaluations_per_pair=rule['cached_radii']*len(plan.profiles)/n,
        output_sample_visits_per_pair=rule['samples']*len(plan.requests)/n,
        accepted_storage_bytes_per_pair=row['batch_owned_bytes']/n,
        comparison_samples_per_pair=rule['comparison_sample_count']/n,
        fp32_flops_per_pair_by_stage={x['name']:x['fp32_flops']/n for x in stages},
        fp32_flops_per_pair=sum(x['fp32_flops'] for x in stages)/n,
        scope='Accepted prepared rule; excludes geometric preparation and higher-order comparison. Storage is accepted batch-owned arrays, not measured memory traffic.')
    sources += sorted(a.evidence.glob('host-?d.txt'))
    sources += sorted(a.evidence.glob('ncu-*-command.json'))
    sources += [a.evidence/name for name in ('profile_all.py', 'profile_memory.py', 'timeline_all.py', 'host_profile.py')]
    for path in sources:
        result['evidence_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(a.output)


if __name__ == '__main__':
    main()
