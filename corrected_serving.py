"""Sequential HF/vLLM serving and measured occupied-KV context experiments."""
import json
import argparse
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from corrected_suite import log

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'corrected'/'serving';OUT.mkdir(parents=True,exist_ok=True)
MODEL='Qwen/Qwen2.5-7B-Instruct';REV='a09a35458c702b33eeacc393d103063234e8bc28'
PYTHON=sys.executable

def command(name,cmd,timeout=600):
    log('SERVING START '+' '.join(cmd));start=time.perf_counter()
    with (OUT/f'{name}.log').open('w') as f:
        result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    elapsed=time.perf_counter()-start
    (OUT/f'{name}_process.json').write_text(json.dumps({'command':cmd,'returncode':result.returncode,'process_seconds':elapsed},indent=2))
    log(f'SERVING END {name} returncode={result.returncode} process_seconds={elapsed:.3f}')
    if result.returncode:raise RuntimeError(f'Fix {OUT/name}.log')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revisit-only',action='store_true');args=parser.parse_args()
    suffix='_revisit' if args.revisit_only else ''
    kvout=OUT/'revisit'/'kv' if args.revisit_only else OUT
    kvout.mkdir(parents=True,exist_ok=True)
    import torch,vllm,transformers
    gpu_slug=torch.cuda.get_device_name().replace(' ','_')
    result_dir=OUT/'revisit' if args.revisit_only else OUT
    result_path=result_dir/f'{gpu_slug}_vllm_contextdefault.json'
    boundary_path=OUT/'revisit'/'kv'/'kv_capacity_boundary.json'
    if result_path.exists() and boundary_path.exists():
        published=json.loads(result_path.read_text());boundary=json.loads(boundary_path.read_text())
        if {row['batch'] for row in published}=={1,8,32} and boundary.get('next_tested_failed_resident_prompt_tokens') is not None:
            log('SERVING REUSE verified completed results for '+gpu_slug+suffix)
            return
    metadata={'model':MODEL,'revision':REV,'torch':torch.__version__,'vllm':vllm.__version__,
      'transformers':transformers.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
      'precision':'float16','seed':42,'kv_dtype':'auto (float16)',
      'kv_method':'Observed cache occupancy fraction × actual profiled cache token capacity × FP16 KV bytes/token; preemption-free resident capacity tested at concurrency 8.'}
    (OUT/f'environment{suffix}.json').write_text(json.dumps(metadata,indent=2))
    (OUT/'gpu_uuid_capture.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
    hf=OUT/(torch.cuda.get_device_name().replace(' ','_')+'_hf_contextdefault.json')
    if not hf.exists():command('hf_baseline',[PYTHON,'serving_benchmark.py','--backend','hf','--output-dir',str(OUT)])
    cmd=[str(Path(PYTHON).parent/'vllm'),'serve',MODEL,'--revision',REV,'--host','127.0.0.1','--port','8000',
       '--dtype','half','--seed','42','--gpu-memory-utilization','0.9','--max-model-len','16384',
       '--max-num-seqs','32','--max-num-batched-tokens','2048','--enable-chunked-prefill',
       '--no-enable-prefix-caching','--enforce-eager','--attention-backend','FLASH_ATTN','--no-enable-log-requests','--block-size','16']
    (OUT/f'server_command{suffix}.json').write_text(json.dumps(cmd,indent=2))
    env=os.environ.copy();env['VLLM_LOG_STATS_INTERVAL']='0.1';env['VLLM_NO_USAGE_STATS']='1'
    env['PATH']=str(Path(PYTHON).parent)+':'+env.get('PATH','')
    log('VLLM SERVER START '+' '.join(cmd))
    serverlog=OUT/f'vllm_server{suffix}.log';start_server=time.perf_counter()
    with serverlog.open('w') as output:
        server=subprocess.Popen(cmd,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            for _ in range(360):
                if server.poll() is not None:raise RuntimeError('vLLM server exited: inspect vllm_server.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=1) as response:
                        if response.status==200:break
                except Exception:time.sleep(1)
            else:raise TimeoutError('vLLM did not become healthy within six minutes')
            if args.revisit_only:
                command('vllm_revisit',[PYTHON,'serving_benchmark.py','--backend','vllm','--output-dir',str(OUT/'revisit')])
            else:
                vl=OUT/(torch.cuda.get_device_name().replace(' ','_')+'_vllm_contextdefault.json')
                if not vl.exists():command('vllm_throughput',[PYTHON,'serving_benchmark.py','--backend','vllm','--output-dir',str(OUT)])
            text=re.sub(r'\x1b\[[0-9;]*m','',serverlog.read_text())
            match=re.findall(r'GPU KV cache size:\s*([\d,]+)\s*tokens',text)
            if not match:raise RuntimeError('Need actual profiled cache token count from vLLM server log')
            capacity_tokens=int(match[-1].replace(',',''))
            # Actual model geometry: 28 layers, 4 KV heads, head dim 128, FP16.
            bytes_per_token=2*28*4*128*2
            capacity=capacity_tokens*bytes_per_token
            block_match=re.findall(r'block_size[=:]\s*(\d+)',text)
            pool={'profiled_cache_tokens':capacity_tokens,'kv_bytes_per_token':bytes_per_token,
              'profiled_kv_payload_capacity_bytes':capacity,'block_size_from_log':int(block_match[-1]) if block_match else None,
              'note':'Payload capacity is derived from the actual profiled GPU token-slot count and this model geometry, not from total nvidia-smi usage.'}
            (kvout/'kv_pool.json').write_text(json.dumps(pool,indent=2))
            command('kv_contexts',[PYTHON,'kv_context_sweep.py','--revision',REV,'--cache-capacity-bytes',str(capacity),
              '--contexts','256','512','1024','2048','4096','8192','12288','14336','15360','16000','16128','16256','16320',
              '--output-tokens','64','--output-dir',str(kvout)],timeout=600)
            kvpath=kvout/'kv_context_sweep.json'
            measurements=json.loads(kvpath.read_text())
            (kvout/'kv_context_sweep_coarse.json').write_text(json.dumps(measurements,indent=2))
            def fits(row):
                return row['all_requests_succeeded'] and row['preemptions_delta']==0 and row['peak_observed_running_requests']>=8
            passed=sorted(r['context_tokens'] for r in measurements if fits(r))
            failed=sorted(r['context_tokens'] for r in measurements if not fits(r))
            if passed:
                lo=max(passed);upper=[n for n in failed if n>lo];hi=min(upper) if upper else None
                if hi:
                    while hi-lo>16:
                        mid=((lo+hi)//2//16)*16
                        if mid<=lo:mid=lo+16
                        command(f'kv_refine_{mid}',[PYTHON,'kv_context_sweep.py','--revision',REV,
                          '--cache-capacity-bytes',str(capacity),'--contexts',str(mid),
                          '--output-tokens','64','--output-dir',str(kvout)],timeout=300)
                        point=json.loads(kvpath.read_text())[0];measurements.append(point)
                        if fits(point):lo=mid
                        else:hi=mid
                (kvout/'kv_capacity_boundary.json').write_text(json.dumps({'batch':8,'longest_tested_no_preemption_prompt_tokens':lo,
                   'next_tested_failed_resident_prompt_tokens':hi,'resolution_tokens':None if hi is None else hi-lo,
                   'output_tokens':64,'engine_context_limit':16384,'architectural_context_limit':32768,
                   'criterion':'All eight requests succeed, observed active requests >=8, zero new preemptions. Paging/scheduler behavior is reported rather than relabeled CUDA OOM.'},indent=2))
            kvpath.write_text(json.dumps(sorted(measurements,key=lambda r:r['context_tokens']),indent=2))
        finally:
            try:os.killpg(server.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGKILL);server.wait()
            duration=time.perf_counter()-start_server
            (OUT/f'server_process{suffix}.json').write_text(json.dumps({'elapsed_seconds':duration,'returncode':server.returncode},indent=2))
            log(f'VLLM SERVER STOP elapsed_seconds={duration:.3f}')

if __name__=='__main__':main()
