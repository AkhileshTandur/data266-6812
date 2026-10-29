"""Measure live vLLM KV occupancy at concurrency 8, separately from reserved VRAM.

Requires the actual profiled cache pool bytes from this server's startup logs.
Metrics are periodically updated, so sampled occupancy is a lower bound.
Do not call a model context rejection or scheduler preemption a CUDA OOM.
"""
import argparse
import concurrent.futures
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
from serving_benchmark import http_request

def metrics(url):
    with urllib.request.urlopen(url.rstrip('/')+'/metrics',timeout=10) as response:
        raw=response.read().decode()
    result={}
    for line in raw.splitlines():
        if not line.startswith('vllm:'):continue
        match=re.match(r'([^ {]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)',line)
        if match:
            # This tool assumes one engine, one model and one GPU.
            key,value=match.groups();result[key]=max(result.get(key,0),float(value))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='http://localhost:8000')
    p.add_argument('--model',default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--revision',required=True)
    p.add_argument('--cache-capacity-bytes',required=True,type=int,help='Profiled cache pool bytes from actual server log')
    p.add_argument('--contexts',type=int,nargs='+',default=[256,512,1024,2048,4096,8192,16384,30000])
    p.add_argument('--output-tokens',type=int,default=256)
    p.add_argument('--output-dir',default='results/serving')
    args=p.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    config=AutoConfig.from_pretrained(args.model,revision=args.revision)
    layers=config.num_hidden_layers;heads=config.num_key_value_heads
    dim=getattr(config,'head_dim',None) or config.hidden_size//config.num_attention_heads
    kv_bytes_per_token=2*layers*heads*dim*2 # FP16, standard full attention only
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for context in args.contexts:
        seed_tokens=tokenizer.encode('Describe GPU memory engineering carefully. ',add_special_tokens=False)
        tokens=(seed_tokens*((context+len(seed_tokens)-1)//len(seed_tokens)))[:context]
        baseline=metrics(args.url)
        barrier=threading.Barrier(9)
        def request(index):
            # Prefix caching must also be disabled at the server.
            barrier.wait();start=time.perf_counter()
            try:
                response=http_request(args.url,{'model':args.model,'prompt':tokens,'temperature':0,
                    'max_tokens':args.output_tokens,'ignore_eos':True,'seed':42+index})
                return {'status':'success','latency_seconds':time.perf_counter()-start,'usage':response['usage']}
            except urllib.error.HTTPError as e:
                return {'status':'HTTP_error','code':e.code,'body':e.read().decode(errors='replace')}
            except Exception as e:
                return {'status':'error','error':repr(e)}
        samples=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(request,i) for i in range(8)]
            barrier.wait()
            while not all(f.done() for f in futures):
                try:samples.append({'monotonic_seconds':time.perf_counter(),**metrics(args.url)})
                except Exception as e:samples.append({'metrics_error':repr(e)})
                time.sleep(.05)
            requests=[f.result() for f in futures]
        time.sleep(.3)  # Allow the periodic stats callback to publish final preemptions.
        try:after=metrics(args.url)
        except Exception:after={}
        fractions=[s['vllm:kv_cache_usage_perc'] for s in samples if 'vllm:kv_cache_usage_perc' in s]
        peak=max(fractions) if fractions else None
        row={'context_tokens':context,'batch':8,'requests':requests,'metrics_samples':samples,
            'cache_pool_bytes_from_log':args.cache_capacity_bytes,'peak_sampled_cache_fraction':peak,
            'peak_sampled_occupied_kv_bytes':None if peak is None else peak*args.cache_capacity_bytes,
            'ideal_prompt_kv_bytes':kv_bytes_per_token*8*context,
            'ideal_prompt_plus_output_kv_bytes':kv_bytes_per_token*8*(context+args.output_tokens),
            'peak_observed_running_requests':max((s.get('vllm:num_requests_running',0) for s in samples),default=0),
            'preemptions_delta':after.get('vllm:num_preemptions_total',0)-baseline.get('vllm:num_preemptions_total',0),
            'all_requests_succeeded':all(r['status']=='success' for r in requests),
            'note':'Sampled live occupancy is a lower bound; theoretical payload is labeled separately. Successful requests may have been serialized by the scheduler. The model context limit is not a memory wall.'}
        rows.append(row)
        (out/'kv_context_sweep.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k not in ('metrics_samples','requests')}),flush=True)
        if not row['all_requests_succeeded']:break

if __name__=='__main__':main()
