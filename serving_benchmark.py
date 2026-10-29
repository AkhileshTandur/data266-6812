"""HF batch-1 baseline or actual per-request latency against a vLLM HTTP server.

Linux example:
  vllm serve Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --dtype half --max-model-len 4096 --seed 42 --no-enable-prefix-caching
  python serving_benchmark.py --backend vllm --url http://localhost:8000
  python serving_benchmark.py --backend hf

Use the SAME immutable --revision and software environment on both cards.
"""
import argparse
import concurrent.futures
import json
import threading
import time
import urllib.request
from pathlib import Path
import numpy as np

PROMPT='Explain in three sentences how gradient checkpointing changes GPU training memory and computation.'

def http_request(url, payload):
    request=urllib.request.Request(url.rstrip('/')+'/v1/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=['hf','vllm'],required=True)
    p.add_argument('--model',default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--revision',default='a09a35458c702b33eeacc393d103063234e8bc28')
    p.add_argument('--url',default='http://localhost:8000')
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--output-tokens',type=int,default=64)
    p.add_argument('--prompt',default=PROMPT)
    p.add_argument('--context-length',type=int,help='Exact tokenized prompt length for KV context experiments')
    p.add_argument('--output-dir',default='results/serving')
    args=p.parse_args()
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    prompt=tokenizer.apply_chat_template([{'role':'user','content':args.prompt}],tokenize=False,add_generation_prompt=True)
    if args.context_length:
        # For HTTP completions, token-ID input preserves the exact requested length.
        source=tokenizer.encode(prompt,add_special_tokens=False)
        token_ids=(source*((args.context_length+len(source)-1)//len(source)))[:args.context_length]
        prompt=token_ids
    pipe=None
    if args.backend=='hf':
        from transformers import pipeline
        pipe=pipeline('text-generation',model=args.model,revision=args.revision,tokenizer=tokenizer,
                      device=0,dtype=torch.float16)
        if isinstance(prompt,list):
            raise ValueError('Use the vLLM backend for exact token-ID context sweeps')
    payload={'model':args.model,'prompt':prompt,'max_tokens':args.output_tokens,
             'temperature':0,'ignore_eos':True,'seed':42}
    def one(barrier=None):
        if barrier: barrier.wait()
        start=time.perf_counter()
        if pipe:
            torch.cuda.synchronize(); start=time.perf_counter()
            output=pipe(prompt,max_new_tokens=args.output_tokens,min_new_tokens=args.output_tokens,
                        do_sample=False,return_tensors=True,pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            count=len(output[0]['generated_token_ids'])-len(tokenizer.encode(prompt))
            generated=tokenizer.decode(output[0]['generated_token_ids'][-count:],skip_special_tokens=True)
        else:
            response=http_request(args.url,payload)
            count=response['usage']['completion_tokens']
            generated=response['choices'][0]['text']
        return {'latency_ms':1000*(time.perf_counter()-start),'output_tokens':count,'generated_text':generated}
    rows=[]
    folder=Path(args.output_dir);folder.mkdir(parents=True,exist_ok=True)
    gpu=torch.cuda.get_device_name().replace(' ','_')
    result_path=folder/f'{gpu}_{args.backend}_context{args.context_length or "default"}.json'
    for batch in ([1] if pipe else [1,8,32]):
        # Warm all concurrency levels; exclude engine initialization and warmup.
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as executor:
            list(executor.map(lambda _:one(),range(batch)))
            requests=[]; elapsed=0
            for repeat in range(args.repeats):
                barrier=threading.Barrier(batch+1)
                futures=[executor.submit(one,barrier) for _ in range(batch)]
                start=time.perf_counter();barrier.wait()
                completed=[f.result() for f in futures]
                elapsed+=time.perf_counter()-start
                for record in completed:
                    record['repeat']=repeat
                    requests.append(record)
        latencies=[r['latency_ms'] for r in requests]
        row={'backend':args.backend,'model':args.model,'requested_revision':args.revision,
             'prompt_text':args.prompt,
             'formatted_input':prompt if isinstance(prompt,str) else 'Exact context-length token IDs',
             'gpu':torch.cuda.get_device_name(),'dtype':'float16','batch':batch,
             'prompt_tokens':len(prompt) if isinstance(prompt,list) else len(tokenizer.encode(prompt)),
             'requested_output_tokens':args.output_tokens,'requests':requests,
             'total_output_tokens':sum(r['output_tokens'] for r in requests),'elapsed_seconds':elapsed,
             'tokens_per_second':sum(r['output_tokens'] for r in requests)/elapsed,
             'p50_ms':float(np.percentile(latencies,50)),'p95_ms':float(np.percentile(latencies,95)),
             'latency_definition':('Local HF pipeline call plus output decoding/counting; CUDA synchronized; no HTTP' if pipe else 'Client wall time from request dispatch to full non-streaming response; HTTP overhead included'),
             'cache_note':'Disable prefix caching on the server for these repeated prompts; record server flags.'}
        rows.append(row)
        result_path.write_text(json.dumps(rows,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!='requests'}),flush=True)
    result_path.write_text(json.dumps(rows,indent=2))

if __name__=='__main__': main()
