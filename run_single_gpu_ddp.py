"""Sequential paired single-GPU training; logs every actual torchrun command."""
import json
import subprocess
import sys
import time
from pathlib import Path
from corrected_suite import log,ROOT

out=ROOT/'corrected'/'single_gpu_ddp';out.mkdir(exist_ok=True)
for batch in [32,256]:
    for mode in ['plain','ddp']:
        target=out/f'{mode}_batch{batch}.json'
        if target.exists():continue
        cmd=([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1']
             if mode=='ddp' else [sys.executable])+['single_gpu_ddp.py','--mode',mode,'--batch',str(batch)]
        log('SINGLE_GPU_DDP START '+' '.join(cmd));start=time.perf_counter()
        with target.with_suffix('.log').open('w') as f:
            result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=300)
        if result.returncode:raise RuntimeError(f'Fix {target.with_suffix(".log")}')
        data=json.loads(target.read_text());data['external_process_seconds']=time.perf_counter()-start
        target.write_text(json.dumps(data,indent=2));log(f'SINGLE_GPU_DDP END {mode} batch={batch} process_seconds={data["external_process_seconds"]:.3f}')
