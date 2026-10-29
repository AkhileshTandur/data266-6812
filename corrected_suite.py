"""Resume sequential Linux CUDA experiments; no fabricated memory cap or metrics."""
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'corrected';OUT.mkdir(exist_ok=True)
PYTHON=sys.executable

def log(text):
    line=datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+text
    print(line,flush=True)
    with (ROOT/'RUN_LOG.txt').open('a') as f:f.write(line+'\n')

def run(tech,batch,probe=False):
    name=f'{tech}_batch{batch}'+('_probe' if probe else '')
    path=FOLDER/f'{name}.json'
    if path.exists():return json.loads(path.read_text())
    cmd=[PYTHON,'-u','corrected_train.py','--tech',tech,'--batch',str(batch)]+(['--probe'] if probe else [])
    log('CORRECTED START '+' '.join(cmd));start=time.perf_counter()
    with (FOLDER/f'{name}.log').open('w') as f:
        try:
            result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=300)
            status='error' if result.returncode else 'missing_result'
        except subprocess.TimeoutExpired:status='timeout'
    if not path.exists():path.write_text(json.dumps({'technique':tech,'batch':batch,'kind':'capacity_probe' if probe else 'full_training','status':status}))
    record=json.loads(path.read_text());record['external_process_seconds']=time.perf_counter()-start
    path.write_text(json.dumps(record,indent=2))
    log(f"CORRECTED END {name} status={record['status']} process_seconds={record['external_process_seconds']:.3f}")
    if record['status'] in ['error','missing_result']:raise RuntimeError(f'Fix {name}.log before continuing')
    return record

def main():
    global FOLDER
    import torch
    gpu=torch.cuda.get_device_name().replace(' ','_');FOLDER=OUT/gpu;FOLDER.mkdir(exist_ok=True)
    (FOLDER/'gpu_uuid_capture.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
    subprocess.run([PYTHON,'-m','pip','freeze'],stdout=(FOLDER/'requirements-lock.txt').open('w'),check=True)
    subprocess.run([PYTHON,'corrected_train.py','--prepare'],cwd=ROOT,check=True)
    batches=[8,16,32,64,128,256,512,1024,2048,4096,8192]
    for batch in batches:
        if run('baseline',batch)['status']!='success':break
    for tech in ['amp','checkpoint','adam8bit','amp8bit','combined']:
        if run(tech,8)['status']!='success':raise RuntimeError('Technique failed at HW4 baseline batch')
        maximum=8
        for batch in batches[1:]:
            if run(tech,batch,True)['status']!='success':break
            maximum=batch
        if maximum!=8:run(tech,maximum)
        if maximum>=32:run(tech,32)
    run('accumulate',8)
    log('CORRECTED TRAINING SUITE COMPLETE')

if __name__=='__main__':main()
