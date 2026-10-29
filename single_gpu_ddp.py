"""Actual world-size-one DDP alternative: no simulated second GPU or bandwidth.

torchrun --standalone --nproc_per_node=1 single_gpu_ddp.py --mode ddp --batch 32
python single_gpu_ddp.py --mode plain --batch 32
Uses exactly the corrected HW4 data, initialization, optimizer and epoch count.
"""
import argparse
import contextlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader,TensorDataset
from transformers import BertForSequenceClassification
import corrected_train as training

ROOT=Path(__file__).resolve().parent

def sync_time():
    torch.cuda.synchronize();return time.perf_counter()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['plain','ddp'],required=True)
    parser.add_argument('--batch',type=int,required=True);args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.cuda.set_device(0)
    if args.mode=='ddp':
        if int(os.environ.get('WORLD_SIZE','0'))!=1:raise ValueError('This alternative explicitly uses one process and one GPU')
        dist.init_process_group('nccl')
    try:
        data=training.prepare();cfg=training.CFG;training.seed();torch.backends.cuda.matmul.allow_tf32=False
        free,total=torch.cuda.mem_get_info();budget=max(2**20,free-64*2**20)
        torch.cuda.set_per_process_memory_fraction(min(1,budget/total))
        model=BertForSequenceClassification.from_pretrained(cfg['model'],revision=cfg['revision'],num_labels=2,
          id2label={0:'negative',1:'positive'},label2id={'negative':0,'positive':1}).cuda()
        wrapper=DistributedDataParallel(model,device_ids=[0]) if args.mode=='ddp' else model
        optimizer=torch.optim.AdamW(wrapper.parameters(),lr=cfg['learning_rate'])
        def subset(name):
            index=data[name]
            return TensorDataset(data['input_ids'][index],data['attention_mask'][index],data['labels'][index])
        train=subset('train');loader=DataLoader(train,batch_size=args.batch,shuffle=True,
           generator=torch.Generator().manual_seed(cfg['seed']))
        val=DataLoader(subset('validation'),batch_size=cfg['eval_batch'])
        test=DataLoader(subset('test'),batch_size=cfg['eval_batch'])
        torch.cuda.reset_peak_memory_stats();start=sync_time();history=[];steps=0
        for epoch in range(cfg['epochs']):
            wrapper.train();loss_sum=0;count=0;epoch_start=sync_time()
            for ids,masks,labels in loader:
                optimizer.zero_grad(set_to_none=True)
                ids,masks,labels=[x.cuda() for x in (ids,masks,labels)]
                output=wrapper(input_ids=ids,attention_mask=masks,labels=labels)
                loss_sum+=output.loss.item()*len(labels);count+=len(labels)
                output.loss.backward();optimizer.step();steps+=1
            duration=sync_time()-epoch_start
            vloss,vacc,_,_=training.evaluate(model,val,False)
            row={'epoch':epoch+1,'train_loss':loss_sum/count,'val_loss':vloss,'val_accuracy':vacc,
                 'training_seconds':duration,'samples_per_second':count/duration}
            history.append(row);print(json.dumps(row),flush=True)
        test_loss,accuracy,truth,pred=training.evaluate(model,test,False)
        wall=sync_time()-start;peak=torch.cuda.max_memory_allocated()/2**30
        record={'mode':args.mode,'gpu':torch.cuda.get_device_name(),'world_size':1,'physical_gpu_count':1,
          'topology':'Single-node, single physical GPU, one process/rank','hostname':socket.gethostname(),
          'batch':args.batch,'global_batch':args.batch,'config_sha256':data['config_hash'],'epochs':history,
          'throughput':len(train)*cfg['epochs']/sum(r['training_seconds'] for r in history),
          'wall_seconds':wall,'peak_allocated_gib':peak,'optimizer_steps':steps,
          'test_loss':test_loss,'test_accuracy':accuracy,'test_truth':truth,'test_predictions':pred,
          'allocator_budget_bytes':budget,'inter_rank_bytes_per_step':0,'interconnect_used':'None; no remote rank',
          'effective_inter_rank_bandwidth_GBps':None,'torch':torch.__version__,'cuda':torch.version.cuda}
        if args.mode=='ddp':
            # Diagnostic is separate from timed full training, at fixed trained weights.
            batch=[x.cuda() for x in next(iter(loader))]
            def step(synchronize_gradients):
                wrapper.train();optimizer.zero_grad(set_to_none=True)
                manager=contextlib.nullcontext() if synchronize_gradients else wrapper.no_sync()
                with manager:
                    output=wrapper(input_ids=batch[0],attention_mask=batch[1],labels=batch[2]);output.loss.backward()
            for _ in range(3):step(True);step(False)
            paired=[]
            for repeat in range(30):
                timings={}
                for sync in ([True,False] if repeat%2==0 else [False,True]):
                    start_step=sync_time();step(sync);timings['sync' if sync else 'no_sync']=sync_time()-start_step
                paired.append(timings)
            delta=np.array([r['sync']-r['no_sync'] for r in paired])
            record['sync_no_sync_diagnostic']={'paired_step_seconds':paired,
              'mean_sync_seconds':float(np.mean([r['sync'] for r in paired])),
              'mean_no_sync_seconds':float(np.mean([r['no_sync'] for r in paired])),
              'mean_signed_difference_seconds':float(delta.mean()),
              'difference_standard_error_seconds':float(delta.std(ddof=1)/np.sqrt(len(delta))),
              'interpretation':'Signed timing difference measures one-rank reducer/bookkeeping noise; there is no inter-rank payload or network bandwidth.'}
            payload=torch.zeros(sum(p.numel() for p in model.parameters()),device='cuda',dtype=torch.float32)
            for _ in range(5):dist.all_reduce(payload)
            start_collective=sync_time()
            for _ in range(100):dist.all_reduce(payload)
            record['one_rank_allreduce_api_seconds']=(sync_time()-start_collective)/100
            record['allreduce_buffer_bytes']=payload.numel()*payload.element_size()
            record['allreduce_note']='One-rank all_reduce is an API no-op, not a transfer of this buffer. Do not divide buffer size by API latency to invent bandwidth.'
        out=ROOT/'corrected'/'single_gpu_ddp';out.mkdir(exist_ok=True)
        (out/f'{args.mode}_batch{args.batch}.json').write_text(json.dumps(record,indent=2))
        (out/'gpu_uuid_capture.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
        print(json.dumps({k:v for k,v in record.items() if k not in ['test_predictions','test_truth','epochs','sync_no_sync_diagnostic']},indent=2))
    finally:
        if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':main()
