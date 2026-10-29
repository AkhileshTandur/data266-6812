"""Corrected, isolated HW4/HW4.5 runs. Real metrics with an explicit VRAM policy.

The old slice-0 experiments remain untouched. This establishes a new HW4
baseline on one explicitly chosen consecutive, balanced Sentiment140 slice.
"""
import argparse
import gc
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'corrected_config.json').read_text())
OUT=ROOT/'corrected'
OUT.mkdir(exist_ok=True)

def prepare():
    cache=OUT/'data_cache.pt'
    provenance_path=OUT/'data_provenance.json'
    previous=json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    def digest(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    config_hash=hashlib.sha256(json.dumps(CFG,sort_keys=True).encode()).hexdigest()
    if cache.exists():
        if previous.get('cache_sha256') and digest(cache)!=previous['cache_sha256']:
            raise ValueError('Data cache checksum mismatch; refusing stale/corrupt tensors')
        if previous.get('selected_rows_sha256') and digest(OUT/'selected_rows.csv')!=previous['selected_rows_sha256']:
            raise ValueError('Selected source rows differ from the documented baseline')
        data=torch.load(cache,weights_only=True)
        if data['config_hash']!=config_hash:raise ValueError('Existing data cache belongs to another config; archive it first')
        return data
    source_path=ROOT/CFG['dataset'];start=CFG['slice']
    if source_path.exists():
        df=pd.read_csv(source_path);source_sha=digest(source_path);source_count=len(df)
        if previous.get('dataset_sha256') and source_sha!=previous['dataset_sha256']:
            raise ValueError('Full dataset checksum changed; archive the old baseline before replacing data')
        end=min(start+CFG['max_rows'],len(df))
        if start<0 or start>=len(df):raise ValueError('SLICE outside source data')
        work=df.iloc[start:end].copy();work['original_row']=np.arange(start,end)
        prepared_from='full source CSV'
    else:
        selected=OUT/'selected_rows.csv'
        if not selected.exists() or not previous.get('selected_rows_sha256'):
            raise FileNotFoundError('Provide twitter.csv or the verified packaged selected_rows.csv and data_provenance.json')
        if digest(selected)!=previous['selected_rows_sha256']:
            raise ValueError('Packaged selected-row checksum mismatch')
        if previous.get('config_sha256')!=config_hash:
            raise ValueError('Packaged selection belongs to a different configuration')
        work=pd.read_csv(selected);end=start+len(work)
        if work.original_row.tolist()!=list(range(start,end)) or len(work)>CFG['max_rows']:
            raise ValueError('Packaged rows are not the exact consecutive configured slice')
        source_sha=previous['dataset_sha256'];source_count=previous['dataset_rows']
        prepared_from='verified packaged consecutive slice; full-source checksum retained from original capture'
    if work[['text','label']].isna().any().any():raise ValueError('Missing selected text/labels; refusing to drop source rows')
    if work.label.nunique()!=2:raise ValueError('Corrected baseline requires support for both sentiment classes')
    train,other=train_test_split(np.arange(len(work)),test_size=.3,random_state=CFG['seed'])
    val,test=train_test_split(other,test_size=.5,random_state=CFG['seed'])
    tokenizer=BertTokenizer.from_pretrained(CFG['model'],revision=CFG['revision'])
    encoded=tokenizer(work.text.tolist(),max_length=CFG['max_length'],padding='max_length',truncation=True,return_tensors='pt')
    work.to_csv(OUT/'selected_rows.csv',index=False)
    split=[]
    for name,indices in [('train',train),('validation',val),('test',test)]:
        split.extend({'original_row':int(work.iloc[i].original_row),'split':name} for i in indices)
    pd.DataFrame(split).to_csv(OUT/'split_indices.csv',index=False)
    data={'config_hash':config_hash,'input_ids':encoded['input_ids'],'attention_mask':encoded['attention_mask'],
          'labels':torch.tensor(work.label.to_numpy(),dtype=torch.long),'train':train.tolist(),
          'validation':val.tolist(),'test':test.tolist(),'texts':work.text.tolist(),
          'rows':work.original_row.tolist()}
    torch.save(data,cache)
    metadata={'config':CFG,'config_sha256':config_hash,'row_start_inclusive':start,'row_end_inclusive':end-1,
       'dataset_rows':source_count,'selected_distribution':work.label.value_counts().to_dict(),
       'splits':{name:len(data[name]) for name in ['train','validation','test']},
       'dataset_sha256':source_sha,'selected_rows_sha256':digest(OUT/'selected_rows.csv'),
       'cache_sha256':digest(cache),'prepared_from':prepared_from,
       'correction':'New HW4 baseline; previous slice 0–4999 had only negative labels. Original measurements are preserved and must not be mixed with these training comparisons.'}
    (OUT/'data_provenance.json').write_text(json.dumps(metadata,indent=2))
    return data

def seed():
    random.seed(CFG['seed']);np.random.seed(CFG['seed']);torch.manual_seed(CFG['seed']);torch.cuda.manual_seed_all(CFG['seed'])

def evaluate(model,loader,amp):
    model.eval();loss=0;count=0;truth=[];pred=[]
    with torch.inference_mode():
        for ids,masks,labels in loader:
            ids,masks,labels=[x.cuda() for x in (ids,masks,labels)]
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                output=model(input_ids=ids,attention_mask=masks,labels=labels)
            loss+=output.loss.item()*len(labels);count+=len(labels)
            truth.extend(labels.cpu().tolist());pred.extend(output.logits.argmax(-1).cpu().tolist())
    return loss/count,float(np.mean(np.array(truth)==np.array(pred))),truth,pred

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--tech',choices=['baseline','amp','checkpoint','accumulate','adam8bit','amp8bit','combined'],default='baseline')
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args()
    data=prepare()
    if args.prepare:
        print((OUT/'data_provenance.json').read_text());return
    if not torch.cuda.is_available():raise RuntimeError('Real CUDA GPU required')
    seed();torch.backends.cuda.matmul.allow_tf32=False
    gpu=torch.cuda.get_device_name();gpu_slug=gpu.replace(' ','_')
    folder=OUT/gpu_slug;folder.mkdir(exist_ok=True)
    name=f'{args.tech}_batch{args.batch}'+('_probe' if args.probe else '')
    target=folder/f'{name}.json'
    record={'gpu':gpu,'technique':args.tech,'batch':args.batch,'kind':'capacity_probe' if args.probe else 'full_training',
      'config_sha256':data['config_hash'],'allocator_fraction':'unset (default CUDA allocator)',
      'torch':torch.__version__,'cuda':torch.version.cuda,'seed':CFG['seed']}
    # WSL inherits WDDM overcommit. Bound the resident experiment to actual free
    # device VRAM, rather than an arbitrary 90% fraction or host-RAM paging.
    free_bytes,total_bytes=torch.cuda.mem_get_info()
    budget=max(2**20,free_bytes-64*2**20)
    torch.cuda.set_per_process_memory_fraction(min(1.0,budget/total_bytes))
    record.update(memory_policy='free resident VRAM minus 64 MiB workspace headroom',
        free_vram_before_model_bytes=free_bytes,total_vram_bytes=total_bytes,
        allocator_budget_bytes=budget,allocator_fraction=min(1.0,budget/total_bytes))
    start_process=time.perf_counter()
    amp=args.tech in ['amp','amp8bit','combined'];checkpoint=args.tech in ['checkpoint','combined']
    use8=args.tech in ['adam8bit','amp8bit','combined']
    accumulation=CFG['accumulation_steps'] if args.tech=='accumulate' else 1
    record['effective_batch']=args.batch*accumulation
    try:
        model=BertForSequenceClassification.from_pretrained(CFG['model'],revision=CFG['revision'],num_labels=2,
            id2label={0:'negative',1:'positive'},label2id={'negative':0,'positive':1}).cuda()
        if checkpoint:model.gradient_checkpointing_enable()
        if use8:
            import bitsandbytes as bnb
            optimizer=bnb.optim.AdamW8bit(model.parameters(),lr=CFG['learning_rate'])
        else:optimizer=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'])
        scaler=torch.amp.GradScaler('cuda',enabled=amp)
        def subset(name):
            index=data[name]
            return TensorDataset(data['input_ids'][index],data['attention_mask'][index],data['labels'][index])
        train=subset('train');val=DataLoader(subset('validation'),batch_size=CFG['eval_batch'])
        test=DataLoader(subset('test'),batch_size=CFG['eval_batch'])
        loader=DataLoader(train,batch_size=args.batch,shuffle=True,generator=torch.Generator().manual_seed(CFG['seed']))
        history=[];optimizer_steps=0;attempted_steps=0;skipped_steps=0
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start_wall=time.perf_counter()
        def update():
            nonlocal optimizer_steps,attempted_steps,skipped_steps
            old=scaler.get_scale();scaler.step(optimizer);scaler.update();attempted_steps+=1
            if scaler.get_scale()<old:skipped_steps+=1
            else:optimizer_steps+=1
            optimizer.zero_grad(set_to_none=True)
        if args.probe:
            # Repeat actual training rows only to test tensor shapes > dataset size.
            index=torch.tensor(data['train'])[torch.arange(args.batch)%len(train)]
            ids=data['input_ids'][index].cuda();masks=data['attention_mask'][index].cuda();labels=data['labels'][index].cuda()
            model.train()
            for _ in range(12):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                    output=model(input_ids=ids,attention_mask=masks,labels=labels)
                scaler.scale(output.loss).backward();update()
                if optimizer_steps>=3:break
            if optimizer_steps<3:raise RuntimeError('Probe did not achieve three finite optimizer updates')
            torch.cuda.synchronize()
            record.update(status='success',probe_seconds=time.perf_counter()-start_wall,
               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
               peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        else:
            for epoch in range(CFG['epochs']):
                model.train();optimizer.zero_grad(set_to_none=True);total_loss=0;count=0
                torch.cuda.synchronize();start=time.perf_counter()
                for step,(ids,masks,labels) in enumerate(loader):
                    ids,masks,labels=[x.cuda() for x in (ids,masks,labels)]
                    with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                        output=model(input_ids=ids,attention_mask=masks,labels=labels)
                    total_loss+=output.loss.item()*len(labels);count+=len(labels)
                    group_start=(step//accumulation)*accumulation*args.batch
                    group_samples=min(accumulation*args.batch,len(train)-group_start)
                    scaler.scale(output.loss*len(labels)/group_samples).backward()
                    if (step+1)%accumulation==0 or step+1==len(loader):update()
                torch.cuda.synchronize();duration=time.perf_counter()-start
                vloss,vacc,_,_=evaluate(model,val,amp)
                row={'epoch':epoch+1,'train_loss':total_loss/count,'val_loss':vloss,'val_accuracy':vacc,
                     'training_seconds':duration,'samples_per_second':count/duration}
                history.append(row);print(json.dumps(row),flush=True)
            training_peak=torch.cuda.max_memory_allocated()/2**30
            test_loss,accuracy,truth,pred=evaluate(model,test,amp)
            torch.cuda.synchronize();wall=time.perf_counter()-start_wall
            train_seconds=sum(r['training_seconds'] for r in history)
            record.update(status='success',actual_largest_training_batch=min(args.batch,len(train)),
               throughput=len(train)*CFG['epochs']/train_seconds,training_seconds=train_seconds,wall_seconds=wall,
               avg_epoch_seconds=train_seconds/CFG['epochs'],peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
               training_peak_allocated_gib=training_peak,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
               test_loss=test_loss,test_accuracy=accuracy,optimizer_steps=optimizer_steps,attempted_steps=attempted_steps,
               skipped_amp_steps=skipped_steps,best_val_epoch=min(history,key=lambda r:r['val_loss'])['epoch'])
            pd.DataFrame(history).to_csv(folder/f'{name}_epochs.csv',index=False)
            matrix=confusion_matrix(truth,pred,labels=[0,1])
            pd.DataFrame(matrix,index=['negative','positive'],columns=['negative','positive']).to_csv(folder/f'{name}_confusion.csv')
            report=classification_report(truth,pred,labels=[0,1],target_names=['negative','positive'],output_dict=True,zero_division=0)
            (folder/f'{name}_classification.json').write_text(json.dumps(report,indent=2))
            predictions=[]
            for index,t,prediction in zip(data['test'],truth,pred):
                predictions.append({'original_row':data['rows'][index],'text':data['texts'][index],'actual':t,'predicted':prediction})
            pd.DataFrame(predictions).to_csv(folder/f'{name}_predictions.csv',index=False)
            if args.tech=='baseline' and args.batch==CFG['baseline_batch']:
                checkpoint_dir=folder/'checkpoints'/'hw4_corrected_final'
                model.save_pretrained(checkpoint_dir)
                BertTokenizer.from_pretrained(CFG['model'],revision=CFG['revision']).save_pretrained(checkpoint_dir)
            record['optimizer_state_bytes']=sum(v.numel()*v.element_size() for state in optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor))
    except torch.cuda.OutOfMemoryError as exc:
        record.update(status='OOM',error=str(exc))
    record['process_seconds_excluding_import']=time.perf_counter()-start_process
    target.write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
