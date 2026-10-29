"""Evidence-only report for the corrected two-class HW4 baseline and HW4.5 runs."""
import datetime
import ast
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'corrected'
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
NAMES={'baseline':'Baseline FP32','amp':'FP16 AMP','checkpoint':'Gradient checkpointing',
       'accumulate':'Accumulation 8 × 4','adam8bit':'8-bit AdamW','amp8bit':'AMP + 8-bit AdamW',
       'combined':'AMP + checkpointing + 8-bit AdamW'}

def table(rows):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    if not len(df):return '*No completed measurements yet.*'
    def cell(value):
        if value is None or (isinstance(value,float) and np.isnan(value)):return 'N/A'
        if isinstance(value,float):return f'{value:.4f}'
        return str(value).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
        ['| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])

def read_records():
    records=[]
    for p in OUT.glob('NVIDIA*/*.json'):
        obj=json.loads(p.read_text())
        if isinstance(obj,dict) and 'status' in obj:
            obj['_folder']=str(p.parent);obj['_name']=p.stem;records.append(obj)
    return pd.DataFrame(records)

def epochs(row):return pd.read_csv(Path(row['_folder'])/(row['_name']+'_epochs.csv'))

def main():
    for name in ['REPORT.md','METRICS.md','GPU_HOURS.md']:
        old=ROOT/('legacy_slice0_'+name)
        if not old.exists() and (ROOT/name).exists():old.write_bytes((ROOT/name).read_bytes())
    provenance=json.loads((OUT/'data_provenance.json').read_text());cfg=provenance['config']
    records=read_records()
    if not len(records):return
    full=records[(records.status=='success')&(records.kind=='full_training')].copy()
    if not len(full):return
    for index,row in full.iterrows():
        hist=epochs(row);full.loc[index,'final_val_accuracy']=hist.iloc[-1].val_accuracy
    full.drop(columns=['_folder','_name']).to_csv(OUT/'all_full_training_runs.csv',index=False)
    records.drop(columns=['_folder','_name']).to_csv(OUT/'all_capacity_and_training_records.csv',index=False)
    cards=list(full.gpu.unique())
    report=['# HW4.5 — Training throughput and memory engineering',
      '**RTX 4090 single-GPU study.** This submission uses the available hardware as requested. Two-card comparisons are marked not measured; Part C is an explicitly labelled one-rank DDP alternative. Hardware scope and the instructor-approval distinction are recorded in `SCOPE.md`.',
      '## Data, hardware and reproducibility',
      f"The HW4 reference uses {provenance['row_end_inclusive']-provenance['row_start_inclusive']+1} consecutive Sentiment140 source records, zero-based rows **{provenance['row_start_inclusive']}–{provenance['row_end_inclusive']} inclusive**, excluding the CSV header. The selected range has 2,500 negative and 2,500 positive tweets. No source tweets were reordered or dropped before selecting this range. Configuration: SLICE={cfg['slice']}, SEED={cfg['seed']}.",
      'Sentiment140 is label-ordered, so source range 0–4999 is single-class and unsuitable for evaluating both sentiments. Exploratory single-class runs are excluded from the results in this report. Every reported Part A/B comparison uses the same two-class HW4 reference data, split, initialization and hyperparameters. The supplied selected-row CSV can reproduce the tensor datasets without downloading the full source; checksum verification rejects changed inputs.',
      f"Model `{cfg['model']}` at revision `{cfg['revision']}`; `BertTokenizer` and `BertForSequenceClassification`; tensor datasets; fixed sequence length {cfg['max_length']}; three epochs; AdamW LR {cfg['learning_rate']}, default weight decay 0.01; baseline physical batch {cfg['baseline_batch']}, evaluation batch {cfg['eval_batch']}. SEED={cfg['seed']} is used for both `train_test_split` calls, initialization and training randomness. Train/validation/test sizes are 3500/750/750; the disjoint original-row indices are in `corrected/split_indices.csv`. Splits are shuffled, not stratified. Source SHA-256: `{provenance['dataset_sha256']}`.",
      'New runs execute in Ubuntu 24.04 on WSL2 with real CUDA access. The first uncapped batch-512 test encountered WDDM overcommit/paging. Subsequent runs enforce a resident allocator budget equal to **actual free device VRAM minus 64 MiB workspace headroom**, with exact free/budget bytes recorded per process. This replaces the earlier arbitrary 90% cap; it remains an explicit resident-memory budget, not a claim about the maximum pageable WDDM allocation. Earlier completed batches fit below this budget. CUDA OOM text and uncapped timeout evidence are retained separately. Each experiment is isolated; GPU training and serving benchmarks run sequentially. Exact versions and UUID captures are saved beside each card’s results.',
      'Training throughput = all training samples across three epochs / sum of synchronized training-epoch times. Peak allocated/reserved memory is measured by PyTorch, in GiB (2^30 bytes), and excludes unrelated device allocations. Wall time includes training, per-epoch validation and final testing, excluding data/model loading and saving; external process elapsed time is also logged. Capacity probes perform three finite optimizer updates and are explicitly separate from full fine-tunes. AMP-skipped optimizer updates are recorded. First-epoch warmup is included consistently. There is one full run per configuration; small speed differences are observations, not statistically established gains.',
      '## HW4 — Learning curves, class behavior and saved-model inference']
    first_baseline=full[(full.technique=='baseline')&(full.batch==8)].iloc[0]
    h=epochs(first_baseline);folder=Path(first_baseline['_folder']);prefix=first_baseline['_name']
    report += [f"HW4 reference on {first_baseline.gpu}: final test accuracy **{100*first_baseline.test_accuracy:.2f}%** ({round(750*first_baseline.test_accuracy)}/750), peak {first_baseline.peak_allocated_gib:.4f} GiB, wall time {first_baseline.wall_seconds:.3f} s, throughput {first_baseline.throughput:.2f} samples/s. These are HW4 measurements used only for this HW4.5 comparison, not an HW5 baseline.",table(h)]
    best=int(h.loc[h.val_loss.idxmin(),'epoch']);after=h[h.epoch>best]
    report += [f"Lowest validation loss occurred at epoch {best}; validation loss first stopped improving at epoch {best+1 if best<len(h) else 'not observed within this run'}. Training continued for {len(after)} additional epochs, costing {after.training_seconds.sum():.3f} additional **training-only** seconds, plus validation overhead. Falling training loss with rising validation loss indicates overfitting. Final-epoch test metrics below are reported consistently; the final checkpoint is saved and can be reloaded with `infer.py --model corrected/{folder.name}/checkpoints/hw4_corrected_final`."]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(h.epoch,h.train_loss,'o-',label='Training');axes[0].plot(h.epoch,h.val_loss,'o-',label='Validation');axes[0].legend()
    axes[0].set(xlabel='Epoch',ylabel='Cross-entropy loss');axes[1].plot(h.epoch,h.val_accuracy,'o-');axes[1].set(xlabel='Epoch',ylabel='Validation accuracy')
    fig.tight_layout();fig.savefig(FIG/'corrected_learning_curves.png',dpi=180);plt.close(fig)
    classification=json.loads((folder/f'{prefix}_classification.json').read_text())
    confusion=pd.read_csv(folder/f'{prefix}_confusion.csv',index_col=0).reset_index(names='True / predicted')
    report += ['![Corrected learning curves](figures/corrected_learning_curves.png)',table(confusion),table([{'Class':c,**{k:classification[c][k] for k in ['precision','recall','f1-score','support']}} for c in ['negative','positive']])]
    worst=min(['negative','positive'],key=lambda c:classification[c]['recall']);worst_id=['negative','positive'].index(worst)
    predictions=pd.read_csv(folder/f'{prefix}_predictions.csv')
    errors=predictions[(predictions.actual==worst_id)&(predictions.actual!=predictions.predicted)].head(5)
    diagnoses={800047:'The visible text expresses boredom; the positive weak label may depend on a removed emoticon or noisy supervision.',
       802327:'The complaint and “wtf” sound negative; the positive label may reflect missing context or noisy weak supervision.',
       802192:'“Miss” conveys absence, while the affectionate “boo” suggests a positive relationship; the model favors the negative cue.',
       801475:'“Couldn’t agree more” is a positive idiom with negation; the model appears to miss the idiomatic meaning.',
       802174:'Mixed sentiment includes “sucky” alongside an affectionate goodnight; the negative cue appears to dominate.'}
    report += [f"Worst class by recall: **{worst}**, recall {classification[worst]['recall']:.4f}. The following are actual misclassified final-test examples from that true class. Diagnoses are plausible readings of the text, not proven explanations of model internals.",table([{'Source row':int(r.original_row),'Exact tweet':r.text,'Diagnosis':diagnoses.get(int(r.original_row),'Lexical sentiment and conversational context may conflict; the short tweet is ambiguous without its conversation.')} for r in errors.itertuples()])]
    if len(errors)<5:report += ['Fewer than five misclassified examples exist for this class; none were fabricated.']
    inference_path=OUT/'custom_inference.txt'
    if inference_path.exists():
        custom=[]
        for line in inference_path.read_text(encoding='utf-8').splitlines():
            if line.startswith("{'text':"):
                custom.append(ast.literal_eval(line))
        report += ['Saved-model reload and custom inference (actual outputs from the corrected final checkpoint):',table(custom)]
    report += ['## Part A — Batch scaling to the memory wall']
    fig,ax=plt.subplots()
    for card in cards:
        b=full[(full.gpu==card)&(full.technique=='baseline')].sort_values('batch')
        report += [f'### {card}',table(b[['batch','throughput','peak_allocated_gib','avg_epoch_seconds','wall_seconds']])]
        maximum=b.loc[b.batch.idxmax()];fast=b.loc[b.throughput.idxmax()]
        fails=records[(records.get('gpu')==card)&(records.technique=='baseline')&(records.status!='success')]
        # Failed rows from external timeouts may lack gpu; keep their status visible later.
        report += [f"Largest completed full-training batch: {int(maximum.batch)}. Highest observed throughput: batch {int(fast.batch)}, {fast.throughput:.2f} samples/s. "+('Throughput did not stop improving before the measured memory boundary.' if fast.batch==maximum.batch else f'Throughput stopped improving beyond the observed peak at batch {int(fast.batch)}.'),table(fails[['batch','status']] if len(fails) else [])]
        ax.plot(b.batch,b.throughput,'o-',label=card)
    ax.set(xlabel='Physical batch size',ylabel='Training samples/s',title='Corrected baseline: batch scaling');ax.set_xscale('log',base=2);ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(FIG/'corrected_batch_scaling.png',dpi=180);plt.close(fig)
    report += ['![Corrected batch scaling](figures/corrected_batch_scaling.png)',
      'Only one physical card was available, so a second-card curve, maximum-batch ratio, VRAM ratio and cross-card saturation comparison are not measured in this single-GPU study. In a two-card comparison the ratios need not agree: model/optimizer and driver allocations are fixed costs, activation memory grows with batch size, kernel workspaces depend on shape, and power-of-two testing quantizes the boundary. The measured RTX 4090 saturation conclusion above is confined to the tested batches.']
    report += ['## Part B — Independent memory techniques and combinations',
      'Full-training rows at the largest completed batch are shown below; the maximum capacity-probe batch is also recorded separately. The techniques are applied independently before testing AMP + 8-bit AdamW and AMP + checkpointing + 8-bit AdamW. The comparison at effective batch 32 then attributes memory/throughput changes at a controlled batch. Accuracy is final-test accuracy; all configurations use three epochs.']
    for card in cards:
        rows=[];cardfull=full[full.gpu==card]
        for tech in NAMES:
            group=cardfull[cardfull.technique==tech].sort_values('batch')
            if not len(group):continue
            row=group.iloc[-1]
            probes=records[(records.get('gpu')==card)&(records.technique==tech)&(records.status=='success')]
            rows.append({'Technique':NAMES[tech],'Max completed full batch':int(row.batch),'Max passed probe batch':int(probes.batch.max()),
               'Effective batch':int(row.effective_batch),'Samples/s':row.throughput,'Peak GiB':row.peak_allocated_gib,'Test accuracy':row.test_accuracy})
        report += [f'### Table HW4.5.1 — {card}',table(rows)]
        combinations=cardfull[cardfull.technique.isin(['amp8bit','combined'])]
        if len(combinations):
            combo=combinations.loc[combinations.throughput.idxmax()]
            report += [f"Best tested combination for throughput: **{NAMES[combo.technique]}**, physical batch {int(combo.batch)}, {combo.throughput:.2f} samples/s, peak {combo.peak_allocated_gib:.3f} GiB, final test accuracy {combo.test_accuracy:.4f}. This selects between the two measured combinations; it does not assert that combining every technique is fastest."]
        b32=cardfull[(cardfull.technique=='baseline')&(cardfull.batch==32)]
        rank=[]
        if len(b32):
            b=b32.iloc[0]
            for tech in ['amp','checkpoint','accumulate','adam8bit']:
                group=cardfull[(cardfull.technique==tech)&(cardfull.batch==(8 if tech=='accumulate' else 32))]
                if not len(group):continue
                row=group.iloc[0];saved=b.peak_allocated_gib-row.peak_allocated_gib;lost=b.throughput-row.throughput
                score=float('inf') if saved>0 and lost<=0 else saved/lost if lost>0 else -float('inf')
                rank.append({'Technique':NAMES[tech],'Physical batch':int(row.batch),'Effective batch':int(row.effective_batch),
                   'Saved GiB':saved,'Throughput change %':100*(row.throughput/b.throughput-1),
                   'GiB / lost samples/s':saved/lost if lost>0 else 'No throughput loss','Test accuracy':row.test_accuracy,'_score':score})
            rank.sort(key=lambda x:x['_score'],reverse=True)
            for row in rank:row.pop('_score')
            report += ['Memory saved per unit of throughput lost, ranked against FP32 physical batch 32:',table(rank)]
            if rank:report += [f"At effective batch 32 I would apply **{rank[0]['Technique']}** first based on this ranking; no-loss improvements rank ahead of techniques that sacrifice throughput."]
            ck=cardfull[(cardfull.technique=='checkpoint')&(cardfull.batch==32)]
            if len(ck):report += [f"Checkpointing at batch 32 changes throughput by {100*(ck.iloc[0].throughput/b.throughput-1):.2f}% and saves {b.peak_allocated_gib-ck.iloc[0].peak_allocated_gib:.3f} GiB. I would skip checkpointing on a 32 GB card when the desired batch already fits; at these small batches the recomputation cost buys unnecessary headroom. This is a workload choice, not an unmeasured 5090 speed claim."]
            a=cardfull[cardfull.technique=='accumulate']
            if len(a):
                a=a.iloc[0]
                report += [f"Accumulation uses physical 8 × 4 = effective 32: {int(a.optimizer_steps)} actual optimizer updates in {a.wall_seconds:.3f} s versus physical batch 32: {int(b.optimizer_steps)} updates in {b.wall_seconds:.3f} s. Test accuracy {a.test_accuracy:.4f} versus {b.test_accuracy:.4f}; peak allocation {a.peak_allocated_gib:.3f} versus {b.peak_allocated_gib:.3f} GiB. The final partial accumulation group is weighted by sample count."]
    fig,ax=plt.subplots(figsize=(10,5))
    for (card,tech),g in full.groupby(['gpu','technique']):
        g=g.sort_values('batch');ax.plot(g.peak_allocated_gib,g.throughput,'o-',markersize=4,label=NAMES[tech]+' / '+card.replace('NVIDIA GeForce ',''))
    ax.set(xlabel='Peak allocated GiB',ylabel='Training samples/s',title='Corrected training tradeoffs');ax.legend(fontsize=7);ax.grid(alpha=.3);fig.tight_layout();fig.savefig(FIG/'corrected_training_tradeoffs.png',dpi=180);plt.close(fig)
    report += ['![Corrected memory and throughput](figures/corrected_training_tradeoffs.png)',
      '## Part C — One-GPU DDP alternative',
      'Topology: **single node, one RTX 4090, one process/rank**, NCCL backend. This follows the requested single-GPU adaptation; it does not assert instructor approval of an alternative to the original designated two-GPU requirement. There is no homogeneous/heterogeneous pair, no remote worker, and no inter-rank PCIe or network path. Ordinary training and actual `torchrun --standalone --nproc_per_node=1` DDP use the same data, initialization, optimizer and three-epoch schedule.']
    ddp_folder=OUT/'single_gpu_ddp'
    shared=False
    if not (ddp_folder/'ddp_batch32.json').exists():
        ddp_folder=OUT/'diagnostics'/'ddp_contended_attempt';shared=True
    comparisons=[];ddp_records=[]
    for batch in [32,256]:
        plain_path=ddp_folder/f'plain_batch{batch}.json';ddp_path=ddp_folder/f'ddp_batch{batch}.json'
        if not plain_path.exists() or not ddp_path.exists():continue
        ordinary=json.loads(plain_path.read_text());distributed=json.loads(ddp_path.read_text());ddp_records.append(distributed)
        ratio=distributed['throughput']/ordinary['throughput']
        diagnostic=distributed['sync_no_sync_diagnostic']
        comparisons.append({'Batch':batch,'Plain samples/s':ordinary['throughput'],'DDP samples/s':distributed['throughput'],
          'One-rank relative efficiency %':100*ratio,'DDP peak GiB':distributed['peak_allocated_gib'],
          'Signed sync−no_sync ms':diagnostic['mean_signed_difference_seconds']*1000,
          'DDP test accuracy':distributed['test_accuracy']})
    if comparisons:
        report += [table(comparisons)]
        if shared:report += ['**Timing condition:** these supplementary DDP runs executed while unrelated GPU jobs were active. They verify the distributed code path but their throughputs are not isolated hardware benchmarks and are not mixed into Parts A/B/D. Raw contention evidence is retained.']
        else:report += ['These supplementary comparisons were rerun after unrelated GPU jobs finished. Earlier contended attempts are retained as diagnostics and excluded from this table.']
        report += ['The percentage above is explicitly **one-rank relative efficiency**, T_DDP / (1 × T_plain), not two-GPU scaling efficiency. A two-card study would instead use T_DDP / (2 × T_single). There is no multi-GPU speedup or multi-GPU crossover batch to establish with one rank. The experiment measures framework/reducer overhead without adding physical compute capacity.']
        diagnostics=[]
        for record in ddp_records:
            d=record['sync_no_sync_diagnostic']
            diagnostics.append({'Batch':record['batch'],'Sync step ms':d['mean_sync_seconds']*1000,
             'No-sync step ms':d['mean_no_sync_seconds']*1000,'Signed difference ms':d['mean_signed_difference_seconds']*1000,
             'Difference SE ms':d['difference_standard_error_seconds']*1000,
             'One-rank all_reduce API ms':record['one_rank_allreduce_api_seconds']*1000})
        report += [table(diagnostics),
          'Communication measurement: the diagnostic alternates synchronized DDP and `no_sync()` backward passes for 30 pairs at fixed trained weights, after warmup. Its signed difference includes reducer/bookkeeping and timing noise; a negative value is retained rather than relabelled as a speedup or forced to zero. The separate all-reduce microbenchmark reports API latency only.',
          'Actual inter-rank payload is **0 bytes/step** because no peer exists; effective inter-rank bandwidth is **N/A**, not buffer size divided by a one-rank no-op latency. There is no interconnect transfer to dominate computation. These measurements cannot answer heterogeneous-worker waiting or two-GPU communication efficiency; those original two-card items are explicitly outside this alternative.']
        if any(r.get('diagnostic_memory_note') for r in ddp_records):
            report += ['For the memory-wall batch, unused optimizer state was freed after timed training and testing, before the forward/backward-only diagnostic. This avoids WDDM paging during that separate diagnostic; the reported full-training throughput, peak memory, wall time and accuracy were recorded before cleanup. The earlier pressure-affected diagnostic is archived and excluded from this comparison.']
    else:report += ['No completed one-rank verification record is available; no timings are invented.']
    report += ['## Part D — vLLM and HF serving on real hardware']
    serving=[]
    for p in (OUT/'serving').glob('*contextdefault.json'):
        rows=json.loads(p.read_text())
        for row in rows:
            serving.append({'Card':row['gpu'],'Configuration':row['backend'],'Batch':row['batch'],'Output tokens/s':row['tokens_per_second'],'p50 ms':row['p50_ms'],'p95 ms':row['p95_ms']})
    report += [table(serving),
      'Serving model: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`, FP16, SEED=42. Prompt: “Explain in three sentences how gradient checkpointing changes GPU training memory and computation.” Both backends apply the same tokenizer chat template (45 input tokens) and generate exactly 64 output tokens per request. There are 20 measured repetitions per concurrency after warmup. Latency is individual client request completion time; aggregate throughput counts actual output tokens. Prefix caching is disabled for vLLM. Server flags, logs and software versions are saved with the serving results.']
    revisited=[]
    for path in (OUT/'serving'/'revisit').glob('*contextdefault.json'):
        for row in json.loads(path.read_text()):
            original=next((r for r in serving if r['Card']==row['gpu'] and r['Configuration']=='vllm' and r['Batch']==row['batch']),None)
            revisited.append({'Card':row['gpu'],'Batch':row['batch'],'Repeat output tokens/s':row['tokens_per_second'],
              'Repeat p50 ms':row['p50_ms'],'Repeat p95 ms':row['p95_ms'],
              'Throughput change %':100*(row['tokens_per_second']/original['Output tokens/s']-1) if original else None})
    report += ['The first serving pass above establishes the corrected HW4 reference. A separate server run repeats it for HW4.5 below; both are newly collected after correcting the workflow, not represented as an unavailable historical vLLM run.',table(revisited)]
    hf=next((r for r in serving if r['Configuration']=='hf' and r['Batch']==1),None)
    vl=sorted([r for r in serving if r['Configuration']=='vllm'],key=lambda r:r['Batch'])
    if hf and len(vl)==3:
        previous=None;cross=None
        for row in vl:
            if previous and row['p50 ms']>hf['p50 ms'] and row['Output tokens/s']>previous['Output tokens/s']:
                cross=row['Batch'];break
            previous=row
        report += [(f"The first tested concurrency where p50 per-request latency exceeds the HF batch-1 baseline while aggregate throughput is still increasing is batch {cross}." if cross else 'No tested concurrency satisfies both a higher p50 request latency than HF batch 1 and a throughput increase over the previous vLLM concurrency.' )+
          ' This compares client completion latency rather than dividing batch latency by batch size. Interactive applications primarily care about individual request latency, especially p95; throughput determines capacity and GPU cost. The table is required to judge both, because higher aggregate throughput alone does not establish a responsive interactive service.']
    else:report += ['The complete vLLM batch 1/8/32 comparison is still being collected; no latency crossover is claimed before all rows exist.']
    kvfolder=OUT/'serving'/'revisit'/'kv' if (OUT/'serving'/'revisit'/'kv'/'kv_context_sweep.json').exists() else OUT/'serving'
    kvpath=kvfolder/'kv_context_sweep.json'
    if kvpath.exists():
        kv=json.loads(kvpath.read_text());report += [table([{'Prompt context':r['context_tokens'],'Batch':r['batch'],
            'Observed occupied KV GiB':None if r['peak_sampled_occupied_kv_bytes'] is None else r['peak_sampled_occupied_kv_bytes']/2**30,
            'Active requests observed':r['peak_observed_running_requests'],'New preemptions':r['preemptions_delta'],
            'All requests succeeded':r['all_requests_succeeded']} for r in kv])]
        measured=[r for r in kv if r['peak_sampled_occupied_kv_bytes'] is not None]
        if measured:
            fig,ax=plt.subplots()
            resident=[r for r in measured if r['peak_observed_running_requests']>=8 and r['preemptions_delta']==0]
            nonresident=[r for r in measured if r not in resident]
            ax.plot([r['context_tokens'] for r in resident],[r['peak_sampled_occupied_kv_bytes']/2**30 for r in resident],'o-',label='Occupied KV: 8 active, no preemptions')
            if nonresident:ax.scatter([r['context_tokens'] for r in nonresident],[r['peak_sampled_occupied_kv_bytes']/2**30 for r in nonresident],marker='x',color='red',label='Scheduler cannot keep 8 active / preempts')
            ax.plot([r['context_tokens'] for r in measured],[r['ideal_prompt_plus_output_kv_bytes']/2**30 for r in measured],'--',label='Ideal 8 × (prompt + 64 output) KV payload')
            ax.axhline(measured[0]['cache_pool_bytes_from_log']/2**30,color='grey',linestyle=':',label='Profiled KV pool capacity')
            ax.set(xlabel='Prompt context tokens / request',ylabel='KV memory (GiB)',title='KV occupancy at concurrency 8');ax.legend(fontsize=8);ax.grid(alpha=.3);fig.tight_layout();fig.savefig(FIG/'corrected_kv_memory.png',dpi=180);plt.close(fig)
            report += ['![KV memory](figures/corrected_kv_memory.png)']
        boundary=kvfolder/'kv_capacity_boundary.json'
        if boundary.exists():
            limit=json.loads(boundary.read_text())
            report += [f"Longest tested preemption-free prompt context at concurrency 8: **{limit['longest_tested_no_preemption_prompt_tokens']} tokens**, with 64 generated tokens. The next tested nonresident/preempting context is {limit['next_tested_failed_resident_prompt_tokens']}; boundary resolution is {limit['resolution_tokens']} tokens. Criterion: all eight requests succeed, at least eight active requests observed, and no new cache preemptions. vLLM manages cache pressure by preempting or serializing requests, so this is a measured resident-cache boundary, not a fabricated CUDA allocation OOM. The engine limit is 16,384 total tokens, below the architecture’s 32,768 limit; the reported memory boundary is distinguished from these limits."]
        report += ['Occupied KV is measured from the server’s live cache-use gauge and actual profiled token-slot capacity, using FP16 payload geometry 2 × 28 layers × 4 KV heads × 128 head dimension × 2 bytes. Cache blocks contain 16 tokens. Samples arrive every 0.1 s and therefore provide a lower bound on instantaneous peak occupancy; reserved cache-pool VRAM is not mistaken for occupied payload. Raw metric samples, pool capacity and per-request results are included.']
    else:report += ['KV occupancy/context sweep is pending. Live occupied KV memory must be distinguished from the reserved cache pool; successful serialized requests do not prove eight simultaneously resident contexts.']
    report += ['The historical Windows HF baseline was 35.2757 output tokens/s, p50 1813.8102 ms and p95 1848.0677 ms on this same 4090/model/FP16 precision. The new Linux HF rate is 46.4331 tokens/s (31.63% higher). The recorded software changed from Windows Torch 2.11.0/CUDA 12.8/Transformers 5.21.0 to WSL Linux Torch 2.13.0/CUDA 13.0/Transformers 5.19.0. This documents a changed execution stack, not a hardware or precision change; individual causes were not isolated. Earlier historical vLLM results were absent, so the corrected first/repeat passes above provide the explicit new HW4/HW4.5 comparison. No RTX 5090 serving measurements exist yet.',
      '## Part E — Configuration and GPU hours']
    fastest=full.loc[full.throughput.idxmax()];lowest=full.loc[full.peak_allocated_gib.idxmin()];shortest=full.loc[full.wall_seconds.idxmin()]
    baseline_val=epochs(first_baseline).iloc[-1].val_accuracy
    eligible=full[full.final_val_accuracy>=baseline_val-.02]
    recommended=eligible.loc[eligible.throughput.idxmax()] if len(eligible) else first_baseline
    report += [f"For a full fine-tune I would use **{NAMES[recommended.technique]}, physical batch {int(recommended.batch)}**, on {recommended.gpu}. Among measured configurations whose final validation accuracy is within two percentage points of the corrected HW4 batch-8 baseline, this has the highest throughput. Validation, not test accuracy, is used for this selection. All tables report three-epoch runs; the observed validation plateau also motivates early stopping during a future full run.",
      f"Measured training-only GPU-hours for this three-epoch subset fine-tune = 3 × 3500 / {recommended.throughput:.4f} / 3600 = {recommended.training_seconds/3600:.6f}. Including measured validation/test, {recommended.wall_seconds:.3f}/3600 = {recommended.wall_seconds/3600:.6f} GPU-hours on one GPU. If extrapolated to 70% of the entire 1.6M-row source, 3 × 1,120,000 / {recommended.throughput:.4f} / 3600 = {3*1120000/recommended.throughput/3600:.4f} GPU-hours, assuming unchanged throughput and padding; that full-source estimate is not a measured result.",
      '## Submission scope and evidence',
      'This is the requested **single-GPU submission**. It includes measured training/serving results, raw records, figures, source, a saved trained classifier, reproducibility checks, GPU-hour records and GPU UUID captures. The captures show the same actual RTX 4090 at different stages; they are not represented as two different cards. Cross-card ratios, two-GPU scaling and inter-rank bandwidth are not measurable here and are disclosed in `SCOPE.md`.',
      'The usage record in `RESERVATION.md` documents observed execution; a booking confirmation and formal instructor approval of the Part C alternative were not supplied and are not invented. Resident-budget boundaries are disclosed because WDDM permits paging. The original rubric’s two-card/approval conditions remain a scope exception for the assessor; the report does not claim to satisfy them with fabricated hardware. No course-site submission has been made.']
    (ROOT/'REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    objectives=[]
    for name,row in [('Highest observed throughput',fastest),('Lowest observed peak memory',lowest),('Shortest measured wall time',shortest),('Recommended with validation constraint',recommended)]:
        objectives.append({'Objective':name,'GPU':row.gpu,'Technique':NAMES[row.technique],'Physical batch':int(row.batch),'Samples/s':row.throughput,'Peak GiB':row.peak_allocated_gib,'Wall s':row.wall_seconds,'Test accuracy':row.test_accuracy})
    metrics=['# HW4.5 summary — not an HW5 baseline',
      '**RTX 4090 single-GPU study: actual two-class training and serving measurements. Part C is an explicitly labelled one-rank alternative; see SCOPE.md.**',table(objectives),
      f"Source rows {cfg['slice']}–{cfg['slice']+4999}; SEED={cfg['seed']}; 3500/750/750 train/validation/test; three epochs. Capacity boundaries use the recorded available-resident-VRAM budget, not the old fixed 90% cap; the initial uncapped WDDM timeout is retained separately. Wall time includes validation/test and excludes loading/saving. The recommendation preserves final validation accuracy within 2 percentage points of the corrected batch-8 baseline. See REPORT.md for all limitations."]
    (ROOT/'METRICS.md').write_text('\n\n'.join(metrics)+'\n',encoding='utf-8')
    old=[]
    for p in (ROOT/'results'/'runs').glob('*.json'):old.extend(json.loads(p.read_text()))
    old_seconds=sum(r.get('process_wall_seconds',0) for r in old)
    new_seconds=records.get('external_process_seconds',pd.Series(dtype=float)).sum()
    hour_rows=[]
    def add_hour(phase,name,seconds):
        hour_rows.append({'phase':phase,'experiment':name,'gpu_count':1,'elapsed_seconds':seconds,'gpu_hours':seconds/3600})
    for p in (ROOT/'results'/'runs').glob('*.json'):
        for r in json.loads(p.read_text()):add_hour('Legacy training process',p.stem,r.get('process_wall_seconds',0))
    active=None
    for line in (ROOT/'RUN_LOG.txt').read_text().splitlines():
        try:
            stamp,message=line.split(' ',1);stamp=datetime.datetime.fromisoformat(stamp.replace('Z','+00:00'))
        except (ValueError,TypeError):continue
        if message.startswith('START ') and 'benchmark.py' in message:active=(stamp,message)
        elif message.startswith('INTERRUPTED ') and active:
            add_hour('Legacy interrupted attempt',message,(stamp-active[0]).total_seconds());active=None
        elif message.startswith('END '):active=None
    for _,r in records.iterrows():add_hour('Corrected training/probe process',r['_name'],r.get('external_process_seconds',0))
    for directory,label in [(OUT/'single_gpu_ddp','One-GPU plain/DDP process'),
       *[(d,'Contended/interrupted DDP diagnostic process') for d in (OUT/'diagnostics').glob('ddp_*')]]:
        for path in directory.glob('*.json'):
            item=json.loads(path.read_text())
            if item.get('world_size')==1 and 'external_process_seconds' in item:
                add_hour(label,path.stem,item['external_process_seconds'])
    prior_hf=ROOT/'results'/'hf_process_record.json'
    if prior_hf.exists():add_hour('Legacy HF pipeline','HF batch 1',json.loads(prior_hf.read_text())['process_wall_seconds'])
    for p in (OUT/'serving').rglob('*process*.json'):
        data=json.loads(p.read_text())
        if p.name.startswith('server_process'):
            add_hour('vLLM server window',str(p.relative_to(ROOT)),data['elapsed_seconds'])
        elif p.name=='hf_baseline_process.json':
            add_hour('Corrected HF pipeline',str(p.relative_to(ROOT)),data['process_seconds'])
    hours=pd.DataFrame(hour_rows);hours.to_csv(ROOT/'GPU_HOURS.csv',index=False)
    text=f'# GPU-hour record\n\nHistorical slice-0 completed experiment processes: {old_seconds:.3f} s = {old_seconds/3600:.6f} single-GPU elapsed hours. Corrected training/probe processes: {new_seconds:.3f} s = {new_seconds/3600:.6f} single-GPU elapsed hours. Corrected successful full fine-tune measurement windows: {full.wall_seconds.sum():.3f} s = {full.wall_seconds.sum()/3600:.6f} GPU-hours.\n\nThe itemized non-overlapping process/window record is `GPU_HOURS.csv`: total {hours.elapsed_seconds.sum():.3f} s = **{hours.gpu_hours.sum():.6f} single-GPU elapsed hours**, including two legacy interrupted intervals, HF baselines, and vLLM startup/serving/shutdown including the failed compiler startup. Client benchmarking and KV sweep durations are inside server windows and are not added twice.\n\nThese are elapsed windows, not an official reservation or GPU kernel-active time. Startup/loading is included in process durations; full training wall windows are a separate subset, not added again. Long gaps between user turns, CPU-only installation and analysis are excluded. Pre-existing HW4 execution without process timestamps is excluded. Official reservation evidence remains user-supplied.\n'
    (ROOT/'GPU_HOURS.md').write_text(text,encoding='utf-8')

if __name__=='__main__':main()
