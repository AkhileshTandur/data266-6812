"""Independently reconcile selected rows, split coverage and saved predictions."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from corrected_report import read_records,epochs
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'corrected'

def main():
    metadata=json.loads((OUT/'data_provenance.json').read_text())
    selected=pd.read_csv(OUT/'selected_rows.csv');split=pd.read_csv(OUT/'split_indices.csv')
    if metadata.get('selected_rows_sha256'):
        assert hashlib.sha256((OUT/'selected_rows.csv').read_bytes()).hexdigest()==metadata['selected_rows_sha256']
    assert len(selected)==5000
    assert selected.original_row.tolist()==list(range(797500,802500))
    assert selected.label.value_counts().to_dict()=={0:2500,1:2500}
    assert split.original_row.is_unique and set(split.original_row)==set(selected.original_row)
    assert split.groupby('split').size().to_dict()=={'test':750,'train':3500,'validation':750}
    test_rows=set(split[split.split=='test'].original_row)
    source=selected.set_index('original_row')
    original_path=ROOT/metadata['config']['dataset']
    if original_path.exists():
        with original_path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==metadata['dataset_sha256']
        original=pd.read_csv(original_path).iloc[797500:802500]
        assert original.text.tolist()==selected.text.tolist()
        assert original.label.tolist()==selected.label.tolist()
    records=read_records();successful=records[(records.status=='success')&(records.kind=='full_training')]
    for _,record in successful.iterrows():
        assert record.config_sha256==metadata['config_sha256']
        history=epochs(record);assert len(history)==3
        assert np.isclose(history.training_seconds.sum(),record.training_seconds)
        assert np.isclose(10500/record.training_seconds,record.throughput)
        folder=Path(record['_folder']);name=record['_name']
        predictions=pd.read_csv(folder/f'{name}_predictions.csv')
        assert len(predictions)==750 and predictions.original_row.is_unique
        assert set(predictions.original_row)==test_rows
        assert all(int(row.actual)==int(source.loc[row.original_row,'label']) for row in predictions.itertuples())
        assert np.isclose((predictions.actual==predictions.predicted).mean(),record.test_accuracy)
        matrix=pd.read_csv(folder/f'{name}_confusion.csv',index_col=0).to_numpy()
        assert matrix.sum()==750 and np.isclose(matrix.trace()/750,record.test_accuracy)
        assert record.attempted_steps==record.optimizer_steps+record.skipped_amp_steps
    serving_count=0
    for path in (OUT/'serving').rglob('*contextdefault.json'):
        for record in json.loads(path.read_text()):
            requests=record['requests'];latency=[r['latency_ms'] for r in requests]
            assert len(requests)==20*record['batch']
            assert all(r['output_tokens']==64 for r in requests)
            assert np.isclose(np.percentile(latency,50),record['p50_ms'])
            assert np.isclose(np.percentile(latency,95),record['p95_ms'])
            assert np.isclose(sum(r['output_tokens'] for r in requests)/record['elapsed_seconds'],record['tokens_per_second'])
            serving_count+=1
    kv_count=0
    for path in (OUT/'serving').rglob('kv_context_sweep.json'):
        for record in json.loads(path.read_text()):
            assert len(record['requests'])==8
            assert 0<=record['peak_sampled_cache_fraction']<=1
            assert np.isclose(record['peak_sampled_occupied_kv_bytes'],record['peak_sampled_cache_fraction']*record['cache_pool_bytes_from_log'])
            for request in record['requests']:
                if request['status']=='success':
                    assert request['usage']['prompt_tokens']==record['context_tokens']
                    assert request['usage']['completion_tokens']==64
            kv_count+=1
    text=f'PASS: consecutive balanced source slice; 3500/750/750 split coverage and exclusivity.\nPASS: {len(successful)} complete fine-tunes reconciled against epochs, original labels, predictions and confusion matrices.\nPASS: {serving_count} serving configurations reconciled against per-request latency and output-token counts.\n'
    ddp_count=0
    for directory in [OUT/'single_gpu_ddp',*list((OUT/'diagnostics').glob('ddp_*'))]:
        for path in directory.glob('*.json'):
            item=json.loads(path.read_text())
            if item.get('world_size')!=1:continue
            assert item['physical_gpu_count']==1 and item['inter_rank_bytes_per_step']==0
            assert item['effective_inter_rank_bandwidth_GBps'] is None
            assert item['config_sha256']==metadata['config_sha256']
            assert len(item['epochs'])==3 and len(item['test_predictions'])==750
            truth=source.loc[split[split.split=='test'].original_row,'label'].tolist()
            assert item['test_truth']==truth
            assert np.isclose(np.mean(np.array(truth)==np.array(item['test_predictions'])),item['test_accuracy'])
            assert np.isclose(10500/sum(r['training_seconds'] for r in item['epochs']),item['throughput'])
            ddp_count+=1
    text+=f'PASS: {ddp_count} one-rank plain/DDP records reconciled; timing conditions are labelled in Part C.\n'
    text+='These checks cover numerical integrity; single-GPU scope and administrative-evidence limitations are documented in SCOPE.md.\n'
    text+=f'PASS: {kv_count} KV context measurements reconciled against prompt lengths, token usage and occupancy fractions.\n'
    (OUT/'VALIDATION.txt').write_text(text,encoding='utf-8');print(text)

if __name__=='__main__':main()
