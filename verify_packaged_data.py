"""Rebuild tensors without twitter.csv and compare them to the measured run inputs."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import torch
import corrected_train as training

ROOT=Path(__file__).resolve().parent
canonical=(torch.load(ROOT/'corrected'/'data_cache.pt',weights_only=True)
           if (ROOT/'corrected'/'data_cache.pt').exists() else training.prepare())
with tempfile.TemporaryDirectory(prefix='hw45-packaged-data-') as directory:
    staging=Path(directory);out=staging/'corrected';out.mkdir()
    for name in ['selected_rows.csv','data_provenance.json']:
        shutil.copy2(ROOT/'corrected'/name,out/name)
    training.ROOT=staging;training.OUT=out
    actual=training.prepare()
    for name in ['input_ids','attention_mask','labels']:assert torch.equal(actual[name],canonical[name]),name
    for name in ['train','validation','test','rows','texts','config_hash']:assert actual[name]==canonical[name],name
    with (out/'selected_rows.csv').open('ab') as stream:stream.write(b'\n')
    try:training.prepare()
    except ValueError:rejected=True
    else:raise AssertionError('Changed packaged CSV was not rejected')
    result={'full_source_present':False,'selected_rows':5000,'input_ids_identical':True,'attention_masks_identical':True,
       'labels_identical':True,'split_indices_identical':True,'changed_csv_rejected':rejected,
       'config_sha256':canonical['config_hash'],'torch':torch.__version__}
(ROOT/'corrected'/'PACKAGED_DATA_VALIDATION.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
