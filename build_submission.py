"""Build a curated, tagged upload bundle; keep legacy work separate in workspace."""
import datetime
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from dulwich import porcelain
from dulwich.repo import Repo
import corrected_report
from export_report_pdf import export

ROOT=Path(__file__).resolve().parent
DEST=ROOT.parent/'submission'/'HW4_5_RTX4090_Submission'
SOURCES=['corrected_config.json','corrected_train.py','corrected_suite.py','corrected_finish_training.py',
 'single_gpu_ddp.py','run_single_gpu_ddp.py','corrected_serving.py','serving_benchmark.py','kv_context_sweep.py',
 'infer.py','corrected_report.py','validate_corrected.py','verify_packaged_data.py','run_corrected.sh',
 'export_report_pdf.py','build_submission.py','download_twitter.py']

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage-only',action='store_true');args=parser.parse_args()
    corrected_report.main();export(ROOT/'REPORT.md',ROOT/'REPORT.pdf')
    DEST.mkdir(parents=True,exist_ok=True)
    for name in SOURCES+['REPORT.md','REPORT.pdf','METRICS.md','RUN_LOG.txt','AI_USE.md','GPU_HOURS.md','GPU_HOURS.csv','SCOPE.md']:
        shutil.copy2(ROOT/name,DEST/name)
    shutil.copytree(ROOT/'corrected',DEST/'corrected',dirs_exist_ok=True,
      ignore=shutil.ignore_patterns('data_cache.pt','__pycache__','*.pyc'))
    figure_dir=DEST/'figures';figure_dir.mkdir(exist_ok=True)
    for path in (ROOT/'figures').glob('corrected_*.png'):shutil.copy2(path,figure_dir/path.name)
    shutil.copy2(ROOT/'corrected'/'requirements-linux-lock.txt',DEST/'requirements-linux-lock.txt')
    (DEST/'requirements-report.txt').write_text('reportlab==5.1.0\nmarkdown2==2.5.5\nbeautifulsoup4==4.15.0\nmatplotlib==3.11.2\nPillow==13.0.0\ndulwich==1.2.20\n',encoding='utf-8')
    (DEST/'.gitignore').write_text('__pycache__/\n*.pyc\n.venv/\ndata_cache.pt\n*.safetensors\n*.zip\n',encoding='utf-8')
    checkpoint=DEST/'corrected'/'NVIDIA_GeForce_RTX_4090'/'checkpoints'/'hw4_corrected_final'
    model=checkpoint/'model.safetensors'
    if not model.exists():raise FileNotFoundError('Trained classifier must be included in the submission')
    model_card=f'''# Trained BERT sentiment classifier

Base model: bert-base-uncased, revision 86b5e0934494bd15c9632b12f734a8a67f723594. Labels: 0 negative, 1 positive. Fine-tuned on 3,500 training tweets from consecutive source rows 797500–802499; validation/test 750/750; seed 42; three epochs; FP32 AdamW; batch 8; sequence length 128; learning rate 2e-5. Final test accuracy: 0.844 (633/750).

This is the final HW4 reference checkpoint, not the AMP throughput configuration. Validation loss was lowest at epoch 1; the report discloses the cost of continuing through epoch 3. Test metrics in the report use the final checkpoint consistently. Twitter labels are weak supervision and short texts may be ambiguous.

The actual trained FP32 checkpoint and tokenizer are included in the ZIP under `{checkpoint.relative_to(DEST).as_posix()}`. Model SHA-256: `{sha(model)}`. The large weights are excluded from Git history, but their checksum is committed in the manifest and they are included in this upload bundle.

Run `python infer.py "I enjoyed this lab."` after installing the pinned Python dependencies. The model loads from local files; no GPU or Hugging Face download is required for this CPU inference path.
'''
    (DEST/'MODEL_CARD.md').write_text(model_card,encoding='utf-8')
    usage='''# GPU usage / reservation evidence

| Field | Observed record |
| --- | --- |
| System | ADS-R15-836-08; Windows host with Ubuntu 24.04 WSL2 |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB |
| UUID | GPU-3b8e816c-80f8-6ffd-b5ba-72725de33e1f |
| Execution timestamps | RUN_LOG.txt; UTC ISO-8601 timestamps |
| Recorded GPU-hour windows | GPU_HOURS.md and itemized GPU_HOURS.csv |
| GPU identity captures | corrected/NVIDIA_GeForce_RTX_4090/gpu_uuid_capture.txt and corrected/gpu_uuid_capture_end.txt |
| Booking confirmation / reservation ID | Not supplied by the operator |

This record documents observed use. It is not a fabricated reservation confirmation or instructor approval. Start/end UUID files describe the same physical card. The original rubric's formal reservation requirement is disclosed as an evidence limitation in SCOPE.md.
'''
    (DEST/'RESERVATION.md').write_text(usage,encoding='utf-8')
    (ROOT/'RESERVATION.md').write_text(usage,encoding='utf-8')
    checklist='''# Submission contents

| Item | File / evidence |
| --- | --- |
| Main readable report | REPORT.pdf; editable REPORT.md |
| HW4.5 summary | METRICS.md (not an HW5 baseline) |
| Real fine-tuning, memory and serving measurements | corrected/ and figures/ |
| Class behavior and five actual errors | Report; baseline_batch8_predictions.csv and classification JSON |
| Saved classifier and tokenizer | corrected/NVIDIA_GeForce_RTX_4090/checkpoints/hw4_corrected_final/ |
| Exact selected data and split | corrected/selected_rows.csv, split_indices.csv and data_provenance.json |
| Source and pinned environment | Python/shell scripts; requirements-linux-lock.txt |
| Single-GPU alternative and original-scope exceptions | SCOPE.md; report Part C |
| UUID captures | Actual start/end captures under corrected/ |
| Execution and AI disclosure | RUN_LOG.txt; AI_USE.md |
| Usage / booking evidence statement | RESERVATION.md |
| GPU-hour record | GPU_HOURS.md; GPU_HOURS.csv |
| Numerical and data reproduction checks | corrected/VALIDATION.txt; PACKAGED_DATA_VALIDATION.json |
| Artifact integrity and version | MANIFEST.json; .git/refs/tags/hw4-5 |

Two-card results are not fabricated. The one-GPU scope and lack of supplied formal booking/instructor confirmation are disclosed for assessment. No course-site submission was made by these scripts.
'''
    (DEST/'SUBMISSION_CHECKLIST.md').write_text(checklist,encoding='utf-8')
    readme='''# HW4.5 — RTX 4090 submission

Open **REPORT.pdf** first. This is a finished single-GPU study using the available RTX 4090, as requested. Read SCOPE.md for the distinction between this adaptation and the original two-GPU rubric. Measurements are actual; unavailable cross-card quantities are not invented.

The ZIP includes the trained BERT classifier, exact selected data, source, results, figures, logs, GPU-hour/usage records, AI disclosure, an integrity manifest and a local Git repository tagged **hw4-5**. The model weights are included in the ZIP and checksum manifest, but excluded from Git history because of their size.

## Quick inference

Use Python 3.12. After installing the dependencies, run from this directory:

```bash
python infer.py "I enjoyed this lab." "This was a frustrating day."
```

This loads the included trained classifier and runs on CPU. Full source twitter.csv and external model downloads are not required for this inference.

## Reproduce and validate

Training/serving use Linux or WSL2 with CUDA. The measured environment was Ubuntu 24.04, PyTorch 2.13.0/CUDA 13.0, Transformers 5.19.0 and vLLM 0.32.0. Use a GPU driver compatible with that runtime. Provision Python 3.12 and build dependencies, then install the pinned environment:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv build-essential python3-dev
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-linux-lock.txt
python corrected_train.py --prepare
python validate_corrected.py
python infer.py "I loved the GPU lab today!"
```

The data loader verifies the included 5,000-row CSV and rebuilds exactly the measured token tensors and splits without the 1.6M-row CSV. `verify_packaged_data.py` compares that reconstruction with a local canonical cache and detects modified inputs. `corrected/PACKAGED_DATA_VALIDATION.json` records the successful pre-submission check.

The training suite resumes completed records. To measure fresh runs, first copy the published outputs to a separate archive and run in a clean copy containing source, config, selected_rows.csv and data_provenance.json. Keep benchmark GPU usage exclusive; unrelated jobs invalidate timing comparisons.

```bash
bash run_corrected.sh
```

The vLLM generator is Qwen/Qwen2.5-7B-Instruct at the pinned revision in the source. Its weights are downloaded from Hugging Face or reused from HF_HOME. The 14 GiB generator is not duplicated in this upload; its complete measured outputs and server configuration are included. BERT's trained classifier is included.

## Integrity and version

`MANIFEST.json` gives SHA-256 for every submitted file except the manifest itself and Git internals. `git show hw4-5` identifies the submitted source/results snapshot. No remote push or course submission is performed. Large trained weights are untracked in Git but covered by the committed manifest.
'''
    (DEST/'README.md').write_text(readme,encoding='utf-8')
    (DEST/'VALIDATION.txt').write_bytes((ROOT/'corrected'/'VALIDATION.txt').read_bytes())
    files=sorted(p for p in DEST.rglob('*') if p.is_file() and '.git' not in p.relative_to(DEST).parts and p.name!='MANIFEST.json')
    manifest={p.relative_to(DEST).as_posix():sha(p) for p in files}
    (DEST/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');files.append(DEST/'MANIFEST.json')
    if not (DEST/'.git').exists():porcelain.init(str(DEST))
    repo=Repo(str(DEST))
    tracked=[str(p.relative_to(DEST)) for p in files if p.suffix!='.safetensors']
    porcelain.add(repo,paths=tracked)
    commit=porcelain.commit(repo,message=b'Package measured RTX 4090 single-GPU HW4.5 submission and disclosed scope',
      author=b'Codex experiment assistant <codex@local>',committer=b'Codex experiment assistant <codex@local>')
    repo.refs[b'refs/tags/hw4-5']=commit
    if args.stage_only:
        print(json.dumps({'staging_directory':str(DEST),'files':len(files),'tag':commit.decode(),'model_included':True},indent=2))
        return
    archive=ROOT.parent/'HW4_5_RTX4090_Submission.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for path in DEST.rglob('*'):
            if path.is_file():z.write(path,arcname='HW4_5_RTX4090_Submission/'+path.relative_to(DEST).as_posix())
    shutil.copy2(ROOT/'REPORT.pdf',ROOT.parent/'HW4_5_Report.pdf')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(files),
       'tag':commit.decode(),'model_included':True,'scope':'single physical RTX 4090; scope exceptions disclosed'},indent=2))

if __name__=='__main__':main()
