# HW4.5 — RTX 4090 submission

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
