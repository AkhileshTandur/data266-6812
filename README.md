# GPU Assignment II: Training throughput and memory

This project compares BERT training settings and vLLM inference on an RTX 4090. The baseline sentiment classifier reached **84.4% test accuracy (633/750)**. The full results and discussion are in [REPORT.pdf](REPORT.pdf), with a shorter summary in [METRICS.md](METRICS.md).

Only one GPU was available. The two-card comparisons are missing, and the Part C experiment uses one DDP rank. [SCOPE.md](SCOPE.md) explains these limitations and the approval still needed for the alternative.

## Files

- `corrected_train.py`: BERT training and evaluation.
- `corrected/`: selected data, experiment results, predictions and logs.
- `figures/`: training curves and benchmark plots.
- `serving_benchmark.py` and `kv_context_sweep.py`: serving and KV-cache measurements.
- `RUN_LOG.txt`, `GPU_HOURS.md` and `RESERVATION.md`: execution and hardware-use records.
- `AI_USE.md`: AI assistance disclosure.

## Run inference

The trained BERT weights are in `HW4_5_RTX4090_Submission.zip`. They are excluded from Git because of their size. Extract the ZIP before using the saved model; cloning this repository alone does not include the weights.

With Python 3.12 and the dependencies installed, run:

```bash
python infer.py "I enjoyed this lab." "This was a frustrating day."
```

Inference runs on CPU and does not need the full Twitter CSV.

## Reproduce the experiments

The measured environment was Ubuntu 24.04 under WSL2, PyTorch 2.13.0 with CUDA 13.0, Transformers 5.19.0 and vLLM 0.32.0. Training and serving require CUDA and a compatible driver.

```bash
sudo apt-get update
sudo apt-get install -y python3-venv build-essential python3-dev
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-linux-lock.txt
python corrected_train.py --prepare
python validate_corrected.py
```

The data loader checks the included 5,000-row CSV and recreates the token tensors and splits. It does not require the original 1.6-million-row dataset. The recorded reconstruction check is in `corrected/PACKAGED_DATA_VALIDATION.json`.

```bash
bash run_corrected.sh
```

The suite skips completed runs. For fresh measurements, preserve the published results separately and use a clean copy with the source, configuration, selected CSV and provenance file. Keep other GPU jobs stopped while benchmarking.

Serving uses `Qwen/Qwen2.5-7B-Instruct` at the revision pinned in the source. Its weights are downloaded from Hugging Face or reused from `HF_HOME`; they are not included in this repository or the submission ZIP.

## Version and checksums

The `hw4-5` tag identifies the original experiment submission. `MANIFEST.json` records file checksums, including the BERT weights distributed in the ZIP. Documentation revisions after that tag do not change the recorded experiment results.
