#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$task_root"
if [[ -z "${HF_HOME:-}" && -d /mnt/c/Users/018206812/.cache/huggingface ]]; then
  export HF_HOME=/mnt/c/Users/018206812/.cache/huggingface
fi
task_python="${HW45_PYTHON:-python3}"
"$task_python" -u corrected_suite.py | tee -a corrected_suite_linux.log
"$task_python" -u corrected_finish_training.py | tee -a corrected_followup_linux.log
"$task_python" -u run_single_gpu_ddp.py | tee -a single_gpu_ddp_linux.log
"$task_python" -u corrected_serving.py | tee -a corrected_serving_linux.log
"$task_python" -u corrected_serving.py --revisit-only | tee -a corrected_serving_revisit_linux.log
"$task_python" corrected_report.py
"$task_python" validate_corrected.py
