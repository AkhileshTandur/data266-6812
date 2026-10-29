# GPU-hour record

Historical slice-0 completed experiment processes: 2110.691 s = 0.586303 single-GPU elapsed hours. Corrected training/probe processes: 1464.686 s = 0.406857 single-GPU elapsed hours. Corrected successful full fine-tune measurement windows: 709.605 s = 0.197112 GPU-hours.

The itemized non-overlapping process/window record is `GPU_HOURS.csv`: total 5282.027 s = **1.467230 single-GPU elapsed hours**, including two legacy interrupted intervals, HF baselines, and vLLM startup/serving/shutdown including the failed compiler startup. Client benchmarking and KV sweep durations are inside server windows and are not added twice.

These are elapsed windows, not an official reservation or GPU kernel-active time. Startup/loading is included in process durations; full training wall windows are a separate subset, not added again. Long gaps between user turns, CPU-only installation and analysis are excluded. Pre-existing HW4 execution without process timestamps is excluded. Official reservation evidence remains user-supplied.
