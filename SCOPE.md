# Hardware scope and limitations

The experiments used one NVIDIA GeForce RTX 4090 (24,564 MiB) on host `ADS-R15-836-08`. Its UUID is `GPU-3b8e816c-80f8-6ffd-b5ba-72725de33e1f`.

An RTX 5090 and a second GPU were not available. Batch scaling, memory techniques, serving and KV-cache measurements were therefore run on the RTX 4090 only.

| Assignment item | What is included |
| --- | --- |
| Batch scaling and memory techniques | Measurements on the RTX 4090. |
| Comparison between two cards | Not measured. |
| Part C | One-process, one-GPU NCCL DDP compared with ordinary single-GPU training. |
| Two-GPU scaling and communication bandwidth | Not measured. One-rank relative efficiency is reported separately; inter-rank bandwidth is N/A. |
| Serving and KV cache | RTX 4090 measurements and a separate repeat. |
| UUID captures | Start and end captures of the same GPU. |
| Reservation | An observed usage record; no booking confirmation was supplied. |

The original rubric requires instructor approval for an alternative when a supported two-GPU pairing is unavailable. That approval has not been supplied. The one-rank experiment does not satisfy the original two-GPU requirement by itself.

WSL can fall back to WDDM shared memory. To avoid treating host paging as usable GPU capacity, the capacity tests used a resident allocator budget equal to measured free VRAM minus 64 MiB of headroom. Reported OOM boundaries apply to that budget. The uncapped paging attempt is saved separately.
