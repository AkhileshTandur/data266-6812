# Hardware scope and interpretation

This submission uses the available NVIDIA GeForce RTX 4090 (24,564 MiB), UUID `GPU-3b8e816c-80f8-6ffd-b5ba-72725de33e1f`, on host `ADS-R15-836-08`.

The student requested that the solution use the available GPU and proceed without an RTX 5090. Accordingly, this is a **single-GPU adaptation** of the assignment. No second-card results, UUID, VRAM ratio, cross-card throughput curve or multi-GPU bandwidth are invented.

| Original item | Single-GPU treatment |
| --- | --- |
| Per-card batch scaling and memory techniques | Measured on the RTX 4090. |
| Comparison between two cards | Not measured; only one physical card was available. |
| Part C, headed “use any 1 GPU” | Actual one-process/one-GPU NCCL DDP experiment, compared with ordinary one-GPU training. |
| Two-GPU scaling efficiency and interconnect bandwidth | Not measurable with one rank. The report gives only explicitly labelled one-rank relative efficiency; inter-rank bandwidth is N/A. |
| Serving and KV cache | Measured on the available RTX 4090, including a separate repeat. |
| GPU UUID captures | Start/end captures of the same actual GPU; not represented as two different devices. |
| Reservation confirmation | Observed execution record supplied; no booking confirmation was provided. |

The assignment text also says an unsupported two-GPU pairing requires an instructor-approved alternative. A student's hardware-scope request is not represented as instructor approval. The single-GPU adaptation and this limitation are disclosed for assessment.

WSL inherits WDDM shared-memory fallback. Capacity experiments therefore record a resident allocator budget based on measured free VRAM minus 64 MiB workspace headroom. The uncapped paging attempt is preserved separately. OOM labels mean actual allocator OOM under the stated resident budget, not an unrestricted pageable-memory maximum.
