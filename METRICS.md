# HW4.5 summary — not an HW5 baseline

**RTX 4090 single-GPU study: actual two-class training and serving measurements. Part C is an explicitly labelled one-rank alternative; see SCOPE.md.**

| Objective | GPU | Technique | Physical batch | Samples/s | Peak GiB | Wall s | Test accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Highest observed throughput | NVIDIA GeForce RTX 4090 | FP16 AMP | 256 | 1039.7222 | 13.9053 | 12.4138 | 0.8333 |
| Lowest observed peak memory | NVIDIA GeForce RTX 4090 | AMP + checkpointing + 8-bit AdamW | 8 | 137.7015 | 1.0552 | 78.8487 | 0.8373 |
| Shortest measured wall time | NVIDIA GeForce RTX 4090 | FP16 AMP | 256 | 1039.7222 | 13.9053 | 12.4138 | 0.8333 |
| Recommended with validation constraint | NVIDIA GeForce RTX 4090 | FP16 AMP | 256 | 1039.7222 | 13.9053 | 12.4138 | 0.8333 |

Source rows 797500–802499; SEED=42; 3500/750/750 train/validation/test; three epochs. Capacity boundaries use the recorded available-resident-VRAM budget, not the old fixed 90% cap; the initial uncapped WDDM timeout is retained separately. Wall time includes validation/test and excludes loading/saving. The recommendation preserves final validation accuracy within 2 percentage points of the corrected batch-8 baseline. See REPORT.md for all limitations.
