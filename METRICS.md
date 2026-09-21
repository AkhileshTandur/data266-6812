# Homework 4 Metrics

These are the model settings I used in the notebook.

| Item | Value |
|---|---:|
| Sequence length | 128 |
| Hidden size | 128 |
| Attention heads | 4 |
| Decoder layers | 2 |
| Epochs | 5 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 3e-4 |

From the run:

| Item | Value |
|---|---:|
| Vocabulary size | 49 |
| Total parameters | 425,777 |
| Final training loss | 2.3315 |
| Final validation loss | 2.5750 |

The notebook also saves:

- `hw4_training_loss.png`
- `hw4_decoding_samples.txt`
- `HW4_findings.pdf`
