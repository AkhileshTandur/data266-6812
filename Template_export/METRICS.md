# Metrics

Personal parameters: `SID4=6812`, `SEED=6812`, `SLICE=812`, `HP_ID=2`, `CLS_A=2`, `CLS_B=9`.

For HW1, `HP_ID=2` means the modified neural network uses hidden layers `[64, 32]`, learning rate `0.003`, and `30` epochs. The baseline uses hidden layers `[64, 32]`, learning rate `0.001`, and `30` epochs.

## Diabetes Neural Networks

Fixed split: 70% training, 15% validation, 15% testing using `random_state=SEED`.

| Framework | Model | Hidden layers | Learning rate | Epochs | Training seeds | Test accuracy mean | Test accuracy std |
|---|---|---:|---:|---:|---|---:|---:|
| PyTorch | Baseline | [64, 32] | 0.001 | 30 | 6812, 6813, 6814 | 0.7339 | 0.0268 |
| PyTorch | Modified HP2 | [64, 32] | 0.003 | 30 | 6812, 6813, 6814 | 0.7602 | 0.0101 |
| TensorFlow | Baseline | [64, 32] | 0.001 | 30 | 6812, 6813, 6814 | 0.7251 | 0.0221 |
| TensorFlow | Modified HP2 | [64, 32] | 0.003 | 30 | 6812, 6813, 6814 | 0.7573 | 0.0051 |

TensorFlow values came from the executed Colab notebook. The required notebook file is `neural_networks.ipynb`.

## CUDA Matrix Multiplication

CUDA timing table from the A100 Colab run of `cuda.ipynb`. The program output also reported max CPU/GPU result error of `0.000015`, `0.000092`, and `0.001221` for sizes 256, 1024, and 4096.

Profiler used: Nsight Compute (`ncu`). Nsight Systems (`nsys`) was attempted first but was not installed in the Colab environment. The profiler reported `matmul_kernel` durations of about `28.80 us`, `895.10 us`, and `67.76 ms` for sizes 256, 1024, and 4096. The table below uses the normal non-profiler run because `ncu` adds replay overhead to the printed program timings.

| Matrix size | CPU (ms) | GPU kernel (ms) | H2D+D2H (ms) | Speedup |
|---:|---:|---:|---:|---:|
| 256 | 21.643 | 0.121 | 0.279 | 54.020 |
| 1024 | 3305.822 | 0.903 | 3.013 | 844.163 |
| 4096 | 657903.697 | 67.836 | 42.927 | 5939.709 |

In these measurements, GPU end-to-end time was already beneficial at matrix size 256. The crossover is not at size zero because CUDA has fixed overhead from launching a kernel and copying data between host and device. For very small matrices, that overhead can dominate the arithmetic work. As the matrices get larger, the available parallel work grows much faster than the transfer overhead, so the GPU speedup becomes much larger.
