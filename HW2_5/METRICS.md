# HW2.5 Metrics

GPU: NVIDIA GeForce RTX 5090

GPU UUID: `GPU-41051861-72ca-e198-fd60-2bf89fd2a67b`

## Summary

| Measurement                        | Your GPU                          | GPU UUID                                 |
|:-----------------------------------|:----------------------------------|:-----------------------------------------|
| Peak achieved TFLOPS (BF16)        | 232.81164484939026                | GPU-41051861-72ca-e198-fd60-2bf89fd2a67b |
| % of theoretical peak (BF16)       | 111.1272767777519                 | GPU-41051861-72ca-e198-fd60-2bf89fd2a67b |
| Effective bandwidth (GB/s)         | 1530.705965969138                 | GPU-41051861-72ca-e198-fd60-2bf89fd2a67b |
| % of specified memory bandwidth    | 85.41885970809923                 | GPU-41051861-72ca-e198-fd60-2bf89fd2a67b |
| Naive attention boundary           | OOM not observed through L=94208  | GPU-41051861-72ca-e198-fd60-2bf89fd2a67b |
| Fused attention boundary           | OOM not observed through L=262144 | GPU-41051861-72ca-e198-fd60-2bf89fd2a67b |
| Steady-state / peak throughput (%) | 95.60813005682894                 | GPU-41051861-72ca-e198-fd60-2bf89fd2a67b |
| Throttle onset                     | none observed in sampled trace    | GPU-41051861-72ca-e198-fd60-2bf89fd2a67b |

## Part B notes
- Lower precision attempted: FP8 (torch.float8_e4m3fn).
- Result: attempted; direct PyTorch matmul was not supported on this software/kernel path: "addmm_cuda" not implemented for 'Float8_e4m3fn'
- Dense BF16 reference peak used: 209.5 TFLOP/s.
- Percentages above 100% are reported as measured/reference, not as hardware efficiency.

## Part C roofline
- Ridge point: 116.91 FLOP/byte.
- FP32 add: 0.0833 FLOP/byte -> memory-bandwidth-bound side.
- FP16 matmul: 2730.67 FLOP/byte -> compute-bound side.
- Effective bandwidth: 1530.71 GB/s (85.42% of 1792 GB/s).

## Part D attention
- Naive attention: OOM not observed through L=94208.
- Fused attention: OOM not observed through L=262144.
- Naive peak memory follows the measured quadratic fit reported in the notebook.

## Part E sustained load
- Peak first 30 s: 212.95 TFLOP/s.
- Mean final 5 min: 203.60 TFLOP/s.
- Steady/peak: 95.61%.
- Maximum sampled temperature: 73 C.
- Maximum sampled power: 577.38 W; median reported power limit: 575.0 W.
- Throttle interpretation: none observed in sampled trace.
