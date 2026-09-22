# AI Use

I used an AI assistant for debugging the Windows/PyTorch setup, reviewing benchmark structure, and checking that the required analysis was represented. I ran the experiments on the physical GPU and kept the measured outputs.

## Specific incorrect AI output and correction

An earlier AI suggestion used an OOM-search call to a function named `benchmark_naive_attention`, but that function did not exist in my notebook. I discovered this by checking the actual Part D function definitions. I corrected the search to use the existing `measure_attention`, `naive_attention`, and `fused_attention` functions and retained bounded testing so an unobserved OOM boundary would not be invented.

I verified the final outputs against the saved benchmark CSV/log files and GPU UUID.
