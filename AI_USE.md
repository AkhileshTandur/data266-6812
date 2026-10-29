# AI use

OpenAI Codex was used extensively for this assignment. Assistance included reviewing the existing code, fixing gradient accumulation, setting up the WSL/CUDA environment, writing and running training and serving experiments, checking results, and preparing figures, documentation and the submission package.

The initial data slice contained only one class. It was replaced with rows 797500–802499, using seed 42, and the baseline and comparisons were rerun. The reported results come from those saved runs under `corrected/`. Earlier slice-0 results are not used as the final training baseline.

Codex also helped implement the one-rank DDP experiment, verify the packaged data reconstruction, and identify timing attempts affected by other GPU jobs. Those attempts are labelled separately from the measurements used in the report. This wording revision also used Codex.

Only one RTX 4090 was available. The missing two-card measurements and the need for instructor approval of the single-GPU alternative are described in `SCOPE.md`. No instructor approval or reservation confirmation is claimed.
