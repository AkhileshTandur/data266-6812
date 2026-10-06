# Homework 6 — STL-10 representation learning

SID4: 6812 | SEED: 6812 | SLICE: 812 | HP_ID: 2 | CLS_A: 2 | CLS_B: 9

This assignment compares supervised learning, rotation self-supervision and SimCLR
using ResNet-18 without pretrained weights. All three methods use the same 500
labeled images for classification training. The executed notebook includes the
training output, test results and nearest-neighbor plots.

| Model | Test accuracy |
|---|---:|
| Supervised | 43.60% |
| Rotation SSL | 46.30% |
| SimCLR | 51.79% |

## Files

- `HW6_STL10_Representation_Learning.ipynb`: executed submission notebook.
- `METRICS.md`: results and training-loss summary.
- `HW6_findings.md`: comparison, error analysis and improvement ideas.
- `RUN_LOG.txt`: console output recovered from the notebook; original timestamps unavailable.
- `AI_USE.md`: assistance disclosure and a specific correction; review in your own words.
- `CHECKPOINTS.md`: training command and missing-checkpoint status.
- `figures/`: original plots extracted from the executed notebook.
- `hw6.py`, `requirements.txt`: source and dependencies.
- `Demo_6_Self_Supervised_Learning.ipynb`: instructor reference, not the submission notebook.

## Running

Open the HW6 notebook in a GPU runtime and run its cells in order. The script
equivalent is `python hw6.py` or `py -3 hw6.py` on Windows. The notebook downloads
STL-10 and writes artifacts to `hw6_outputs/`. Save those artifacts before ending
the session. Checkpoints are saved every epoch; automatic resume is not implemented.

The recorded run used a Tesla T4. A smaller GPU may need a smaller SimCLR batch,
which changes the experiment. Python, NumPy and PyTorch seeds are set to 6812,
and deterministic algorithms are enabled. Matching results still depends on using
the same software and hardware conditions.

The demo's four SimCLR augmentation types match this implementation. Crop scale
and the probability of applying color jitter differ; see HW6_findings.md.

Do not run `build_notebook.py` over the executed submission: it regenerates the
notebook and clears its outputs. `validate_hw6.py` is for the unexecuted development
notebook and includes an empty-output assertion; it is not a submission-run command.

## Submission status

The notebook has completed all four parts. The original timestamped Colab log and
model checkpoints have not been recovered. RUN_LOG.txt makes the missing timestamps
explicit; it is not a replacement for the original timestamped log if that can still
be downloaded. See CHECKPOINTS.md for the remaining model-artifact requirement.

Submit through the private `data266-6812` repository, give the instructor and graders
access, and tag the completed submission `hw6`. The configured repository is
https://github.com/AkhileshTandur/data266-6812. Privacy, grader access, final commits
and the submission tag have not been verified here. Preserve the real incremental
development history. HW6 requires no extra six-image gallery or second parameter sweep.
