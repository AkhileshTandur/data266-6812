# AI use — Homework 6

I used ChatGPT/Codex to help write the notebook, set up the three training methods,
check the contrastive loss and frozen encoders, and draft the supporting documents.
The assistant also helped interpret the saved results and compare the augmentations
with the instructor's demo. I ran the notebook in Colab and provided the executed
notebook for review. The initial implementation and these documents were generated
with substantial AI assistance; I am not claiming that I wrote them independently.

One mistake was the first version of RUN_LOG.txt. It started with:

```text
Homework 6 — Implementation validation

Local environment inspected:
Python 3.13; PyTorch 2.6.0+cu124; torchvision 0.21.0+cu124.
CUDA available; NVIDIA GeForce GTX 1650, 4 GB VRAM.
```

This was a summary of local code checks, not the console log from the actual
training run. The problem became clear when the standing requirements were compared
with the file: it did not contain the required timestamped training output.

The assistant added logging around each notebook stage so that printed output and
errors would be saved with UTC timestamps. That fixes the logging behavior for runs.
However, I did not download the original log from Colab. The current RUN_LOG.txt
was recovered from the notebook's saved output, so it contains the training progress
but not the original timestamps. That limitation is stated in the file.

The assistant checked the contrastive loss against a separate scalar calculation,
checked cosine retrieval with known vectors, and checked that encoder weights and
BatchNorm statistics stayed unchanged during linear training. The saved full run
finished all four parts without a recorded exception. These checks do not replace
my responsibility to understand and explain the notebook.

This appendix is an assisted draft. Before submitting, I need to review it and
describe my own understanding and contributions in my own words.
