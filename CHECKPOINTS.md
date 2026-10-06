# Homework 6 checkpoints

The saved notebook ran all training stages in Colab and wrote checkpoints under
`/content/hw6_outputs/`. Those checkpoint files are not present in this local folder.
They cannot be recovered from the notebook's printed outputs or plots.

Expected training checkpoints:

- `supervised.pt`: supervised encoder and classification head.
- `rotation.pt`: rotation encoder and pretext head.
- `rotation_linear.pt`: frozen-encoder linear classifier.
- `simclr.pt`: contrastive encoder and projection head.
- `simclr_linear.pt`: frozen-encoder linear classifier.

The original run used the notebook cells in order. Equivalent script command:

```text
python hw6.py
```

On Windows: `py -3 hw6.py`. Use the saved notebook configuration and matching
software/hardware for reproducibility. The recorded runtime was a Tesla T4,
Python 3.13.15, PyTorch 2.11.0+cu130 and torchvision 0.26.0+cu130.

If the Colab session is still available, download these files or copy them to
persistent storage and add accessible links here. No checkpoint link is available yet.
