# AI use — Homework 7

This is an assistant-prepared disclosure draft. I need to review it and express my own understanding before submitting it as my personal appendix.

## What the assistant did

ChatGPT/Codex drafted the GAN explanations, wrote the FC-AE and VAE implementation, ran the experiments, prepared the notebook, and checked the saved results. It also helped interpret the reconstruction plots and remove unrelated homework files. The implementation and report were substantially assistant-generated. I supplied the assignment and standing requirements; this draft does not claim that I independently wrote the implementation.

## A specific mistake

The first implementation left the evaluation metrics as NumPy scalars:

```python
rec, kl, mse, mae, bce = sums / count
```

Those values were saved inside the checkpoint's validation dictionary. After the first FC-AE training run, the checkpoint reload failed with this output:

```text
_pickle.UnpicklingError: Weights only load failed.
WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default.
```

This was found by running the actual training pipeline, not just checking the model's tensor shapes. Training reached epoch 20, but execution stopped when `torch.load(..., weights_only=True)` tried to read the best checkpoint.

## The correction and verification

The corrected line is:

```python
rec, kl, mse, mae, bce = map(float, sums / count)
```

Python floats are supported metadata in the restricted checkpoint loader. This preserves the numeric values without disabling `weights_only=True`. The experiment was rerun after the correction. The validation script reloads both checkpoints and recomputes MSE, MAE, BCE, KL, and objective on all 10,000 test images, checking them against the saved results. It also checks a closed-form BCE/KL example, the data split, and gradients into both VAE encoder heads.

After the full standing requirements arrived, validation sampling and generation were also changed from offset seeds to SEED = 6812. The final reported experiment uses that seed and is run through the notebook. I should be able to explain why the VAE uses reparameterization, why its KL term is summed over latent dimensions, and why total VAE loss is not directly comparable to the FC-AE loss.
