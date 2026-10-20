# AI use — Homework 7

I used ChatGPT/Codex for substantial help with this assignment. This appendix describes that assistance and a specific error in the generated code.

## How I used AI

I provided the Homework 7 prompt and the standing requirements, and asked ChatGPT/Codex to help prepare the submission. It drafted the GAN explanations, generated the FC-AE and VAE code, executed the experiments, prepared the notebook, and checked the saved results. It also helped interpret the plots and organize the files. I did not independently write the implementation or the initial report; both involved substantial AI assistance.

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

After I supplied the full standing requirements, validation sampling and generation were changed from offset seeds to SEED = 6812. The final experiment uses that seed and was executed through the notebook. Before submitting, I need to review the code and explain in my own words why the VAE uses reparameterization, why KL is summed over latent dimensions, and why its total loss cannot be directly compared with the FC-AE loss.
