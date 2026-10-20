# Homework 7 metrics

**SID4: 6812 | SEED: 6812 | SLICE: 812 | HP_ID: 2 | CLS_A: 2 | CLS_B: 9**

Final executed notebook run: 2026-10-20 UTC. These values supersede the development run recorded earlier in RUN_LOG.txt.

55,000 training / 5,000 validation / 10,000 test examples. Both models: Adam, learning rate 0.001, batch size 256, latent dimension 32, 20 epochs. Best checkpoint selected by validation objective.

| Model | Selected epoch | Parameters | Test MSE | Test MAE | Test BCE/pixel |
|---|---:|---:|---:|---:|---:|
| FC-AE | 20 | 476,720 | 0.012105 | 0.057258 | 0.279904 |
| VAE (posterior mean) | 20 | 480,848 | 0.015510 | 0.065589 | 0.289150 |

MSE, MAE, and BCE are averaged over pixels and test images. VAE reconstruction metrics use the posterior mean. Labels are used for analysis only.

VAE sampled test objective: 242.946528 nats/image = reconstruction BCE 230.062475 + KL 12.884054. One posterior sample per image, sampling seed 6812. This total is not directly comparable to FC-AE reconstruction-only loss.

| Class | FC-AE test MSE | VAE test MSE |
|---|---:|---:|
| T-shirt/top | 0.011067 | 0.014618 |
| Trouser | 0.006065 | 0.008368 |
| Pullover | 0.011517 | 0.014936 |
| Dress | 0.011140 | 0.014519 |
| Coat | 0.010726 | 0.013874 |
| Sandal | 0.018372 | 0.023258 |
| Shirt | 0.012063 | 0.015062 |
| Sneaker | 0.008907 | 0.011741 |
| Bag | 0.019209 | 0.023114 |
| Ankle boot | 0.011985 | 0.015614 |

The FC-AE reconstructs more accurately in this experiment. VAE regularization allows prior sampling but produces smoother reconstructions. See HW7.md for interpretation and figures.

Exact training command: `py -3.13 execute_hw7.py` (executes HW7.ipynb). Equivalent script: `py -3.13 hw7.py`.

Checkpoints: `hw7_outputs/fc_ae_best.pt`, `hw7_outputs/vae_best.pt`. Each contains model and optimizer states, epoch, seed, configuration, and validation metrics. The exact environment and unrounded results are in `hw7_outputs/metrics.json`.

Validation: `py -3.13 hw7_validate.py` passed after the final notebook run. Both reloaded checkpoints reproduced the recorded full-test results.
