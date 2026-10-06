# Homework 6 results

SID4: 6812 | SEED: 6812 | SLICE: 812 | HP_ID: 2 | CLS_A: 2 | CLS_B: 9

These results are copied from the saved, executed notebook. All models used
ResNet-18 without pretrained weights and the same 500 labeled training images.
Test accuracy was measured on all 8,000 STL-10 test images.

| Model | Test accuracy | Mean precision@5 for the three queries |
|---|---:|---:|
| Supervised | 43.60% | 60.00% |
| Rotation SSL | 46.30% | 33.33% |
| SimCLR | 51.79% | 53.33% |

SimCLR had the highest test accuracy, 8.19 percentage points above supervised
learning and 5.49 points above rotation SSL. Supervised learning had the highest
retrieval score for the three selected queries.

| Training stage | Epochs | First epoch loss | Last epoch loss |
|---|---:|---:|---:|
| Supervised | 15 | 2.2351 | 0.1029 |
| Rotation pretraining | 15 | 0.9463 | 0.2238 |
| Rotation linear classifier | 20 | 3.5381 | 1.2308 |
| SimCLR pretraining | 20 | 4.4767 | 2.2030 |
| SimCLR linear classifier | 20 | 2.7842 | 0.7257 |

The losses belong to different tasks and should not be compared directly.
Rotation used 100,000 unlabeled images. SimCLR used 20,000 and temperature 0.2.
Both linear classifiers used frozen encoders. Training ran on a Tesla T4 with
Python 3.13.15, PyTorch 2.11.0+cu130 and torchvision 0.26.0+cu130.

The retrieval counts below were checked against the saved neighbor plot.

| Query | Supervised correct neighbors | Rotation SSL | SimCLR |
|---|---:|---:|---:|
| Airplane, test index 3 | 5/5 | 2/5 | 2/5 |
| Bird, test index 5 | 0/5 | 0/5 | 1/5 |
| Car, test index 24 | 4/5 | 3/5 | 5/5 |

Precision@5 is the fraction of retrieved neighbors with the same class as the
query, averaged over these three queries. The query itself was excluded.
