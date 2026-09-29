# Homework 5 — Metrics

## Experiment Setup

- Base model: `google/flan-t5-small`
- Dataset: DialogSum (`neil-code/dialogsum-test`)
- Training examples: 1,000
- Epochs: 1
- Batch size: 8
- Learning rate: 2e-4
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: `q`, `v`
- GPU: NVIDIA L4

## LoRA r=4

- Total parameters: 77,133,184
- Trainable parameters: 172,032
- Trainable percentage: 0.2230%
- Initial batch loss: 2.737375
- Final average training loss: 1.819718
- Last batch loss: 1.704858
- Training steps: 125
- Training examples: 1,000
- Epochs: 1

## LoRA r=16

- Total parameters: 77,649,280
- Trainable parameters: 688,128
- Trainable percentage: 0.8862%
- Initial batch loss: 2.737375
- Final average training loss: 1.822534
- Last batch loss: 1.701030
- Training steps: 125
- Training examples: 1,000
- Epochs: 1

## Rank Comparison

Increasing the LoRA rank from 4 to 16 increased the number of trainable parameters from 172,032 to 688,128, which is 4 times as many trainable LoRA parameters.

The average training losses were very similar:

- r=4: 1.8197
- r=16: 1.8225

The r=4 and r=16 models also generated nearly identical summaries on the inspected examples. Therefore, this experiment did not show a clear quality advantage from increasing the LoRA rank to 16.

## Output Comparison

The baseline focused mainly on one chopstick rule. After LoRA fine-tuning, the generated summaries described the broader conversation about a Chinese feast and table etiquette. The fine-tuned outputs were more aligned with the reference summaries for the inspected examples.

Both LoRA ranks produced very similar results.
