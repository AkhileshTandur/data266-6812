# METRICS.md

## Personal Parameters

- SID4 = 6812
- SEED = 6812
- SLICE = 812
- HP_ID = 2
- CLS_A = 2
- CLS_B = 9

## Prompt Engineering

I ran two prompt examples for each required technique. The prompts were called from code using LangChain, not screenshots.

| Technique | Number of examples |
|---|---:|
| Zero-Shot | 2 |
| Few-Shot | 2 |
| Chain-of-Thought | 2 |
| Zero-Shot CoT | 2 |
| Meta-Prompting | 2 |
| Tree of Thoughts | 2 |

The prompt outputs are saved in `prompt_engineering_outputs.csv`.

One thing I noticed is that zero-shot prompting can sound confident even when the answer is not fully supported. For example, the tulip logic question is a good example to discuss because the model's answer should be checked carefully instead of accepted automatically.

## Self-Attention Results

| Item | Value |
|---|---:|
| Number of word tokens | 46 |
| Vocabulary size | 39 |
| Final unmasked training loss | 0.6318 |
| Final causal masked training loss | 0.9518 |
| Maximum masked attention above diagonal | 0.0 |

The notebook also produces these files:

| Output | File |
|---|---|
| Unmasked attention heatmap | `unmasked_attention_heatmap.png` |
| Causal masked attention heatmap | `masked_attention_heatmap.png` |
| Training loss plot | `attention_training_loss.png` |
| Findings draft | `HW3_findings_draft.pdf` |

## Short Finding

The unmasked model can attend to any token in the sequence. The causal masked model cannot look ahead, and the heatmap shows this because the upper triangle is blocked. The printed check also confirms it, since the largest attention value above the diagonal is `0.0`.
