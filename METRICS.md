# METRICS.md

## Personal Parameters

- SID4 = 6812
- SEED = 6812
- SLICE = 812
- HP_ID = 2
- CLS_A = 2
- CLS_B = 9

## Summary

My main submitted file is `homework2_solution.ipynb`. I ran it in Colab and kept the outputs in the notebook. I also checked the saved notebook afterward. It had 17 executed code cells and 0 saved errors.

## Word2Vec / IMDB

The notebook includes these results:

| Required output | Where it is shown |
|---|---|
| Original top-3 neighbors for `cast`, `score`, `plot`, `screen`, `review` | Original Word2Vec neighbor table |
| Fine-tuned top-3 neighbors for the same words | Fine-tuned neighbor table |
| Before/after comparison | Combined neighbor comparison table |
| Original vs fine-tuned cosine for each word | Cosine shift table |
| Most shifted and least shifted word | Printed below the cosine table |
| 2D visualization | t-SNE plot for `plot` |

## RAG

| Item | Value |
|---|---:|
| Movie documents | 10 |
| Main chunk size | 500 |
| Main overlap | 50 |
| Retrieved chunks per question | 3 |
| Questions | 5 |
| Questions rerun with changed chunking | 2 |
| Alternate chunk size | 800 |
| Alternate overlap | 100 |
| Retrieval Success Rate | 5/5 = 1.00 |

The notebook also shows the retrieved chunks and final answer for each question. For the manual retrieval check, I recorded whether the correct answer passage appeared in the top 3 chunks and the rank of the first relevant chunk.

## RAG Failures

I discussed two failure types:

| Failure type | What happened |
|---|---|
| Chunk ranking sensitivity | Changing chunk size/overlap can change which chunks are retrieved and how the answer is written. |
| Answer phrase split across chunks | If the answer and clue words are separated by chunk boundaries, the retriever may return only partial evidence. |

## Training Optimization Experiments

The notebook includes small controlled experiments for:

| Technique | Metrics reported |
|---|---|
| Tensor creation CPU vs GPU | Time, GPU memory if available, final loss |
| Weight initialization | Time, GPU memory if available, final loss |
| Activation checkpointing | Time, GPU memory if available, final loss |
| Gradient accumulation | Time, GPU memory if available, final loss |
| Mixed precision training | Time, GPU memory if available, final loss |

The exact numeric values are in the executed notebook output table.
