# AI_USE.md

## Personal Parameters

- SID4 = 6812
- SEED = 6812
- SLICE = 812
- HP_ID = 2
- CLS_A = 2
- CLS_B = 9

## How I Used AI

I used ChatGPT/Codex to help me set up the notebook structure and write parts of the code for the Word2Vec fine-tuning, the LangChain RAG pipeline, and the PyTorch training optimization experiments. I did not just submit the first version it gave me. I ran the notebook, checked the errors, and fixed the parts that failed.

The parts I mainly checked myself were the Colab errors, the reruns, the output tables, and whether the final notebook matched the assignment requirements.

## One Thing AI Got Wrong

One wrong thing in the first version was the IMDB dataset loading line:

```python
imdb = load_dataset("imdb")
```

When I ran it in Colab, it failed with this error:

```text
HfUriError: Invalid HF URI 'hf://datasets/imdb@e6281661ce1c48d982bc483cf8a173c1bbeb5d31/.huggingface.yaml'.
Repository id must be 'namespace/name', got 'imdb'.
```

There was also another issue when I ran a later cell before the earlier Word2Vec cells had finished:

```text
NameError: name 'original_neighbors' is not defined
```

## How I Found It

I found these problems by running the notebook in Colab. The dataset error happened at the IMDB loading cell, so the notebook could not continue. The `original_neighbors` error happened because the t-SNE visualization depends on variables created in earlier Word2Vec cells.

## What I Changed

I changed the IMDB loading line to:

```python
imdb = load_dataset("stanfordnlp/imdb")
```

That fixed the Hugging Face dataset problem because the dataset name now uses the correct namespace.

I also added a check before the t-SNE code so that if the required neighbor tables are missing, the notebook gives a clear message instead of a confusing `NameError`.

For the RAG part, the Wikipedia/API call gave a JSON error in Colab, so I changed the loader to try Wikipedia first and then use fallback movie passages if the API fails. This keeps the notebook running top to bottom instead of stopping because of a temporary API response problem.

After the fixes, the saved notebook had:

```text
code_cells = 17
executed_code_cells = 17
errors = 0
```
