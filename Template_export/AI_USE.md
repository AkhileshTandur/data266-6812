# AI-Use Appendix - Akhilesh

1. Which parts did you use an assistant for, and which did you write yourself?

I used an assistant while working on the notebook for guidance, checking, and debugging. I worked with the provided dataset, used my SID4 value, reviewed the notebook cells, ran the code, checked the outputs, filled in the final metrics, and made sure I understood the results before submitting.

2. Give one specific thing it produced that was wrong -- a tensor shape error, a deprecated API, a loss function that trained but was wrong, a plausible-looking metric computed incorrectly. Paste the wrong output.

One bad version computed `BCEWithLogitsLoss` with labels shaped like `(batch,)` while the model returned logits shaped like `(batch, 1)`.

```text
ValueError: Target size (torch.Size([32])) must be the same as input size (torch.Size([32, 1]))
```

3. How did you find out? What did the failure look like?

The first training batch failed before the epoch finished. The traceback pointed to the loss line, so I printed the shapes of `model(xb)` and `yb` and saw that one had a second dimension and the other did not.

4. What did you change, and why does your version work?

I reshaped the labels with `.reshape(-1, 1)` when building the tensor datasets and validation tensors. After that change, each row had one logit and one target value, so the binary cross-entropy loss compared matching tensors.
