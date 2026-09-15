# Homework 3

This is my HW3 folder. The main file is `homework3_solution.ipynb`.

In this notebook I did two main things. First, I tested the required prompt engineering styles using LangChain and a local Hugging Face model. I used a local model because I did not want to use paid OpenAI API calls for this assignment. Second, I built single-head self-attention from scratch in PyTorch and compared the unmasked and causal masked attention heatmaps.

The notebook uses my standing parameters:

- SID4 = 6812
- SEED = 6812
- SLICE = 812
- HP_ID = 2
- CLS_A = 2
- CLS_B = 9

Files in this folder:

- `homework3_solution.ipynb`: main notebook
- `METRICS.md`: the main results I recorded
- `RUN_LOG.txt`: notes from running/debugging the notebook
- `AI_USE.md`: required AI-use write-up

I ran the notebook in Google Colab with GPU enabled. The install cell I used was:

```python
!pip -q install langchain transformers accelerate sentencepiece seaborn pandas matplotlib
```
