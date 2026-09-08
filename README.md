# Homework 2

This workspace contains a complete, runnable solution notebook for the Homework 2 assignment:

- embedding-based transfer learning with `word2vec-google-news-300` on IMDB reviews
- a LangChain RAG pipeline over 10 Wikipedia movie pages
- controlled training optimization experiments

## Files

- `homework2_solution.ipynb`: main notebook submission
- `requirements.txt`: Python dependencies
- `RUN_LOG.txt`: run and verification log
- `METRICS.md`: metric summary
- `AI_USE.md`: required AI-use appendix

## Run

Recommended: open `homework2_solution.ipynb` in Google Colab or Jupyter, then run all cells.

Local setup:

```bash
pip install -r requirements.txt
jupyter notebook homework2_solution.ipynb
```

Colab install cell:

```python
!pip -q install gensim datasets langchain langchain-community langchain-core langchain-huggingface langchain-text-splitters sentence-transformers faiss-cpu transformers torch scikit-learn matplotlib pandas numpy wikipedia tqdm
```

Notes:

- The Word2Vec model is large and downloads through `gensim.downloader`.
- The notebook uses open local Hugging Face models for the RAG LLM by default, so no API key is required.
- GPU-specific memory measurements are reported only when CUDA is available.
