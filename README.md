# Homework 4

Main notebook: `homework4_solution.ipynb`

This notebook is for the mini GPT assignment. I built a small decoder-only model in PyTorch and trained it on character-level text. The notebook includes the data setup, the model code, training, text generation, and a short comparison of the decoding methods.

What is included:

- character-level tokenizer
- sequence length 128 dataset
- manual multi-head masked self-attention
- 2 decoder blocks
- Adam training for 5 epochs
- greedy decoding
- temperature sampling
- top-k sampling
- findings PDF

Files:

- `homework4_solution.ipynb` - main notebook
- `HW4_findings.pdf` - findings write-up
- `hw4_decoding_samples.txt` - generated text samples
- `hw4_training_loss.png` - loss plot from training

If `shakespeare.txt` is in the folder, the notebook uses it. If it is not there, the notebook makes a small local text file so the code can still run without uploading or downloading anything.


