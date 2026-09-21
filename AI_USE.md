# AI Use

I used ChatGPT/Codex while working on this assignment. I mainly used it for help with the notebook layout and for checking that my PyTorch code stayed inside the rules for the assignment.

I went back through the code myself and checked these parts:

- the tokenizer uses characters
- the input and target windows are shifted by one character
- the attention mask is causal
- the model does not use `nn.Transformer`
- the model does not use `nn.MultiheadAttention`
- the decoding methods are separate

The main issue I ran into was the dataset file. I did not have the course `shakespeare.txt` file available, so I added a small local fallback text in the notebook. That lets the notebook run offline, but the real course file would give better generated text.

