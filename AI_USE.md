# AI_USE.md

## Personal Parameters

- SID4 = 6812
- SEED = 6812
- SLICE = 812
- HP_ID = 2
- CLS_A = 2
- CLS_B = 9

## How I Used AI

I used ChatGPT/Codex while working on this assignment. It helped me set up the notebook sections, write the LangChain code for running the prompts, and draft the PyTorch self-attention model. I did not just submit the first version. I ran the notebook in Colab, checked the outputs, and changed parts that did not work well.

The parts I checked myself were the prompt outputs, the training losses, the attention heatmaps, and the causal mask check.

## One Thing AI Got Wrong

The first version of the prompt section expected an `OPENAI_API_KEY`. I did not want to use paid API calls, so that was not a good setup for me.

After that, I tried a local Hugging Face model, `flan-t5-base`. It ran, but some of the answers were bad. For example, it gave an incorrect answer for the jacket discount problem and made mistakes on simple arithmetic.

## How I Found It

I found the problem by reading the actual prompt outputs after the notebook ran. The code itself was running, but the answers were not reliable enough for the assignment.

I also hit this Colab error with a Hugging Face pipeline version:

```text
KeyError: Unknown task text2text-generation
```

## What I Changed

I changed the prompt section to use `Qwen/Qwen2.5-0.5B-Instruct` locally with Hugging Face, and I wrapped the generation function with LangChain. This avoided the OpenAI API key issue and gave better prompt outputs.

For the attention part, I checked the causal mask by printing the largest attention value above the diagonal. It was:

```text
0.0
```

That tells me the masked model was not attending to future tokens.
