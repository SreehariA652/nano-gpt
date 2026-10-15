# nano-gpt

A character-level GPT built from scratch in PyTorch, written while following Andrej Karpathy's lecture [Let's build GPT: from scratch, in code, spelled out](https://youtu.be/kCc8FmEb1nY).

This is a **learning project**, not an original architecture. I followed the lecture step by step to understand how a transformer actually works, then adapted it: I trained it on a different dataset, fixed a few things I noticed along the way, and compared several model sizes. The notes below are what I learned in the process.

## What's in here

| File | What it is |
|---|---|
| `gpt-dev.ipynb` | Notebook from following the lecture, building up piece by piece |
| `bigram.py` | The baseline bigram model (a single lookup table, no attention) |
| `ver2.py` | The full GPT: self-attention, multi-head attention, feed-forward blocks, residual connections, LayerNorm, dropout |
| `poems.txt` | Training data: Walt Whitman's *Leaves of Grass* (public domain, from Project Gutenberg), about 730K characters |

## The build-up (following the lecture)

1. **Bigram model:** predict the next character from the current one only.
2. **Self-attention:** let each position look at earlier positions, using a causal mask so it can't see the future.
3. **Multi-head attention:** several attention heads in parallel, concatenated and projected.
4. **Feed-forward layer:** computation on each position after the communication step.
5. **Residual connections, LayerNorm, dropout:** what makes a deeper stack trainable.

## My own spin

The lecture trains on Shakespeare. I wanted to see how the model behaves on something different, so I trained it on Whitman's poetry, which has long lines, free verse and unusual punctuation.

Changes I made beyond the lecture code:

- **Loads the dataset from a local file** instead of downloading it, so I could swap in my own text.
- **Attention scaling:** I noticed the scale factor used `C` (the embedding size) instead of the head size. I changed it to `k.shape[-1] ** -0.5`.
- **Dropout after the attention projection:** `MultiHeadAttention` defined a dropout layer but never applied it. Fixed.
- **`model.eval()` before generation:** the model was still in training mode after the last loss estimate, so dropout was active while sampling. Turning it off before generating fixed that.
- **Saves the trained model** (`model.pt`) and the generated text (`generated.txt`).
- **Smaller configs:** I reduced `eval_iters` and the model size so experiments run quickly.

## Experiments

Same data, same 90/10 train/validation split, same seed. Model size and training settings changed between runs (see below).

| Config                      | Layers | Embedding dim | Heads | Params | Iterations | Val loss |
|-----------------------------|--------|---------------|-------|--------|------------|----------|
| Original (lecture settings) | 6      | 384           | 6     | ~10.7M | 5000       | ~1.4     |
| Small                       | 4      | 128           | 4     | ~0.8M  | 3000       | ~1.63    |
| Medium                      | 4      | 192           | 6     | ~1.8M  | 5000       | ~1.50    |

Validation loss starts at about 4.6 (close to random guessing over the character vocabulary) and falls steadily. And not to mention the training time was also significantly reduced.

## What I noticed

- **A much smaller model gets most of the way there.** The medium model is within about 0.1 of the original's validation loss, with a fraction of the parameters and training time. More size didn't buy much on a dataset this small.
- **It learns structure before spelling.** The model picked up line breaks, indentation, section numbers, parentheses that open and later close, and Whitman's cadence well before it got the vocabulary right. Many words are still invented.
- **The train/val gap tells you when to stop.** In the medium run the gap grew from about 0.15 to about 0.22. That's mild, but it's the sign that more training on 730K characters won't help much.
- **Small bugs hide in plain sight.** The dropout and `eval()` issues didn't crash anything. I only found them by reading the code carefully and asking why the output looked noisier than expected.

## Sample output (medium model)

```
  O sun and large, these ragain and divine bury blossoms, brawn drops,
  Slains from you clinking with thumps from thee, or workly commanne;
  Shell I come lands and war-purchanged and lauk or north.

  (As from you laws for you are! from man, to sing and on our death,
  In man not langual gripent of all people dicks, those white come religate around
```

It isn't meaningful poetry, but the form is recognisably Whitman.

## Run it

```bash
pip install torch
python ver2.py
```

Put your own text in `poems.txt` to train on something else. Hyperparameters are at the top of `ver2.py`.

## Credits

- Andrej Karpathy for the [lecture](https://youtu.be/kCc8FmEb1nY) and the original code this is based on.
- Walt Whitman, *Leaves of Grass*, via [Project Gutenberg](https://www.gutenberg.org).
