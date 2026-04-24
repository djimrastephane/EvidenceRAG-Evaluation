% Why 12 Attention Heads?
% Non-Technical Model Architecture Explanation

# Why your model has 12 attention heads

`Your pipeline loads the pretrained encoder all-MiniLM-L6-v2.`

## The simple idea

- Think of attention heads as `12 small reading teams` looking at the same sentence in parallel.
- Each team looks for different useful links between words, labels, and numbers.
- The model then combines what those teams found.

## Why 12?

- The model stores each token using `384 numbers` (`hidden_size = 384`).
- Those `384` numbers are split equally across `12` heads.
- `384 / 12 = 32`
- So each head works on `32 values`.

## Why not 16 or 20?

- `20` does not fit cleanly because `384 / 20 = 19.2`
- `16` would fit mathematically because `384 / 16 = 24`
- But `16` would be a different model design
- Your pipeline does not choose the head count at runtime; it inherits the value from the pretrained model checkpoint

## What happens inside one layer

1. Start with one token representation: `384 values`
2. Split into `12 heads`
3. Each head works on `32 values`
4. Merge the results back into one `384-value` representation

## Practical takeaway

- More heads is not automatically better
- The chosen value is part of the model’s speed/quality tradeoff
- Your model uses `12 heads per layer` and `6 layers total`
