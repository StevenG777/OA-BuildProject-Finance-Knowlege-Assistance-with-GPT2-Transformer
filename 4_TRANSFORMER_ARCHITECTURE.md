
# Part 4: Token embeddings + decoder-only GPT architecture

## Detailed Information

- Vocabulary size: 95 including `<UNK>`.
- Input `torch.long`: `(16, 256)` — 16 samples of 128 characters.
- `nn.Embedding(95, 256)`: output `(16, 256, 128)`.
- 4 attention heads, each dimension `128 / 4 = 32` (even, for RoPE).
- 2 Transformer blocks, each with pre-layer-normalization, residual self-attention and a position-wise feed-forward network.
- LM head: logits `(16, 256, 95)` — one distribution over 95 characters per position.
- Cross-entropy: one scalar loss, comparing logits to Part 2's one-token-shifted targets.

## Architecture

```text
Token IDs (B,T) : torch.long
       |
       v
nn.Embedding(vocab_size, n_embd)
       | (B,T,n_embd)
       v
[ Block ] x n_layer
  Pre-LayerNorm
  Causal multi-head self-attention
    - Linear Q,K,V projections per head
    - RoPE rotates Q,K by position
    - Scaled Q @ K.T / sqrt(head_size)
    - Strict lower-triangular attention mask, including diagonal
    - Softmax, attention dropout
    - Concatenate head outputs, output projection, dropout
  Residual addition
  Pre-LayerNorm
  FeedForward: Linear(n_embd,4*n_embd) -> ReLU -> Linear(4*n_embd,n_embd)
  Residual addition
       |
       v
Final LayerNorm -> Linear(n_embd,vocab_size)
       | (B,T,vocab_size)
       v
Optional cross-entropy loss, when targets are passed
```

## GPT structure and RoPE approach, with several explicit choices:

1. **Consistent shapes:** `forward(x, y)` always returns full `(B,T,V)` logits; unlike the mentor example it does not flatten returned logits when targets are supplied. Only the temporary values passed to `cross_entropy` are flattened.
2. **No KV cache yet:** the generator recomputes the most recent context for each new token. Cache support can be added after the model trains, without complicating Part 3's correctness tests.
3. **Parameterized dropout:** default 0.2 as in the mentor demo. `model.eval()` disables dropout during text generation and validation.
4. **RoPE head dimension:** `n_embd // n_head` must be even.
5. **Hyperparameters:** `model_config()` returns settings for a future Part 4 checkpoint, avoiding the need to recreate them by memory.