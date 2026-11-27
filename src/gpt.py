"""Part 3: a small causal, decoder-only GPT inspired by mentor Lecture 6.

The input is Part 2's torch.long tensor of token IDs: (batch, time).
The output logits are (batch, time, vocabulary size). Optional targets
produce a scalar next-token cross-entropy loss.

Positions use RoPE (rotary positional embeddings) inside attention, rather
than an additive, learned position-embedding table. No KV cache is used yet;
generation recomputes at most ``context_size`` tokens each step.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def build_rope(seq_len: int, head_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary cos/sin tables, each shaped (seq_len, head_size // 2)."""
    # Calculate angles in float32 for stability, then cast for mixed precision.
    inverse_frequencies = 1.0 / (
        10000 ** (torch.arange(0, head_size, 2, device=device, dtype=torch.float32) / head_size)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inverse_frequencies)
    return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate even/odd features of (batch, time, head_size) query/key tensors."""
    even, odd = x[..., 0::2], x[..., 1::2]
    cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)  # (1, time, head_size/2)
    return torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    ).flatten(-2)


class Head(nn.Module):
    """One causal attention head with RoPE applied to queries and keys."""

    def __init__(self, head_size: int, n_embd: int, context_size: int, dropout: float):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # A registered buffer moves with .to(device); no gradients or checkpoint weight.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_size, context_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        _, time, _ = x.shape
        q = apply_rope(self.query(x), cos, sin)
        k = apply_rope(self.key(x), cos, sin)
        v = self.value(x)
        # Scaled dot-product attention: (B,T,D) @ (B,D,T) -> (B,T,T).
        scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        scores = scores.masked_fill(~self.causal_mask[:time, :time], float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        return weights @ v  # (B,T,D)


class MultiHeadAttention(nn.Module):
    """Run heads in parallel, concatenate their outputs, project to n_embd."""

    def __init__(self, n_head: int, head_size: int, n_embd: int, context_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, context_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([head(x, cos, sin) for head in self.heads], dim=-1)
        return self.dropout(self.proj(concatenated))


class FeedForward(nn.Module):
    """Independent position-wise MLP: n_embd -> 4*n_embd -> n_embd."""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-layer-normalized Transformer block with residual connections."""

    def __init__(self, n_embd: int, n_head: int, context_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, context_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x), cos, sin)
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Small decoder-only GPT compatible with Part 2's NextTokenDataset.

    Example:
        model = GPTLanguageModel(vocab_size=95, n_embd=128, context_size=128,
                                 n_head=4, n_layer=2)
        logits, loss = model(x, y)  # x/y are (B,T) torch.long tensors
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 128,
        context_size: int = 128,
        n_head: int = 4,
        n_layer: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (vocab_size, n_embd, context_size, n_head, n_layer)):
            raise ValueError("vocab_size, n_embd, context_size, n_head, n_layer must be positive integers")
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if (n_embd // n_head) % 2:
            raise ValueError("RoPE requires an even head_size (n_embd // n_head)")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout_rate = dropout
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, context_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def model_config(self) -> dict:
        """Hyperparameters to save alongside future model checkpoints."""
        return {
            "vocab_size": self.vocab_size,
            "n_embd": self.n_embd,
            "context_size": self.context_size,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "dropout": self.dropout_rate,
        }

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.ndim != 2 or idx.dtype != torch.long:
            raise ValueError("idx must be a (batch, time) torch.long tensor")
        batch, time = idx.shape
        if batch < 1 or not 1 <= time <= self.context_size:
            raise ValueError(f"batch must be nonempty and time must be in [1, {self.context_size}]")
        if targets is not None:
            if targets.shape != idx.shape or targets.dtype != torch.long:
                raise ValueError("targets must match idx shape and have torch.long dtype")

        x = self.token_embedding_table(idx)  # (B,T,n_embd)
        cos, sin = build_rope(time, self.n_embd // self.n_head, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.ln_f(x))  # (B,T,vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * time, self.vocab_size), targets.reshape(batch * time)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Sample new characters; call model.eval() first to disable dropout.

        This simple non-cached decoder recomputes the most recent context
        for every step. Its weights are RANDOM until Part 4 trains the model.
        """
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k is not None and not (isinstance(top_k, int) and 1 <= top_k <= self.vocab_size):
            raise ValueError("top_k must be between 1 and vocab_size")
        # Validate nonempty prompt even when max_new_tokens=0.
        if idx.ndim != 2 or idx.dtype != torch.long or idx.shape[0] < 1 or idx.shape[1] < 1:
            raise ValueError("idx must be a nonempty (batch, time) torch.long tensor")
        for _ in range(max_new_tokens):
            context = idx[:, -self.context_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
