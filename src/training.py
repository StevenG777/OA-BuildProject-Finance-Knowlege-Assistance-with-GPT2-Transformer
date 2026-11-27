"""Part 4: reusable GPT training, validation, checkpoints and inference.

No corpus export or teacher-API calls occur in this module. A checkpoint
contains the vocabulary and model configuration so evaluation can run from
that checkpoint alone. Resume requires the original tokenized bundle.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .gpt import GPTLanguageModel
from .storage import atomic_write
from .tokenizer import CharacterTokenizer


def resolve_device(requested: str) -> torch.device:
    """Choose a supported compute device, rejecting unavailable explicit choices."""
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if requested == "auto":
        requested = (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return torch.device(requested)


def sample_batch(
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample fixed-length next-token windows, without a huge permutation.

    Sampling is with replacement, like the batch sampler in the mentor demo.
    The dataset itself prevents each window from crossing a source boundary.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(dataset) < 1:
        raise ValueError("dataset must contain training windows")
    indices = torch.randint(len(dataset), (batch_size,), generator=generator).tolist()
    examples = [dataset[i] for i in indices]
    x = torch.stack([pair[0] for pair in examples]).to(device)
    y = torch.stack([pair[1] for pair in examples]).to(device)
    return x, y


@torch.inference_mode()
def estimate_loss(
    model: GPTLanguageModel,
    dataset: Dataset,
    *,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
    seed: int = 2026,
) -> tuple[float, float]:
    """Estimate mean next-token cross entropy and perplexity on held-out data.

    Returns (mean_loss, exp(mean_loss)). The fixed evaluation seed gives
    comparable sampled windows across evaluations. This is a *sample estimate*,
    not the loss over every validation window.
    """
    if eval_batches < 1:
        raise ValueError("eval_batches must be positive")
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    try:
        losses: list[float] = []
        for _ in range(eval_batches):
            x, y = sample_batch(dataset, batch_size, device, generator)
            _, loss = model(x, y)
            assert loss is not None
            losses.append(float(loss.item()))
        mean = sum(losses) / len(losses)
        return mean, math.exp(mean) if mean < 709.0 else float("inf")
    finally:
        model.train(was_training)


def save_model_checkpoint(
    path: Path,
    *,
    model: GPTLanguageModel,
    tokenizer: CharacterTokenizer,
    step: int,
    corpus_sha256: str,
    optimizer: torch.optim.Optimizer | None = None,
    sampling_generator: torch.Generator | None = None,
    history: list[dict] | None = None,
) -> None:
    """Atomically save a portable checkpoint for generation or further training.

    torch.save uses Python serialization. Only load checkpoints you trust.
    Storing the tokenizer here means eval never needs to re-read the corpus.
    """
    if step < 0:
        raise ValueError("step must be nonnegative")
    payload = {
        "model_state": model.state_dict(),
        "model_config": model.model_config(),
        "vocab": list(tokenizer.vocab),
        "corpus_sha256": corpus_sha256,
        "step": int(step),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "sampling_rng_state": sampling_generator.get_state() if sampling_generator is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "history": list(history or []),
    }
    binary = io.BytesIO()
    torch.save(payload, binary)
    atomic_write(Path(path), binary.getvalue())


def load_model_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
) -> tuple[GPTLanguageModel, CharacterTokenizer, dict]:
    """Restore model + tokenizer from one checkpoint; optimizer is optional."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint: expected dictionary")
    required = {"model_state", "model_config", "vocab", "corpus_sha256", "step"}
    if not required.issubset(payload):
        raise ValueError(f"Invalid checkpoint: missing {sorted(required - set(payload))}")
    tokenizer = CharacterTokenizer(payload["vocab"])
    config = payload["model_config"]
    if config.get("vocab_size") != tokenizer.vocab_size:
        raise ValueError("Checkpoint vocabulary does not match model vocabulary size")
    model = GPTLanguageModel(**config)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device), tokenizer, payload


def require_matching_bundle(tokenizer: CharacterTokenizer, checkpoint: dict, bundle: dict) -> None:
    """Guard against accidentally resuming/evaluating on another tokenized corpus."""
    if bundle["corpus_sha256"] != checkpoint["corpus_sha256"]:
        raise ValueError("Tokenized corpus hash does not match checkpoint")
    if list(bundle["vocab"]) != list(tokenizer.vocab):
        raise ValueError("Tokenized vocabulary does not match checkpoint")
    if bundle["context_size"] != checkpoint["model_config"]["context_size"]:
        raise ValueError("Tokenized context size does not match checkpoint")


def train_model(
    *,
    model: GPTLanguageModel,
    tokenizer: CharacterTokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    corpus_sha256: str,
    save_path: Path,
    device: torch.device,
    steps: int,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    eval_interval: int = 50,
    eval_batches: int = 10,
    seed: int = 1337,
    resume_checkpoint: dict | None = None,
) -> list[dict]:
    """Perform `steps` NEW AdamW updates; log validation metrics and save often.

    Resume restores optimizer and RNG states when available, and increments the
    saved global step. Checkpoint publication is atomic at every eval interval
    and at the end. Checkpointing is per-interval, not after every update.
    """
    if steps < 1 or eval_interval < 1 or eval_batches < 1 or batch_size < 1:
        raise ValueError("steps, eval_interval, eval_batches and batch_size must be positive")
    if learning_rate <= 0 or weight_decay < 0 or grad_clip <= 0:
        raise ValueError("learning_rate and grad_clip must be positive; weight_decay nonnegative")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    sampling_generator = torch.Generator(device="cpu").manual_seed(seed)
    start_step = 0
    history: list[dict] = []

    if resume_checkpoint is not None:
        if resume_checkpoint.get("optimizer_state") is None:
            raise ValueError("This checkpoint cannot resume: optimizer state is missing")
        if resume_checkpoint.get("sampling_rng_state") is None:
            raise ValueError("This checkpoint cannot resume: sampling RNG state is missing")
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        # Honor the resumed optimizer's learning rate, not a new CLI default.
        # Explicit learning-rate changes should be made in the caller if desired.
        sampling_generator.set_state(resume_checkpoint["sampling_rng_state"])
        if resume_checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(resume_checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and resume_checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume_checkpoint["cuda_rng_state"])
        start_step = int(resume_checkpoint["step"])
        history = list(resume_checkpoint.get("history", []))

    model.to(device)
    for local_step in range(1, steps + 1):
        step = start_step + local_step
        model.train()
        x, y = sample_batch(train_dataset, batch_size, device, sampling_generator)
        _, loss = model(x, y)
        assert loss is not None
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if local_step % eval_interval == 0 or local_step == steps:
            val_loss, val_ppl = estimate_loss(
                model, val_dataset, batch_size=batch_size, eval_batches=eval_batches,
                device=device,
            )
            entry = {
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
            }
            history.append(entry)
            print(
                f"step {step:>6} | train_loss {entry['train_loss']:.4f} | "
                f"val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}",
                flush=True,
            )
            save_model_checkpoint(
                save_path, model=model, tokenizer=tokenizer, step=step,
                corpus_sha256=corpus_sha256, optimizer=optimizer,
                sampling_generator=sampling_generator, history=history,
            )
    return history


@torch.inference_mode()
def generate_text(
    model: GPTLanguageModel,
    tokenizer: CharacterTokenizer,
    prompt: str,
    *,
    token_count: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    """Continue a text prompt using the trained checkpoint and saved vocabulary."""
    if not prompt:
        raise ValueError("Prompt must be nonempty")
    if token_count < 0:
        raise ValueError("token_count must be nonnegative")
    token_ids = tokenizer.encode(prompt)
    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        tokens = model.generate(
            inputs, token_count, temperature=temperature, top_k=top_k
        )
        return tokenizer.decode(tokens[0].tolist())
    finally:
        model.train(was_training)
