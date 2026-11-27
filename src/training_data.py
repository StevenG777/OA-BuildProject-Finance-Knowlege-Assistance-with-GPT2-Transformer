"""Build and persist one self-contained, reproducible tokenized-data bundle.

This stage does not initialize a model or call the teacher API. The single .pt
file contains training/validation document tensors, the tokenizer vocabulary,
and data preparation settings, avoiding multi-file consistency problems.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import torch

from .dataset import NextTokenDataset, parse_corpus_documents, split_documents
from .storage import atomic_write
from .tokenizer import CharacterTokenizer


def build_training_bundle(
    corpus_bytes: bytes,
    *,
    context_size: int = 128,
    val_fraction: float = 0.10,
    seed: int = 1337,
) -> dict:
    """Prepare train/validation tokens without crossing original source boundaries."""
    if context_size < 1:
        raise ValueError("context_size must be >= 1")
    text = corpus_bytes.decode("utf-8")
    documents = parse_corpus_documents(text)
    train_documents, val_documents = split_documents(
        documents, val_fraction=val_fraction, seed=seed
    )
    tokenizer = CharacterTokenizer.from_documents(train_documents)
    train_tokens = [
        torch.tensor(tokenizer.encode(doc, allow_unknown=False), dtype=torch.long)
        for doc in train_documents
    ]
    val_tokens = [
        torch.tensor(tokenizer.encode(doc), dtype=torch.long)
        for doc in val_documents
    ]

    # Fail before publishing a bundle that cannot feed the model.
    NextTokenDataset(train_tokens, context_size)
    NextTokenDataset(val_tokens, context_size)

    return {
        "vocab": tokenizer.vocab,
        "train_documents": train_tokens,
        "val_documents": val_tokens,
        "context_size": context_size,
        "seed": seed,
        "val_fraction": val_fraction,
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
    }


def save_training_bundle(path: Path, bundle: dict) -> None:
    """Atomically publish one PyTorch bundle, preserving an earlier one on failure."""
    binary = io.BytesIO()
    torch.save(bundle, binary)
    atomic_write(path, binary.getvalue())


def load_training_bundle(path: Path) -> tuple[CharacterTokenizer, NextTokenDataset, NextTokenDataset, dict]:
    """Load trusted local bundle and reconstruct both PyTorch datasets.

    weights_only=True restricts the types PyTorch may unpickle. Still use only
    bundles you created or otherwise trust.
    """
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict):
        raise ValueError("Invalid tokenized bundle: expected a dictionary")
    tokenizer = CharacterTokenizer(bundle["vocab"])
    context_size = bundle["context_size"]
    train = NextTokenDataset(bundle["train_documents"], context_size)
    val = NextTokenDataset(bundle["val_documents"], context_size)
    return tokenizer, train, val, bundle
