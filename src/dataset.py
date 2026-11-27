"""Source-aware corpus splitting and next-character prediction windows."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from itertools import accumulate
from random import Random

import torch
from torch.utils.data import Dataset

START_DOCUMENT = "[START DOCUMENT]"
END_DOCUMENT = "[END DOCUMENT]"


def parse_corpus_documents(text: str) -> list[str]:
    """Extract intact labeled documents from Part 1; reject broken boundaries.

    Every resulting document retains its START/END markers. Only whitespace is
    permitted outside document boundaries.
    """
    if not isinstance(text, str):
        raise TypeError("Corpus must be text")

    documents: list[str] = []
    chunk: list[str] = []
    in_document = False
    for line_number, line in enumerate(text.splitlines(keepends=True), 1):
        marker = line.rstrip("\r\n")
        if marker == START_DOCUMENT:
            if in_document:
                raise ValueError(f"Nested document start at line {line_number}")
            in_document = True
            chunk = [line]
        elif marker == END_DOCUMENT:
            if not in_document:
                raise ValueError(f"Document end without start at line {line_number}")
            chunk.append(line)
            document = "".join(chunk)
            if not any(part.strip() for part in chunk[1:-1]):
                raise ValueError(f"Empty document at line {line_number}")
            documents.append(document)
            chunk = []
            in_document = False
        elif in_document:
            chunk.append(line)
        elif line.strip():
            raise ValueError(f"Unexpected text outside a document at line {line_number}")

    if in_document:
        raise ValueError("Unterminated [START DOCUMENT] section")
    if not documents:
        raise ValueError("Corpus contains no labeled documents")
    return documents


def split_documents(
    documents: Sequence[str],
    *,
    val_fraction: float = 0.10,
    seed: int = 1337,
) -> tuple[list[str], list[str]]:
    """Split entire documents rather than allowing source leakage.

    The same seed and corpus document order always produce the same assignment.
    Returned partitions retain their original corpus ordering.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if len(documents) < 2:
        raise ValueError("At least two source documents are required for a train/validation split")

    indices = list(range(len(documents)))
    Random(seed).shuffle(indices)
    num_validation = min(len(documents) - 1, max(1, round(len(documents) * val_fraction)))
    validation_indices = set(indices[:num_validation])
    train = [doc for i, doc in enumerate(documents) if i not in validation_indices]
    validation = [doc for i, doc in enumerate(documents) if i in validation_indices]
    return train, validation


class NextTokenDataset(Dataset):
    """Windowed, shift-by-one autoregressive examples confined to one document.

    Given tokens [10, 11, 12, 13] and context_size=3, the sample is:
      x = [10, 11, 12], y = [11, 12, 13]
    A context window never crosses a document boundary.
    """

    def __init__(self, token_documents: Sequence[torch.Tensor], context_size: int, *, stride: int = 1):
        if context_size < 1:
            raise ValueError("context_size must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.context_size = context_size
        self.stride = stride
        self.documents: list[torch.Tensor] = []
        counts: list[int] = []
        for document in token_documents:
            if not isinstance(document, torch.Tensor) or document.ndim != 1 or document.dtype != torch.long:
                raise TypeError("Each document must be a one-dimensional torch.long tensor")
            self.documents.append(document)
            count = max(0, (len(document) - context_size - 1) // stride + 1)
            counts.append(count)
        self._cumulative = list(accumulate(counts))
        self._length = self._cumulative[-1] if self._cumulative else 0
        if not self._length:
            raise ValueError("No documents are long enough for context_size + 1 tokens")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(index, int):
            raise TypeError("Dataset index must be an integer")
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(f"Dataset index out of bounds: {index}")
        document_idx = bisect_right(self._cumulative, index)
        previous = self._cumulative[document_idx - 1] if document_idx else 0
        offset = (index - previous) * self.stride
        tokens = self.documents[document_idx]
        return (
            tokens[offset:offset + self.context_size],
            tokens[offset + 1:offset + self.context_size + 1],
        )
