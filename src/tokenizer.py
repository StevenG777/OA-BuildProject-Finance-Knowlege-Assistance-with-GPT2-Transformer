"""Deterministic character-level tokenizer for the small finance GPT pilot.

The delimiter strings from corpus export are *ordinary characters*, not single
special tokens. ID 0 is reserved for out-of-vocabulary characters in validation
text and future prompts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

UNK_TOKEN = "<UNK>"


class CharacterTokenizer:
    """Map Unicode characters to integer IDs and back."""

    def __init__(self, vocabulary: Sequence[str]):
        vocab = list(vocabulary)
        if not vocab or vocab[0] != UNK_TOKEN:
            raise ValueError("Vocabulary must begin with the <UNK> placeholder")
        if len(vocab) != len(set(vocab)):
            raise ValueError("Duplicate vocabulary entries")
        if any(not isinstance(char, str) or len(char) != 1 for char in vocab[1:]):
            raise ValueError("Each non-special vocabulary item must be one character")

        self.vocab = vocab
        self.char_to_id = {char: index for index, char in enumerate(vocab)}
        self.unk_id = 0

    @classmethod
    def from_documents(cls, documents: Iterable[str]) -> "CharacterTokenizer":
        """Learn vocabulary from training documents only, with stable ordering."""
        characters = set()
        for document in documents:
            if not isinstance(document, str):
                raise TypeError("Document must be text")
            characters.update(document)
        if not characters:
            raise ValueError("Cannot build a tokenizer from empty documents")
        return cls([UNK_TOKEN, *sorted(characters)])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, *, allow_unknown: bool = True) -> list[int]:
        """Convert characters to IDs; optionally reject unknown characters."""
        if not isinstance(text, str):
            raise TypeError("Expected a string")
        if allow_unknown:
            return [self.char_to_id.get(char, self.unk_id) for char in text]
        try:
            return [self.char_to_id[char] for char in text]
        except KeyError as exc:
            raise ValueError(f"Character absent from training vocabulary: {exc.args[0]!r}") from exc

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode token IDs. Unknown character IDs render as '<UNK>'."""
        chars = []
        for token_id in token_ids:
            index = int(token_id)
            if index < 0 or index >= self.vocab_size:
                raise ValueError(f"Invalid token ID: {index}")
            chars.append(self.vocab[index])
        return "".join(chars)
