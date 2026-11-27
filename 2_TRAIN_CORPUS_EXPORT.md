# Part 2: Combined finance text corpus

This changes only the corpus export stage, not generation, tokenization, or training.

## Commands

Conservative default (approved records only):

```bash
python export_corpus.py
```

Current data has no approved records; the default deliberately does not create an output file.

Experimental, including unverified generated records but excluding source-evidence mismatches:

```bash
python export_corpus.py --allow-unverified
```

Experimental, including ALL unverified records (including failed exact evidence matches):

```bash
python export_corpus.py --allow-unverified --allow-evidence-mismatch
```

By default, experimental exports write to `data/combined/finance_corpus_experimental.txt`; approved-only exports write to `data/combined/finance_corpus.txt`. Override with `--output <path>`.

## Format

```text
[START DOCUMENT]

[START SOURCE]
Topic: ...

<original article text>
[END SOURCE]

[START TUTORIAL]
Topic: ...
Subtopic: ...

Answer: ...
[END TUTORIAL]

[START QA]
Topic: ...
Subtopic: ...

Question: ...

Answer: ...
[END QA]

[START MISCONCEPTION]
Topic: ...
Subtopic: ...

Question: ...

Answer: ...
[END MISCONCEPTION]

[END DOCUMENT]
```

Markers are plain text delimiters; they are not yet tokenizer special tokens. When using a character tokenizer, each character is assigned an integer token ID. The language model later maps those IDs to trainable embeddings.

## Safety and data integrity

- No teacher API requests.
- Source TXT and generated JSONL are never modified.
- Only manifest entries marked `complete` are considered.
- Source and JSONL SHA-256 digests and the expected record-ID set must match the manifest.
- A failed export does not replace an existing corpus TXT.
- `unverified` data remain experimental and are not approved financial guidance.

Next part: source-aware train/validation split and character-level tokenization. Embeddings belong to the PyTorch model, not this exporter.
