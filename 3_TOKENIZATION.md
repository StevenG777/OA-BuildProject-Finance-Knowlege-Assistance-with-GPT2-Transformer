# Part 3: Character-level tokenization and PyTorch next-token Dataset

This component consumes the **Part 1 UTF-8 corpus** and prepares reproducible input/target tensors for a future decoder-only Transformer. No teacher-API calls and no modification of original TXT or generated JSONL files.

## Run

```bash
python prepare_training_data.py \
  --corpus data/combined/finance_corpus_experimental.txt \
  --output data/tokenized/finance_chars.pt \
  --context-size 128 \
  --batch-size 16 \
  --val-fraction 0.10 \
  --seed 1337
```

`--batch-size` controls only the sample output printed by this command; `DataLoader` batch sizes can be changed at training time. Context size governs the saved windowing setting and future model's supported sequence length.

## Output

The single saved `data/tokenized/finance_chars.pt` bundle contains:

- `vocab`: sorted training-only character vocabulary, with `<UNK>` as ID 0.
- `train_documents`: 1-D `torch.long` token-ID tensors, one per source document.
- `val_documents`: 1-D `torch.long` token-ID tensors, one per held-out source document.
- `context_size`, `seed`, `val_fraction`: reproducibility settings.
- `corpus_sha256`: SHA-256 of the input UTF-8 corpus bytes for provenance (not a schema or prompt revision).

The `.pt` bundle is atomically published as one file. It includes both tokenizer vocabulary and token IDs, preventing a mismatch between independently saved files.

## Train/validation integrity

- The corpus is parsed into `[START DOCUMENT] ... [END DOCUMENT]` units from the Part 1 exporter.
- The seed-driven split assigns entire documents to train or validation, so examples from one source aren't split across sets.
- `NextTokenDataset` generates windows *within a single document* and shifts targets one character forward.
- This is intended for a small experimental character-level GPT. It doesn't guarantee financial factual accuracy or perform content review.

Embedding and model architecture are **Part 3**, not part of this implementation. Next: initialize a learnable `torch.nn.Embedding(vocab_size, n_embd)`, optional positional encoding, and the causal Transformer.
