"""Part 2 entry point: convert Part 1 corpus TXT into tokenized PyTorch data."""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.config import ROOT
from src.training_data import (
    build_training_bundle,
    load_training_bundle,
    save_training_bundle,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare character-level finance training data")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/finance_corpus_experimental.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "data/tokenized/finance_chars.pt")
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16, help="Preview batch size (not saved)")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    bundle = build_training_bundle(
        args.corpus.read_bytes(),
        context_size=args.context_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    save_training_bundle(args.output, bundle)
    tokenizer, train_dataset, val_dataset, reloaded = load_training_bundle(args.output)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    x, y = next(iter(train_loader))

    print(f"Corpus: {args.corpus}")
    print(f"Tokenized bundle: {args.output}")
    print(f"Source documents: {len(reloaded['train_documents'])} train / "
          f"{len(reloaded['val_documents'])} validation")
    print(f"Token counts: {sum(len(doc) for doc in reloaded['train_documents']):,} train / "
          f"{sum(len(doc) for doc in reloaded['val_documents']):,} validation")
    print(f"Vocabulary size: {tokenizer.vocab_size} (including <UNK>)")
    print(f"Unknown validation characters: "
          f"{sum((doc == tokenizer.unk_id).sum().item() for doc in reloaded['val_documents']):,}")
    print(f"Next-token windows: {len(train_dataset):,} train / {len(val_dataset):,} validation")
    print(f"Example batch: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}, dtype={x.dtype}")
    print(f"Decoded first context sample: {tokenizer.decode(x[0].tolist()[:80])!r}")
    return args.output


if __name__ == "__main__":
    main()
