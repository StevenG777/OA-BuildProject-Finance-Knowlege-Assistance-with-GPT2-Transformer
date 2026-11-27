"""Part 4 CLI: train a finance GPT, resume, evaluate, or complete a prompt.

Uses the saved Part 2 token bundle; does NOT call the teacher-generation API.
Example:
  python train.py train --bundle data/tokenized/finance_chars.pt --steps 300 --save checkpoints/finance_gpt.pth
  python train.py eval --load checkpoints/finance_gpt.pth --prompt 'Question: What is diversification? Answer:'
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.gpt import GPTLanguageModel
from src.training import (
    estimate_loss,
    generate_text,
    load_model_checkpoint,
    require_matching_bundle,
    resolve_device,
    train_model,
)
from src.training_data import load_training_bundle


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finance GPT: train or generate with a saved checkpoint")
    commands = parser.add_subparsers(dest="mode", required=True)

    train = commands.add_parser("train", help="Train from Part 2 data, optionally resuming a checkpoint")
    train.add_argument("--bundle", type=Path, default=Path("data/tokenized/finance_chars.pt"))
    train.add_argument("--save", type=Path, default=Path("checkpoints/finance_gpt.pth"))
    train.add_argument("--resume", type=Path, help="Checkpoint to continue from; --steps means additional updates")
    train.add_argument("--steps", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--n-embd", type=int, default=None, help="Default 128 for new training")
    train.add_argument("--n-head", type=int, default=None, help="Default 4 for new training")
    train.add_argument("--n-layer", type=int, default=None, help="Default 2 for new training")
    train.add_argument("--dropout", type=float, default=None, help="Default 0.2 for new training")
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--eval-interval", type=int, default=50)
    train.add_argument("--eval-batches", type=int, default=10)
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")

    evaluate = commands.add_parser("eval", help="Load checkpoint, optionally estimate validation loss and generate")
    evaluate.add_argument("--load", type=Path, required=True)
    evaluate.add_argument("--prompt", type=str, default="Question: What is diversification?\nAnswer:")
    evaluate.add_argument("--token-count", type=int, default=120)
    evaluate.add_argument("--temperature", type=float, default=1.0)
    evaluate.add_argument("--top-k", type=int)
    evaluate.add_argument("--seed", type=int, default=1337)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    evaluate.add_argument("--bundle", type=Path, help="Optional bundle for validation-loss/perplexity estimate")
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--eval-batches", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.mode == "train":
        tokenizer, train_dataset, val_dataset, bundle = load_training_bundle(args.bundle)
        resume = None
        if args.resume is not None:
            model, checkpoint_tokenizer, resume = load_model_checkpoint(args.resume, device)
            require_matching_bundle(checkpoint_tokenizer, resume, bundle)
            for flag, key in (("n_embd", "n_embd"), ("n_head", "n_head"),
                              ("n_layer", "n_layer"), ("dropout", "dropout")):
                if getattr(args, flag) is not None and getattr(args, flag) != resume["model_config"][key]:
                    raise ValueError(f"--{flag.replace('_', '-')} conflicts with resumed checkpoint")
        else:
            model = GPTLanguageModel(
                vocab_size=tokenizer.vocab_size,
                context_size=bundle["context_size"],
                n_embd=128 if args.n_embd is None else args.n_embd,
                n_head=4 if args.n_head is None else args.n_head,
                n_layer=2 if args.n_layer is None else args.n_layer,
                dropout=0.2 if args.dropout is None else args.dropout,
            ).to(device)
        print(
            f"Device: {device} | vocab: {tokenizer.vocab_size} | context: {model.context_size} | "
            f"parameters: {sum(p.numel() for p in model.parameters()):,}"
        )
        print(f"Training windows: {len(train_dataset):,} | Validation windows: {len(val_dataset):,}")
        train_model(
            model=model, tokenizer=tokenizer, train_dataset=train_dataset,
            val_dataset=val_dataset, corpus_sha256=bundle["corpus_sha256"],
            save_path=args.save, device=device, steps=args.steps,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            eval_interval=args.eval_interval, eval_batches=args.eval_batches,
            seed=args.seed, resume_checkpoint=resume,
        )
        print(f"Checkpoint saved: {args.save}")
        return

    model, tokenizer, checkpoint = load_model_checkpoint(args.load, device)
    print(f"Checkpoint: {args.load} | step: {checkpoint['step']} | device: {device}")
    if args.bundle is not None:
        _, _, val_dataset, bundle = load_training_bundle(args.bundle)
        require_matching_bundle(tokenizer, checkpoint, bundle)
        loss, perplexity = estimate_loss(
            model, val_dataset, batch_size=args.batch_size,
            eval_batches=args.eval_batches, device=device,
        )
        print(f"Validation loss: {loss:.4f} | Perplexity: {perplexity:.2f}")
    unknowns = sum(char not in tokenizer.char_to_id for char in args.prompt)
    if unknowns:
        print(f"Note: {unknowns} prompt character(s) absent from saved vocabulary map to <UNK>.")
    generated = generate_text(
        model, tokenizer, args.prompt, token_count=args.token_count,
        device=device, temperature=args.temperature, top_k=args.top_k,
    )
    print("\n--- Generated continuation (prompt included) ---")
    print(generated)


if __name__ == "__main__":
    main()
