# Part 5: GPT Training, Evaluation, Checkpoints, and Inference

## Training

Run from the **project root** where `train_inference.py` and `src/` live:

```bash
python train_inference.py train \
    --bundle data/tokenized/finance_chars_ctx_256.pt \
    --save checkpoints/finance_gpt.pth \
    --steps 3000 \
    --batch-size 16 \
    --n-embd 128 \
    --n-head 4 \
    --n-layer 2 \
    --dropout 0.2 \
    --learning-rate 0.001 \
    --weight-decay 0.01 \
    --eval-interval 50 \
    --eval-batches 10 \
    --device auto
```

Each step does `sample_batch -> model(x,y) -> cross_entropy -> loss.backward() -> clip_grad_norm_ -> AdamW.step()`. The training batch is randomly sampled **with replacement** `NextTokenDataset`, keeping windows inside their originating source document. To avoid building a giant shuffled index permutation, the trainer samples window indices directly.

Every `eval_interval` updates (and at the end), it calculates a **sample estimate** of validation cross-entropy and perplexity, prints both metrics, and atomically saves a checkpoint. Validation batches use a fixed evaluation random seed for comparability across steps. The saved `history` contains each evaluation's step, last training-batch loss, validation loss, and validation perplexity.

Mathematically, perplexity is `exp(mean next-token cross-entropy)`. Here tokens are **characters**, not words, and the validation loss is estimated from sampled batches rather than every available window.

## Inference using checkpoint alone

```bash
python train_inference.py eval \
    --load checkpoints/finance_gpt.pth \
    --prompt 'Question: What is diversification?' \
    --token-count 256 \
    --temperature 0.8 \
    --top-k 20 \
    --device auto
```

```bash
python train_inference.py eval \
  --load checkpoints/finance_gpt.pth \
  --prompt $'[START QA]\nTopic: Diversification\n\nSubtopic: Diversification\n\nQuestion: What is diversification?\n\nAnswer: ' \
  --token-count 256 \
  --temperature 1.0 \
  --top-k 40 \
  --device auto
```

### Optional validation in eval mode

Supply the **same** tokenized bundle used in training to estimate held-out validation metrics:

```bash
python train_inference.py eval \
    --load checkpoints/finance_gpt.pth \
    --bundle data/tokenized/finance_chars.pt \
    --batch-size 16 \
    --eval-batches 10 \
    --prompt 'Question: What is risk? Answer:' \
    --token-count 120
```

The CLI refuses to validate against a bundle with a different corpus SHA-256, vocabulary, or context size.

## Resuming training

```bash
python train_inference.py train \
    --bundle data/tokenized/finance_chars_ctx_128.pt \
    --resume checkpoints/finance_gpt.pth \
    --save checkpoints/finance_gpt.pth \
    --steps 100 \
    --batch-size 16 \
    --eval-interval 25
```

Here `--steps 100` means **100 additional optimizer updates**. A checkpoint saved at step 300 is advanced to step 400. The model weights, AdamW moments, and sampling state are restored, along with CPU/CUDA random states as available. The resumed optimizer retains the saved learning rate and weight decay; the new command's learning-rate and weight-decay values are not used to modify a restored optimizer. Use the same tokenized bundle. Checkpoints save periodically, not at every update, so updates since the latest saved checkpoint may need to be replayed after a crash. Exact numerical replay may vary by hardware/device, nondeterministic kernels, and changes to code.

The `--n-embd`, `--n-head`, `--n-layer`, `--dropout` arguments are used for fresh training; a resumed model loads its saved configuration and rejects any explicitly conflicting model settings.

## What is saved in `finance_gpt.pth`?

- `model_state`: learned PyTorch parameters
- `model_config`: vocab size, embedding size, number of heads, layers, context size, dropout
- `vocab`: original Part 2 character vocabulary
- `corpus_sha256`: training-corpus identity
- `step`: total number of completed optimizer updates
- `optimizer_state`: AdamW state for resuming
- `sampling_rng_state`, `torch_rng_state`, `cuda_rng_state`: stochastic states when available
- `history`: logged sampled metrics

Checkpoint publication uses `storage.atomic_write()`: it writes and flushes a temporary file, then replaces the old checkpoint when serialization succeeds. The checkpoint is distinct from the Part 2 `data/tokenized/finance_chars.pt` file. Treat model checkpoints as trusted local artifacts even though `torch.load(weights_only=True)` is used.

## Limitations and next enhancements

- Character-level tokenization means `context_size=128` represents 128 characters, not 128 words. Generation will be slow compared with BPE-based systems and may output `<UNK>`.
- The model is trained **from scratch** and does not become a reliable finance advisor merely by seeing the finance corpus.
- Training uses no automatic learning-rate scheduling, early stopping, mixed precision, KV cache, or gradient accumulation yet.
- The corpus used for the experimental demo contains teacher-generated, **unverified** educational records. Quality review and stronger downstream evaluation remain separate tasks.
- The full `.pth` checkpoint includes optimizer state and is intended for local trusted use. Save a weights-only artifact separately if publishing inference models.
