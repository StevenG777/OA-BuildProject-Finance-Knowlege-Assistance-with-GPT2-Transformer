# Finance Transformer Pilot

This project uses a manually collected finance article as reference text and calls
an OpenAI teacher model to generate **unverified** Finance 101 examples.
It does not train a Transformer, fact-check finance statements, or verify formulas.

## Requirements

- Python 3.10+ (choose a Python version compatible with the pinned PyTorch 2.6 line)
- An OpenAI API account with access/billing enabled for your selected model
- A source file with a URL, topic, and pasted article text

## 1. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies
```bash
python -m pip install -r requirements.txt
```

The `requirements.txt` here keeps your original research/data-science packages and
adds `pytest` for the offline tests. The generator itself only needs `openai`;
other libraries are for your wider Transformer learning project.

## Old Code layout

- `generate.py` — orchestrates the complete generation flow
- `src/config.py` — settings and version tags
- `src/source.py` — parses source text and computes its fingerprint
- `src/prompts.py` — teacher instructions and prompt construction
- `src/schemas.py` — strict Structured Outputs schema
- `srd/api.py` — API client and request/response handling
- `src/validation.py` — mechanical checks (not fact checking)
- `src/storage.py` — persistent JSONL output and ID deduplication
- `tests/` — offline unit tests
