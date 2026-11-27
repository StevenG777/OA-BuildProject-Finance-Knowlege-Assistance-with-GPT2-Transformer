# Part 1: Generate synthetic tutorial data using GPT-6 Luna

## 1. Provide your API key
```bash
export OPENAI_API_KEY="your_key_here"
```

## 2. Add the source text

Replace the placeholder text in `data/sources/compound_interest.txt`:

```text
URL: https://www.investor.gov/your-real-article-url
TOPIC: Compound Interest
CONTENT:
Paste the actual educational article text here.
```

## 3. Generate and inspect

Run **from the project root** (the directory containing `generate.py`):

```bash
python generate.py
```

The script saves JSONL output to `data/generated/finance_dataset.jsonl`, one
item each for tutorial, Q&A, and misconception (or a skipped record if unsupported).
Generated answers remain explicitly **unverified**. Existing record IDs are
skipped on rerun to avoid repeat requests and charges.
