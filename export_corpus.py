"""CLI orchestrator for publishing training text without teacher-model API calls."""

import argparse
import logging

from src.config import ROOT
from src.export import ExportOptions, export_corpus


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export finance JSONL + source TXT as training text")
    parser.add_argument("--sources", type=str, default=str(ROOT / "data/sources"))
    parser.add_argument("--generated", type=str, default=str(ROOT / "data/generated"))
    parser.add_argument("--manifest", type=str, default=str(ROOT / "data/manifest.json"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--allow-unverified", action="store_true",
                        help="Experimental use only: include unreviewed teacher records")
    parser.add_argument("--allow-evidence-mismatch", action="store_true",
                        help="Include records failing source-evidence exact-match checks")
    parser.add_argument("--no-source", action="store_true",
                        help="Exclude original source article text from output")
    args = parser.parse_args(argv)
    from pathlib import Path
    output = Path(args.output) if args.output else (
        ROOT / "data" / "combined" / (
            "finance_corpus_experimental.txt" if args.allow_unverified else "finance_corpus.txt"
        )
    )
    summary = export_corpus(
        Path(args.sources), Path(args.generated), Path(args.manifest), output,
        options=ExportOptions(
            allow_unverified=args.allow_unverified,
            allow_evidence_mismatch=args.allow_evidence_mismatch,
            include_source=not args.no_source,
        ),
    )
    print(f"Exported {summary.eligible_records} records from {summary.sources} sources")
    print(f"Skipped: review={summary.skipped_review}, evidence={summary.skipped_evidence}, "
          f"generation={summary.skipped_generation}")
    print(f"Corpus: {summary.output_path} ({summary.characters:,} characters)")
    if args.allow_unverified:
        logging.warning("EXPERIMENTAL CORPUS: unverified financial content included")
    return summary


if __name__ == "__main__":
    main()
