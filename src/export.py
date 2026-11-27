"""Convert completed finance JSONL datasets and their sources into a TXT corpus.

No API calls, no modification of the underlying research artifacts. A temporary
file is atomically published only if the entire export succeeds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .manifest import load_manifest
from .source import hash_source, read_source
from .storage import atomic_write, load_existing_ids, sha256_file

LOG = logging.getLogger(__name__)
FORMATS = {"tutorial": "TUTORIAL", "qa": "QA", "misconception": "MISCONCEPTION"}


@dataclass(frozen=True)
class ExportOptions:
    allow_unverified: bool = False
    allow_evidence_mismatch: bool = False
    include_source: bool = True


@dataclass(frozen=True)
class ExportSummary:
    sources: int
    eligible_records: int
    skipped_review: int
    skipped_evidence: int
    skipped_generation: int
    characters: int
    output_path: Path


def record_to_training_text(record: dict) -> str:
    """Render one eligible record, excluding provenance and usage metadata."""
    content_type = record.get("content_type")
    if content_type not in FORMATS:
        raise ValueError(f"Unsupported content_type: {content_type!r}")
    answer = record.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Missing answer for record {record.get('id')}")
    label = FORMATS[content_type]
    fields = []
    topic = record.get("topic")
    subtopic = record.get("subtopic")
    if topic:
        fields.append(f"Topic: {str(topic).strip()}")
    if subtopic:
        fields.append(f"Subtopic: {str(subtopic).strip()}")
    question = record.get("question")
    if content_type in ("qa", "misconception"):
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing question for record {record.get('id')}")
        fields.append(f"Question: {question.strip()}")
    fields.append(f"Answer: {answer.strip()}")
    formula = record.get("formula_latex")
    if isinstance(formula, str) and formula.strip():
        fields.append(f"Formula: {formula.strip()}")
    return f"[START {label}]\n" + "\n\n".join(fields) + f"\n[END {label}]\n"


def source_to_training_text(source: dict) -> str:
    """Render the original reference content once per source document."""
    topic = source.get("topic", "").strip()
    body = source.get("text", "").strip()
    if not body:
        raise ValueError("Source text is empty")
    return f"[START SOURCE]\nTopic: {topic}\n\n{body}\n[END SOURCE]\n"


def _read_jsonl(path: Path) -> list[dict]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line: {path}:{line_no}")
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL at {path}:{line_no}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Non-object JSONL record: {path}:{line_no}")
            result.append(item)
    return result


def export_corpus(
    source_dir: Path,
    generated_dir: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    options: ExportOptions = ExportOptions(),
) -> ExportSummary:
    """Validate published artifacts, then atomically write selected corpus text.

    Published datasets must be complete and identical to the manifest's hashes.
    Review-gating applies to generated records. No source text is exported from
    a file unless it has at least one eligible generated educational record.
    """
    manifest = load_manifest(manifest_path)
    entries = manifest["sources"]
    if not entries:
        raise ValueError(f"No completed sources in manifest: {manifest_path}")
    chunks = []
    included_sources = eligible = skipped_review = skipped_evidence = skipped_generation = 0
    for source_name in sorted(entries):
        entry = entries[source_name]
        if entry.get("status") != "complete":
            LOG.info("Ignoring incomplete source %s", source_name)
            continue
        if Path(source_name).name != source_name or not source_name.endswith(".txt"):
            raise ValueError(f"Invalid manifest source name: {source_name!r}")
        source_path = source_dir / source_name
        out_name = entry.get("output_file")
        if out_name != Path(source_name).with_suffix(".jsonl").name:
            raise ValueError(f"Unexpected published filename for {source_name}: {out_name!r}")
        dataset_path = generated_dir / out_name
        if not source_path.is_file() or not dataset_path.is_file():
            raise FileNotFoundError(f"Missing source or output: {source_path}, {dataset_path}")
        if hash_source(source_path) != entry.get("source_sha256"):
            raise ValueError(f"Source changed since publication: {source_path}")
        if sha256_file(dataset_path) != entry.get("output_sha256"):
            raise ValueError(f"Published dataset modified: {dataset_path}")
        ids = load_existing_ids(dataset_path)
        expected = entry.get("expected_ids")
        if not isinstance(expected, list) or len(expected) != len(set(expected)) or ids != set(expected):
            raise ValueError(f"Published dataset is incomplete: {dataset_path}")
        selected = []
        for record in _read_jsonl(dataset_path):
            if record.get("generation_status") != "generated":
                skipped_generation += 1
                continue
            review_status = record.get("review_status")
            if review_status != "approved" and not (
                options.allow_unverified and review_status == "unverified"
            ):
                skipped_review += 1
                continue
            if record.get("source_evidence_exact_match") is False and not options.allow_evidence_mismatch:
                skipped_evidence += 1
                continue
            selected.append(record_to_training_text(record))
        if not selected:
            continue
        # Only parse and include source material when there is eligible output.
        source = read_source(source_path)
        doc = [f"[START DOCUMENT]\n"]
        if options.include_source:
            doc.append(source_to_training_text(source))
        doc.extend(selected)
        doc.append("[END DOCUMENT]\n")
        chunks.append("\n".join(doc))
        included_sources += 1
        eligible += len(selected)
    if eligible == 0:
        raise ValueError(
            "No eligible generated records. Approve records through review, "
            "or explicitly select experimental --allow-unverified."
        )
    corpus = "\n".join(chunks) + "\n"
    atomic_write(output_path, corpus.encode("utf-8"))
    return ExportSummary(
        sources=included_sources,
        eligible_records=eligible,
        skipped_review=skipped_review,
        skipped_evidence=skipped_evidence,
        skipped_generation=skipped_generation,
        characters=len(corpus),
        output_path=output_path,
    )
