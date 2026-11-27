
"""Persistent source-level state and generation decisions."""

import json
from pathlib import Path

from .storage import (
    atomic_write,
    load_existing_ids,
    sha256_file,
)


MANIFEST_SCHEMA_VERSION = 1


def load_manifest(path: Path) -> dict:
    """Load the source-processing manifest."""

    if not path.exists():
        return {
            "version": MANIFEST_SCHEMA_VERSION,
            "sources": {},
        }

    manifest = json.loads(
        path.read_text(encoding="utf-8")
    )

    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != MANIFEST_SCHEMA_VERSION
        or not isinstance(manifest.get("sources"), dict)
    ):
        raise ValueError(
            f"Unsupported manifest: {path}"
        )

    return manifest


def save_manifest(
    path: Path,
    manifest: dict,
) -> None:
    """Atomically persist the source-processing manifest."""

    payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ) + "\n"
    ).encode("utf-8")

    atomic_write(path, payload)


def generation_config(settings) -> dict:
    """Capture settings that affect dataset generation."""

    return {
        "model": settings.model,
        "formats": list(settings.formats),
        "max_output_tokens": settings.max_output_tokens,
    }


def decide_source(
    entry: dict | None,
    source_hash: str,
    config: dict,
    output_path: Path,
    override: bool,
) -> str:
    # An explicit override always starts a fresh generation.
    if override:
        return "start"

    if entry is None:
        return "start"

    status = entry.get("status")

    # Previously interrupted attempts remain resumable.
    if status == "in_progress":
        return "resume"

    if status == "complete":
        expected = set(entry["expected_ids"])

        # Manifest status alone does not prove completeness.
        if (
            not output_path.is_file()
            or sha256_file(output_path)
                != entry.get("output_sha256")
            or load_existing_ids(output_path) != expected
        ):
            raise RuntimeError(
                f"Published dataset is missing/modified: {output_path}"
            )

        return "skip"

    raise ValueError(
        f"Unknown manifest status: {status}"
    )


def mark_in_progress(
    manifest: dict,
    name: str,
    source_hash: str,
    config: dict,
    output_name: str,
    expected_ids: list[str],
) -> None:
    """Record a new active generation attempt."""

    manifest["sources"][name] = {
        "source_sha256": source_hash,
        "generation_config": config,
        "output_file": output_name,
        "expected_ids": expected_ids,
        "status": "in_progress",
    }


def verify_resume_plan(
    entry: dict,
    expected_ids: list[str],
) -> None:
    """Prevent resuming incompatible record-generation plans."""

    if entry.get("expected_ids") != expected_ids:
        raise RuntimeError(
            "Generation plan changed. "
            "Set override=True after increasing pipeline_revision."
        )


def mark_complete(
    manifest: dict,
    name: str,
    output_digest: str,
) -> None:
    """Mark a dataset as committed after successful publication."""

    entry = manifest["sources"][name]

    entry["status"] = "complete"
    entry["output_sha256"] = output_digest
