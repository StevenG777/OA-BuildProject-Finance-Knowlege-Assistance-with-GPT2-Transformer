
"""Atomic record checkpoints and complete JSONL publication."""

import hashlib
import json
import os
import shutil
import tempfile

from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Calculate a persisted file's SHA-256 fingerprint."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    """Replace one file atomically with best-effort durability."""

    path.parent.mkdir(parents=True, exist_ok=True)

    fd, name = tempfile.mkstemp(
        prefix=".tmp-",
        dir=path.parent,
    )

    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(name, path)

    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_existing_ids(path: Path) -> set[str]:
    """Read JSONL IDs and reject invalid or duplicate records."""

    if not path.exists():
        return set()

    ids: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):

            if not line.strip():
                raise ValueError(
                    f"Empty JSONL line {number} in {path}"
                )

            try:
                record = json.loads(line)
                record_id = record["id"]

                if not isinstance(record_id, str) or not record_id:
                    raise ValueError("Invalid ID")

            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Malformed JSONL line {number} in {path}"
                ) from exc

            if record_id in ids:
                raise ValueError(
                    f"Duplicate record ID: {record_id}"
                )

            ids.add(record_id)

    return ids


def checkpoint_path(
    work_dir: Path,
    record_id: str,
) -> Path:
    """Derive a safe filename from the deterministic record ID."""

    name = hashlib.sha256(
        record_id.encode("utf-8")
    ).hexdigest()

    return work_dir / f"{name}.json"


def save_checkpoint(
    work_dir: Path,
    record: dict,
) -> None:
    """Persist one generated record atomically."""

    record_id = record["id"]

    if not isinstance(record_id, str) or not record_id:
        raise ValueError(
            "Record ID must be a nonempty string"
        )

    target = checkpoint_path(work_dir, record_id)

    if target.exists():
        raise ValueError(
            f"Checkpoint already exists: {record_id}"
        )

    payload = json.dumps(
        record,
        ensure_ascii=False,
    ).encode("utf-8")

    atomic_write(target, payload)


def load_checkpoints(work_dir: Path) -> dict[str, dict]:
    """Recover completed records from the staging workspace."""

    records: dict[str, dict] = {}

    if not work_dir.exists():
        return records

    for path in sorted(work_dir.glob("*.json")):

        record = json.loads(
            path.read_text(encoding="utf-8")
        )

        if not isinstance(record, dict):
            raise ValueError(
                f"Invalid checkpoint: {path}"
            )

        record_id = record.get("id")

        if not isinstance(record_id, str) or not record_id:
            raise ValueError(
                f"Checkpoint missing ID: {path}"
            )

        if path != checkpoint_path(work_dir, record_id):
            raise ValueError(
                f"Checkpoint filename does not match record ID: {path}"
            )

        if record_id in records:
            raise ValueError(
                f"Duplicate checkpoint ID: {record_id}"
            )

        records[record_id] = record

    return records


def reset_workspace(work_dir: Path) -> None:
    """Discard staging only when starting a fresh generation."""

    if work_dir.exists():
        shutil.rmtree(work_dir)

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def publish_dataset(
    work_dir: Path,
    output_path: Path,
    ordered_ids: list[str],
) -> str:
    """Publish complete checkpoints as one JSONL dataset."""

    records = load_checkpoints(work_dir)

    if (
        len(ordered_ids) != len(set(ordered_ids))
        or set(records) != set(ordered_ids)
    ):
        raise ValueError(
            "Checkpoint IDs do not match the expected dataset"
        )

    # Assemble the completed dataset.
    payload = "".join(
        json.dumps(
            records[record_id],
            ensure_ascii=False,
        ) + "\n"
        for record_id in ordered_ids
    ).encode("utf-8")

    digest = hashlib.sha256(payload).hexdigest()

    # Recovery: the output may already have been published
    # before an interrupted manifest update.
    if (
        output_path.exists()
        and sha256_file(output_path) == digest
    ):
        return digest

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Archive an existing published dataset.
    if output_path.exists():

        stamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%S%fZ")

        backup_dir = output_path.parent.parent / ".bak"
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_path = backup_dir / (
            output_path.name + f".{stamp}.bak"
        )

        shutil.copy2(
            output_path,
            backup_path,
        )

    # Only publish after all expected records exist.
    atomic_write(output_path, payload)

    return digest
