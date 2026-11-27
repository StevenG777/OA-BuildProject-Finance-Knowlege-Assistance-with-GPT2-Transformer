
import hashlib
from pathlib import Path
from urllib.parse import urlparse


def discover_source(source_dir: Path) -> list[Path]:
    """Discover .txt files directly inside the source directory."""
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Source directory not found: {source_dir}"
        )

    return sorted(
        path for path in source_dir.glob("*.txt")
        if path.is_file()
    )


def hash_source(path: Path) -> str:
    """Generate a SHA-256 fingerprint of the entire source file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_source(path: Path) -> dict:
    """Read source metadata, content, and optional subtopics."""
    raw = path.read_text(encoding="utf-8")
    header, separator, content = raw.partition("CONTENT:")

    metadata = {}
    for line in header.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip().upper()] = value.strip()

    url = metadata.get("URL", "")
    topic = metadata.get("TOPIC", "")

    # Read optional list of subtopics
    subtopics_raw = metadata.get("SUBTOPICS", "")

    subtopics = [
        value.strip()
        for value in subtopics_raw.split("|")
        if value.strip()
    ]

    # Reject duplicate subtopics
    if len(subtopics) != len(set(subtopics)):
        raise ValueError("Duplicate subtopics found.")

    parsed = urlparse(url)

    if (
        not separator
        or not content.strip()
        or content.strip().startswith("[Paste")
    ):
        raise ValueError("Add source article text after CONTENT:.")

    if parsed.scheme != "https" or not parsed.netloc or not topic:
        raise ValueError(
            "Provide a valid HTTPS URL and nonempty TOPIC."
        )

    body = content.strip()

    digest = hashlib.sha256(
        (url + "\n" + body).encode("utf-8")
    ).hexdigest()[:16]

    return {
        "source_id": digest,
        "url": url,
        "topic": topic,
        "subtopics": subtopics,
        "text": body,
    }
