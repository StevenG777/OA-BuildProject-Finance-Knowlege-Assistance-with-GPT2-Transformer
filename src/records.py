"""Generation request planning and dataset record construction."""

import hashlib
from .validation import has_exact_evidence



def build_pending_requests(
    source: dict,
    formats: tuple[str, ...],
    model: str,
    saved_ids: set[str],
) -> list[tuple[str, str | None, str]]:
    """Build requests that have not already been saved."""

    subtopics = source["subtopics"]

    if not isinstance(subtopics, list) or any(
        not isinstance(s, str) or not s.strip()
        for s in subtopics
    ):
        raise ValueError(
            "subtopics must be a list of nonempty strings."
        )

    pending = []
    seen_ids = set(saved_ids)

    for subtopic in subtopics or [None]:

        subtopic_suffix = (
            hashlib.sha256(
                subtopic.encode("utf-8")
            ).hexdigest()[:16]
            if subtopic else None
        )

        for content_type in formats:

            # Preserve your existing record-ID convention.
            record_id = (
                f"{source['source_id']}:{content_type}:{model}:"
            )

            if subtopic_suffix:
                record_id += f":subtopic:{subtopic_suffix}"

            # Prevent both persisted and planned duplicates.
            if record_id not in seen_ids:
                pending.append(
                    (content_type, subtopic, record_id)
                )
                seen_ids.add(record_id)

    return pending


def build_record(
    source: dict, 
    content_type: str, 
    item: dict,
    record_id: str, 
    model: str, 
    usage: dict
) -> dict:
    """Enrich teacher output with provenance and explicitly unverified status."""

    generated = item["generation_status"] == "generated"
    return {
        "id": record_id, 
        "source_id": source["source_id"],
        "source_url": source["url"], 
        "topic": source["topic"],
        "subtopic": source.get("subtopic"),
        "content_type": content_type, 
        "teacher_model": model,
        **item,
        "source_evidence_exact_match": (
            has_exact_evidence(item["source_evidence"], source["text"])
            if generated else None
        ),
        "review_status": "unverified" if generated else "skipped",
        "api_usage": usage,
    }
