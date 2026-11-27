"""Deterministic checks; not semantic or mathematical fact checking."""


def has_exact_evidence(quote: str | None, source_text: str) -> bool:
    """Check a short quote occurs in the source after whitespace normalization."""
    normalize = lambda value: " ".join(value.casefold().split())
    return bool(quote and len(quote.split()) >= 3
                and normalize(quote) in normalize(source_text))


def validate_item(item: dict, content_type: str) -> None:
    """Enforce invariants beyond the JSON schema; fail closed."""
    status = item["generation_status"]
    if status == "skipped":
        if (item["grounding"] != "insufficient_support" or not item["skip_reason"]
                or any(item[k] is not None for k in
                       ("question", "answer", "formula_latex", "source_evidence"))):
            raise ValueError("Skipped items require reason and null content fields.")
        return
    if item["grounding"] == "insufficient_support" or item["skip_reason"] is not None:
        raise ValueError("Generated item has inconsistent grounding or skip_reason.")
    if not item["answer"] or not item["source_evidence"]:
        raise ValueError("Generated item requires answer and source evidence.")
    if content_type == "tutorial" and item["question"] is not None:
        raise ValueError("Tutorial must have question=null.")
    if content_type != "tutorial" and not item["question"]:
        raise ValueError("Q&A and misconception require a question.")


def validate_complete_dataset(
    records: dict[str, dict],
    expected_requests: list[tuple[str, str | None, str]],
    source: dict,
    model: str,
) -> None:
    """Verify completeness, content type, subtopic and provenance."""

    expected = {
        record_id: (content_type, subtopic)
        for content_type, subtopic, record_id in expected_requests
    }

    # Every planned record must exist, with no extra records.
    if (
        len(expected) != len(expected_requests)
        or set(records) != set(expected)
    ):
        raise ValueError(
            "Missing, duplicate, or unexpected record IDs"
        )

    for record_id, record in records.items():

        content_type, subtopic = expected[record_id]

        # Check that persisted records belong to the correct
        # request, source document, and teacher model.
        if (
            record.get("id") != record_id
            or record.get("content_type") != content_type
            or record.get("subtopic") != subtopic
            or record.get("source_id") != source["source_id"]
            or record.get("teacher_model") != model
        ):
            raise ValueError(
                f"Record identity/provenance mismatch: {record_id}"
            )

        # Revalidate the generated content.
        validate_item(record, content_type)
