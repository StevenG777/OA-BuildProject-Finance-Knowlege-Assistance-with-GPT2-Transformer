
"""High-level orchestration for generating complete Finance 101 datasets."""

import logging

from src.api import (
    create_client,
    generate_item,
)

from src.config import load_settings

from src.manifest import (
    decide_source,
    generation_config,
    load_manifest,
    mark_complete,
    mark_in_progress,
    save_manifest,
    verify_resume_plan,
)

from src.records import (
    build_pending_requests,
    build_record,
)

from src.source import (
    discover_source,
    hash_source,
    read_source,
)

from src.storage import (
    load_checkpoints,
    publish_dataset,
    reset_workspace,
    save_checkpoint,
)

from src.validation import (
    validate_complete_dataset,
    validate_item,
)


logger = logging.getLogger(__name__)


def main() -> None:
    """Orchestrate discovery, generation, recovery, and publication."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Step 1: Load configuration and persistent state.
    settings = load_settings()

    manifest = load_manifest(
        settings.manifest_path
    )

    config = generation_config(settings)

    # Step 2: Discover available source documents.
    source_paths = discover_source(
        settings.source_dir
    )

    client = None

    # Step 3: Process each source independently.
    for source_path in source_paths:

        name = source_path.name

        output_path = (
            settings.output_dir
            / f"{source_path.stem}.jsonl"
        )

        work_dir = (
            settings.work_dir
            / source_path.stem
        )

        source_hash = hash_source(source_path)

        entry = manifest["sources"].get(name)

        # Step 4: Determine whether to start, resume, or skip.
        decision = decide_source(
            entry,
            source_hash,
            config,
            output_path,
            settings.override,
        )

        if decision == "skip":
            logger.info(
                "Skipped complete dataset: %s",
                name,
            )
            continue

        # Step 5: Read and plan generation requests.
        source = read_source(source_path)

        expected_requests = build_pending_requests(
            source,
            settings.formats,
            settings.model,
            set(),
        )

        expected_ids = [
            record_id
            for _, _, record_id in expected_requests
        ]

        # Step 6: Initialize a new workspace or resume one.
        if decision == "start":

            reset_workspace(work_dir)

            mark_in_progress(
                manifest,
                name,
                source_hash,
                config,
                output_path.name,
                expected_ids,
            )

            save_manifest(
                settings.manifest_path,
                manifest,
            )

        else:
            verify_resume_plan(
                entry,
                expected_ids,
            )

        # Step 7: Recover already generated records.
        saved_ids = set(
            load_checkpoints(work_dir)
        )

        pending = build_pending_requests(
            source,
            settings.formats,
            settings.model,
            saved_ids,
        )

        logger.info(
            "%s: %d/%d records remaining",
            name,
            len(pending),
            len(expected_ids),
        )

        # Step 8: Generate and checkpoint missing records.
        for content_type, subtopic, record_id in pending:

            if client is None:
                client = create_client()

            request_source = {
                **source,
                "subtopic": subtopic,
            }

            item, usage = generate_item(
                client,
                request_source,
                content_type,
                settings.model,
                settings.max_output_tokens,
            )

            validate_item(
                item,
                content_type,
            )

            record = build_record(
                request_source,
                content_type,
                item,
                record_id,
                settings.model,
                usage,
            )

            save_checkpoint(
                work_dir,
                record,
            )

            logger.info(
                "Checkpoint saved: %s / %s",
                name,
                content_type,
            )

        # Step 9: Verify the entire dataset.
        records = load_checkpoints(work_dir)

        validate_complete_dataset(
            records,
            expected_requests,
            source,
            settings.model,
        )

        # Make sure the source did not change during generation.
        if hash_source(source_path) != source_hash:
            raise RuntimeError(
                f"Source changed during generation: {name}"
            )

        # Step 10: Atomically publish the complete dataset.
        output_digest = publish_dataset(
            work_dir,
            output_path,
            expected_ids,
        )

        # Step 11: Persist successful completion.
        mark_complete(
            manifest,
            name,
            output_digest,
        )

        save_manifest(
            settings.manifest_path,
            manifest,
        )

        logger.info(
            "Published complete dataset: %s",
            output_path,
        )

    if not source_paths:
        logger.info(
            "No .txt sources found in %s",
            settings.source_dir,
        )


if __name__ == "__main__":
    main()
