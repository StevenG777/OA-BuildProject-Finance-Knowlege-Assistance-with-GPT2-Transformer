
"""Application configuration. No API requests occur on import."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # LLM teacher model API
    model: str = "gpt-6-luna"
    max_output_tokens: int = 1800

    # Source discovery and published datasets
    source_dir: Path = ROOT / "data/sources"
    output_dir: Path = ROOT / "data/generated"

    # Private staging and source-processing history
    work_dir: Path = ROOT / "data/.checkpoints"
    manifest_path: Path = ROOT / "data/manifest.json"

    # Generation control
    override: bool = False
    formats: tuple[str, ...] = (
        "tutorial",
        "qa",
        "misconception",
    )


def load_settings() -> Settings:
    """Allow model selection without editing source code."""
    return Settings(
        model=os.getenv("OPENAI_MODEL", "gpt-6-luna")
    )
