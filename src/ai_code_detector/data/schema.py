"""Unified record format shared by every data source, the analysis and the future model."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

import pandas as pd


class Label(StrEnum):
    """Origin class of a code sample."""

    STUDENT = "student"
    AI = "ai"


class SourceKind(StrEnum):
    """Where a code sample comes from."""

    LESSON = "lesson"
    CHALLENGE = "challenge"
    EXAM = "exam"
    GENERATED = "generated"
    EXTERNAL = "external"


@dataclass
class CodeRecord:
    """One code sample with everything needed for analysis and training.

    ``task_id`` links a sample to the task it solves. It is required for grouping
    train/test splits by task, so a model never sees solutions of the same task
    in both splits.
    """

    record_id: str
    code: str
    language: str
    label: Label
    source: SourceKind
    task_id: str
    dataset: str
    author_id: str | None = None
    created_at: datetime | None = None
    generator: str | None = None
    prompt_style: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


RECORD_COLUMNS = [f.name for f in CodeRecord.__dataclass_fields__.values()]


def records_to_frame(records: list[CodeRecord]) -> pd.DataFrame:
    """Convert records into a DataFrame with enum values stored as plain strings."""
    rows = [asdict(r) | {"label": str(r.label), "source": str(r.source)} for r in records]
    return pd.DataFrame(rows, columns=RECORD_COLUMNS)
