"""Code generated for this project by AI models (see ``ai_code_detector.generation``)."""

import json
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

from ai_code_detector.data.schema import CodeRecord, Label, LabelStatus, SourceKind
from ai_code_detector.data.sources.base import DataSource


class GeneratedSource(DataSource):
    """Read ``<generator>.jsonl`` result files; one record per generated language block.

    The origin of every sample is known, so labels are verified AI. ``task_id`` is the
    id of the human task the solution was generated for.
    """

    name = "generated"

    def __init__(self, root: Path, exclude_task_datasets: Iterable[str] = ()) -> None:
        """Remember the result directory and the task datasets whose code is skipped."""
        self.root = root
        self.exclude_task_datasets = frozenset(exclude_task_datasets)

    def load(self) -> Iterator[CodeRecord]:
        """Yield one record per non-empty code block of every generated solution."""
        for path in sorted(self.root.glob("*.jsonl")):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield from self._to_records(json.loads(line))

    def _to_records(self, row: dict) -> Iterator[CodeRecord]:
        if row["task_id"].split(":")[0] in self.exclude_task_datasets:
            return
        for language, code in sorted(row["codes"].items()):
            if not code.strip():
                continue
            yield CodeRecord(
                record_id=f"{self.name}:{row['job_id']}:{language}",
                code=code,
                language=language,
                label=Label.AI,
                label_status=LabelStatus.VERIFIED,
                source=SourceKind.GENERATED,
                task_id=row["task_id"],
                dataset=self.name,
                origin=row["task_id"].split(":")[0],
                author_id=f"{self.name}:{row['generator']}",
                created_at=datetime.fromisoformat(row["created_at"]),
                generator=row["generator"],
                prompt_style=row["style"],
                extra={
                    "model": row["model"],
                    "served_model": row["served_model"],
                    "statement_truncated": row["statement_truncated"],
                },
            )
