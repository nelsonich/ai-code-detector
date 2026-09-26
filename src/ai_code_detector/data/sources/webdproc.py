"""Data source for anonymised exports of the webdproc platform."""

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from ai_code_detector.data.schema import CodeRecord, Label, LabelStatus, SourceKind
from ai_code_detector.data.sources.base import DataSource


class WebdprocSource(DataSource):
    """Read ``*.json`` exports where each file holds a list of samples.

    Expected keys per sample: ``id``, ``code``, ``language``, ``source``, ``task_id``,
    ``author_id``, ``created_at``. All samples are student code written in 2023 or
    later, when AI assistants were available, so the human label is unverified.
    """

    name = "webdproc"

    def __init__(self, export_dir: Path) -> None:
        """Remember the directory that holds the exported JSON files."""
        self.export_dir = export_dir

    def load(self) -> Iterator[CodeRecord]:
        """Yield one record per exported sample."""
        for path in sorted(self.export_dir.glob("*.json")):
            with open(path, encoding="utf-8") as handle:
                samples = json.load(handle)
            for sample in samples:
                yield self._to_record(sample)

    def _to_record(self, sample: dict) -> CodeRecord:
        created = sample.get("created_at")
        return CodeRecord(
            record_id=f"{self.name}:{sample['source']}:{sample['id']}",
            code=sample["code"],
            language=sample["language"],
            label=Label.HUMAN,
            label_status=LabelStatus.UNVERIFIED,
            source=SourceKind(sample["source"]),
            task_id=f"{self.name}:{sample['task_id']}",
            dataset=self.name,
            author_id=sample.get("author_id"),
            created_at=datetime.fromisoformat(created) if created else None,
            extra={k: v for k, v in sample.items() if k.startswith("meta_")},
        )
