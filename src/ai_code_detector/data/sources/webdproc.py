"""Data source for anonymised exports of the webdproc platform.

Export files (JSON lists, one object per row) in one directory:
``samples_lessons|challenges|exams.json``: student code, one row per non-empty language block;
``tasks_lessons|challenges|exams.json``: task texts; ``tasks_challenge_tests.json``: test cases.
"""

import json
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_code_detector.data.schema import CodeRecord, Label, LabelStatus, SourceKind, TaskRecord
from ai_code_detector.data.sources.base import DataSource

DATE_FORMATS = ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S")
# Challenge language modes that map to one language; "web" mixes HTML, CSS and JS.
CHALLENGE_LANGUAGES = {"javascript", "python", "cpp", "csharp", "php", "java"}
STATEMENT_LOCALES = ("en", "am", "ru")


def _clean(value: Any) -> Any:
    """Navicat exports NULL as the string 'None'; turn it and empty strings into None."""
    return None if value in (None, "", "None") else value


def _parse_date(value: str | None) -> datetime | None:
    if not _clean(value):
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(value)


class WebdprocSource(DataSource):
    """Read the webdproc export: student code and the tasks it solves.

    All samples are student code written in 2023 or later, when AI assistants were
    available, so the human label is unverified.
    """

    name = "webdproc"

    def __init__(self, export_dir: Path) -> None:
        """Remember the directory that holds the exported JSON files."""
        self.export_dir = export_dir

    def load(self) -> Iterator[CodeRecord]:
        """Yield one record per exported code sample."""
        for path in sorted(self.export_dir.glob("samples_*.json")):
            for sample in self._read(path):
                yield self._to_record(sample)

    def tasks(self) -> Iterator[TaskRecord]:
        """Yield lesson, challenge and exam tasks found in the export."""
        yield from (self._lesson_task(row) for row in self._read_optional("tasks_lessons"))
        tests = self._challenge_tests()
        for row in self._read_optional("tasks_challenges"):
            yield self._challenge_task(row, tests.get(row["task_id"], []))
        yield from (self._exam_task(row) for row in self._read_optional("tasks_exams"))

    @staticmethod
    def _read(path: Path) -> list[dict]:
        with open(path, encoding="utf-8") as handle:
            return [{k: _clean(v) for k, v in row.items()} for row in json.load(handle)]

    def _read_optional(self, stem: str) -> list[dict]:
        path = self.export_dir / f"{stem}.json"
        return self._read(path) if path.exists() else []

    def _task_id(self, raw_task_id: str) -> str:
        return f"{self.name}:{raw_task_id}"

    def _to_record(self, sample: dict) -> CodeRecord:
        return CodeRecord(
            record_id=f"{self.name}:{sample['source']}:{sample['id']}",
            code=sample["code"],
            language=sample["language"],
            label=Label.HUMAN,
            label_status=LabelStatus.UNVERIFIED,
            source=SourceKind(sample["source"]),
            task_id=self._task_id(sample["task_id"]),
            dataset=self.name,
            author_id=sample.get("author_id"),
            created_at=_parse_date(sample.get("created_at")),
            extra={k: v for k, v in sample.items() if k.startswith("meta_")},
        )

    def _lesson_task(self, row: dict) -> TaskRecord:
        return TaskRecord(
            task_id=self._task_id(row["task_id"]),
            dataset=self.name,
            title=row.get("title"),
            statement=row.get("description_html") or "",
            statement_format="html",
            extra={
                "course_id": row.get("course_id"),
                "course_title": row.get("course_title"),
                "order": row.get("order"),
            },
        )

    def _challenge_task(self, row: dict, tests: list[dict]) -> TaskRecord:
        locale = next((loc for loc in STATEMENT_LOCALES if row.get(f"description_{loc}")), None)
        mode = row.get("language_mode")
        return TaskRecord(
            task_id=self._task_id(row["task_id"]),
            dataset=self.name,
            title=row.get(f"title_{locale}") if locale else None,
            statement=(row.get(f"description_{locale}") or "") if locale else "",
            statement_format="html",
            language=mode if mode in CHALLENGE_LANGUAGES else None,
            extra={
                "locale": locale,
                "language_mode": mode,
                "difficulty": row.get("difficulty"),
                "execution_mode": row.get("execution_mode"),
                "evaluation_mode": row.get("evaluation_mode"),
                "function_entry_names": row.get("function_entry_names"),
                "tests": tests,
            },
        )

    def _exam_task(self, row: dict) -> TaskRecord:
        return TaskRecord(
            task_id=self._task_id(row["task_id"]),
            dataset=self.name,
            title=row.get("title"),
            statement=row.get("description_html") or "",
            statement_format="html",
        )

    def _challenge_tests(self) -> dict[str, list[dict]]:
        tests: dict[str, list[dict]] = defaultdict(list)
        rows = sorted(self._read_optional("tasks_challenge_tests"),
                      key=lambda r: int(r.get("sort_order") or 0))
        for row in rows:
            tests[row["task_id"]].append({
                "input": row.get("input_data") or "",
                "expected_output": row.get("expected_output") or "",
                "is_hidden": str(row.get("is_hidden")) == "1",
            })
        return tests
