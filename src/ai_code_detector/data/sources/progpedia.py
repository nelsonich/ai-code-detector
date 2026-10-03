"""PROGpedia: student submissions to introductory programming exercises (2003-2020).

Dataset: https://zenodo.org/records/7449056 (CC-BY-4.0).
Layout: ``<exercise>/<STATUS>/<author>_<attempt>/<file>.<ext>`` plus ``<exercise>/statement.md``.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ai_code_detector.data.schema import CodeRecord, Label, LabelStatus, SourceKind, TaskRecord
from ai_code_detector.data.sources.base import DataSource

EXTENSIONS = {".java": "java", ".py": "python", ".c": "c", ".cpp": "cpp"}
ACCEPTED = "ACCEPTED"


@dataclass(frozen=True)
class _Submission:
    exercise: str
    status: str
    author: str
    attempt: int
    path: Path

    @property
    def language(self) -> str:
        return EXTENSIONS[self.path.suffix]


class ProgpediaSource(DataSource):
    """Yield one submission per exercise, author and language.

    Students often submit many near-identical attempts; keeping all of them would
    over-weight persistent students. The latest accepted attempt is preferred,
    otherwise the latest attempt of any status. Code predates AI assistants, so
    labels are verified human.
    """

    name = "progpedia"

    def __init__(self, root: Path) -> None:
        """Remember the directory with the unpacked dataset."""
        self.root = root

    def load(self) -> Iterator[CodeRecord]:
        """Yield the selected submission of every exercise, author and language."""
        best: dict[tuple[str, str, str], _Submission] = {}
        for submission in self._scan():
            key = (submission.exercise, submission.author, submission.language)
            current = best.get(key)
            if current is None or self._rank(submission) > self._rank(current):
                best[key] = submission
        for key in sorted(best):
            yield self._to_record(best[key])

    def tasks(self) -> Iterator[TaskRecord]:
        """Yield every exercise with its Markdown statement; its first line is the title."""
        for path in sorted(self.root.glob("*/statement.md")):
            statement = path.read_text(encoding="utf-8")
            yield TaskRecord(
                task_id=self._task_id(path.parent.name),
                dataset=self.name,
                title=statement.splitlines()[0].strip() if statement else None,
                statement=statement,
                statement_format="markdown",
            )

    def _scan(self) -> Iterator[_Submission]:
        for path in self.root.glob("*/*/*/*"):
            if path.suffix not in EXTENSIONS:
                continue
            exercise, status, folder = path.parts[-4:-1]
            author, attempt = folder.split("_")
            yield _Submission(exercise, status, author, int(attempt), path)

    @staticmethod
    def _rank(submission: _Submission) -> tuple[bool, int]:
        return submission.status == ACCEPTED, submission.attempt

    def _task_id(self, exercise: str) -> str:
        return f"{self.name}:{exercise}"

    def _to_record(self, submission: _Submission) -> CodeRecord:
        return CodeRecord(
            record_id=(
                f"{self.name}:{submission.exercise}:{submission.author}_{submission.attempt}"
            ),
            code=submission.path.read_text(encoding="utf-8", errors="replace"),
            language=submission.language,
            label=Label.HUMAN,
            label_status=LabelStatus.VERIFIED,
            source=SourceKind.EXTERNAL,
            task_id=self._task_id(submission.exercise),
            dataset=self.name,
            author_id=f"{self.name}:{submission.author}",
            extra={"meta_status": submission.status, "meta_attempt": submission.attempt},
        )
