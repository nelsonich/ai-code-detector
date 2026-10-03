"""IBM Project CodeNet: online judge submissions (AIZU, AtCoder) collected up to 2021.

Dataset: https://github.com/IBM/Project_CodeNet (code and metadata; CDLA-Permissive-2.0).
Layout: ``data/<problem>/<Language>/<submission>.<ext>``, ``metadata/<problem>.csv``,
``metadata/problem_list.csv``, ``problem_descriptions/<problem>.html``.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ai_code_detector.data.schema import CodeRecord, Label, LabelStatus, SourceKind, TaskRecord
from ai_code_detector.data.sources.base import DataSource

LANGUAGES = {"C#": "csharp", "PHP": "php", "JavaScript": "javascript"}
ACCEPTED = "Accepted"
META_COLUMNS = ["submission_id", "problem_id", "user_id", "date", "language",
                "filename_ext", "status"]


class CodenetSource(DataSource):
    """Yield a balanced, reproducible sample of accepted submissions.

    CodeNet holds millions of submissions, so it is sampled: one submission per
    author and problem, at most ``max_per_problem`` per problem and language (many
    different problems of varying difficulty), up to ``per_language`` in total.
    Only problems with a description are used, so AI solutions can be generated for
    the same tasks. All code predates AI assistants, so labels are verified human.
    """

    name = "codenet"

    def __init__(self, root: Path, per_language: int = 300, max_per_problem: int = 3,
                 seed: int = 42) -> None:
        """Store the dataset root and sampling parameters."""
        self.root = root
        self.per_language = per_language
        self.max_per_problem = max_per_problem
        self.seed = seed

    def load(self) -> Iterator[CodeRecord]:
        """Yield the sampled submissions of every configured language."""
        problems = pd.read_csv(self.root / "metadata" / "problem_list.csv").set_index("id")
        selected = self.sample()
        missing = selected.loc[~selected["path"].map(Path.exists), "path"]
        if not missing.empty:
            raise FileNotFoundError(
                f"{len(missing)} sampled CodeNet files are missing, e.g. {missing.iloc[0]}. "
                "Run scripts/download_data.sh to fetch the full dataset."
            )
        for row in selected.itertuples(index=False):
            yield self._to_record(row, problems)

    def sample(self) -> pd.DataFrame:
        """Return metadata rows of the selected submissions.

        Selection uses metadata only, never the files present on disk, so every
        machine with the same metadata gets the same sample.
        """
        meta = self._candidates()
        meta = meta.sample(frac=1, random_state=self.seed)
        meta = meta.drop_duplicates(["problem_id", "user_id", "language"])
        meta = meta.groupby(["language", "problem_id"]).head(self.max_per_problem)
        return meta.groupby("language").head(self.per_language).reset_index(drop=True)

    def tasks(self) -> Iterator[TaskRecord]:
        """Yield the problems used by the sample, with their HTML descriptions."""
        problems = pd.read_csv(self.root / "metadata" / "problem_list.csv").set_index("id")
        for problem_id in sorted(self.sample()["problem_id"].unique()):
            problem = problems.loc[problem_id]
            path = self.root / "problem_descriptions" / f"{problem_id}.html"
            yield TaskRecord(
                task_id=self._task_id(problem_id),
                dataset=self.name,
                title=problem["name"],
                statement=path.read_text(encoding="utf-8", errors="replace"),
                statement_format="html",
                extra={"judge": problem["dataset"]},
            )

    def _candidates(self) -> pd.DataFrame:
        frames = []
        for description in sorted((self.root / "problem_descriptions").glob("*.html")):
            meta_path = self.root / "metadata" / f"{description.stem}.csv"
            if not meta_path.exists():
                continue
            meta = pd.read_csv(meta_path, usecols=META_COLUMNS)
            meta = meta[meta["language"].isin(LANGUAGES) & (meta["status"] == ACCEPTED)]
            meta["path"] = [
                self.root / "data" / pid / lang / f"{sid}.{ext}"
                for pid, sid, lang, ext in zip(meta["problem_id"], meta["submission_id"],
                                               meta["language"], meta["filename_ext"])
            ]
            frames.append(meta)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _task_id(self, problem_id: str) -> str:
        return f"{self.name}:{problem_id}"

    def _to_record(self, row, problems: pd.DataFrame) -> CodeRecord:
        problem = problems.loc[row.problem_id]
        return CodeRecord(
            record_id=f"{self.name}:{row.submission_id}",
            code=row.path.read_text(encoding="utf-8", errors="replace"),
            language=LANGUAGES[row.language],
            label=Label.HUMAN,
            label_status=LabelStatus.VERIFIED,
            source=SourceKind.EXTERNAL,
            task_id=self._task_id(row.problem_id),
            dataset=self.name,
            author_id=f"{self.name}:{row.user_id}",
            created_at=datetime.fromtimestamp(row.date, tz=UTC),
            extra={
                "meta_judge": problem["dataset"],
                "meta_rating": None if pd.isna(problem["rating"]) else int(problem["rating"]),
            },
        )
