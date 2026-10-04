"""DroidCollection: human and AI code from several public sources (Orel et al., EMNLP 2025).

Dataset: https://huggingface.co/datasets/project-droid/DroidCollection (the test split is
enough for analysis). Every language has human code from more than one population
(competitive programming sites and GitHub), each paired with AI code generated for the
same kind of task. That makes it the control our own data lacks: a feature is a sign of
AI only if it separates AI from human code inside every population.
"""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from ai_code_detector.data.schema import CodeRecord, Label, LabelStatus, SourceKind
from ai_code_detector.data.sources.base import DataSource

LANGUAGES = {"C": "c", "C++": "cpp", "C#": "csharp", "Java": "java",
             "JavaScript": "javascript", "PHP": "php", "Python": "python"}
LABELS = {"HUMAN_GENERATED": Label.HUMAN, "MACHINE_GENERATED": Label.AI}
COMPETITIVE = {"ATCODER", "CODEFORCES", "LEETCODE", "TACO"}
GITHUB_PREFIXES = ("STARCODER_DATA", "THEVAULT")
COLUMNS = ["Code", "Generator", "Generation_Mode", "Source", "Language", "Label"]


def origin_of(source: str) -> str | None:
    """Map a Droid source to the population it represents, or None to skip it."""
    if source in COMPETITIVE:
        return "droid-cp"
    if source.startswith(GITHUB_PREFIXES):
        return "droid-github"
    return None


class DroidSource(DataSource):
    """Yield a reproducible sample of human and AI code per language and origin.

    Machine-refined and adversarial samples are skipped for now: they answer a
    different question (mixed authorship, evasion) than the feature analysis.
    Human code was collected before it was used to prompt the generators, so both
    labels are treated as verified.
    """

    name = "droid"

    def __init__(self, root: Path, per_group: int = 300, seed: int = 42) -> None:
        """Store the directory with Droid parquet files and the sample size per group."""
        self.root = root
        self.per_group = per_group
        self.seed = seed

    def load(self) -> Iterator[CodeRecord]:
        """Yield up to ``per_group`` records per language, origin and label."""
        for row in self.sample().itertuples():
            yield self._to_record(row)

    def sample(self) -> pd.DataFrame:
        """Return the selected rows of all parquet files in ``root``."""
        paths = sorted(self.root.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no Droid parquet files in {self.root}; "
                                    "run scripts/download_data.sh droid")
        frame = pd.concat([pd.read_parquet(p, columns=COLUMNS) for p in paths],
                          ignore_index=True)
        frame["origin"] = frame["Source"].map(origin_of)
        frame = frame[frame["Label"].isin(LABELS) & frame["Language"].isin(LANGUAGES)
                      & frame["origin"].notna()]
        frame = frame.sample(frac=1, random_state=self.seed)
        return frame.groupby(["Language", "origin", "Label"]).head(self.per_group)

    def _to_record(self, row) -> CodeRecord:
        label = LABELS[row.Label]
        return CodeRecord(
            record_id=f"{self.name}:{row.Index}",
            code=row.Code,
            language=LANGUAGES[row.Language],
            label=label,
            label_status=LabelStatus.VERIFIED,
            source=SourceKind.EXTERNAL if label == Label.HUMAN else SourceKind.GENERATED,
            # Droid has no task ids, so each sample is its own task for grouped splits.
            task_id=f"{self.name}:{row.Index}",
            dataset=self.name,
            origin=row.origin,
            generator=row.Generator if label == Label.AI else None,
            prompt_style=row.Generation_Mode.lower() if label == Label.AI else None,
            extra={"droid_source": row.Source},
        )
