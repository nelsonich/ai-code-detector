"""Cleaning of the unified table before feature extraction, with a report of every step."""

import html
import re
from difflib import SequenceMatcher

import pandas as pd

_WHITESPACE = re.compile(r"\s+")
_PRE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")

CATEGORY_COLUMNS = ("language", "label", "label_status", "source", "dataset",
                    "generator", "prompt_style")
# Human code has no generator or prompt style; "none" makes that explicit for grouping.
NOT_APPLICABLE = {"generator": "none", "prompt_style": "none"}


def _compact(code: str) -> str:
    return _WHITESPACE.sub("", code)


def lecture_examples(tasks: pd.DataFrame) -> dict[str, list[str]]:
    """Whitespace-free code examples (``<pre>`` blocks) of every lesson, keyed by task id."""
    examples: dict[str, list[str]] = {}
    lessons = tasks[tasks["task_id"].str.contains(":lesson:", regex=False)]
    for task_id, statement in zip(lessons["task_id"], lessons["statement"].fillna("")):
        blocks = [_compact(html.unescape(_TAG.sub("", body))) for body in _PRE.findall(statement)]
        if blocks := [block for block in blocks if block]:
            examples[task_id] = blocks
    return examples


class CodePreprocessor:
    """Clean the unified table step by step and record what every step changed.

    ``report`` holds, after ``run``: rows removed per step, missing values before and
    after, and the final class balance per language.
    """

    def __init__(self, min_code_chars: int, languages: list[str],
                 example_similarity: float = 0.9) -> None:
        """Store cleaning thresholds."""
        self.min_code_chars = min_code_chars
        self.languages = languages
        self.example_similarity = example_similarity
        self.report: dict = {}

    def run(self, frame: pd.DataFrame, tasks: pd.DataFrame | None = None) -> pd.DataFrame:
        """Apply all cleaning steps and return a new table; ``tasks`` enables lecture checks."""
        self.report = {"input_rows": len(frame), "missing_before": _missing(frame), "steps": []}
        cleaned = self.normalize_code(frame.copy())
        steps = [
            ("drop_empty_or_short", self.drop_empty),
            ("filter_languages", self.filter_languages),
            ("drop_exact_duplicates", self.drop_exact_duplicates),
        ]
        if tasks is not None:
            examples = lecture_examples(tasks)
            steps.append(("drop_lecture_copies",
                          lambda f: self.drop_lecture_copies(f, examples)))
        for name, step in steps:
            before = cleaned
            cleaned = step(cleaned)
            removed = before.drop(cleaned.index)
            self.report["steps"].append({
                "step": name,
                "removed": len(removed),
                "removed_by_label": {str(k): int(v) for k, v in
                                     removed["label"].value_counts().items()},
                "remaining": len(cleaned),
            })

        cleaned = self.fix_types(self.fill_missing(cleaned)).reset_index(drop=True)
        self.report["missing_after"] = _missing(cleaned)
        self.report["balance"] = {
            language: {str(label): int(count) for label, count in counts.items()}
            for language, counts in cleaned.groupby("language", observed=True)["label"]
            .value_counts().unstack(fill_value=0).iterrows()
        }
        return cleaned

    @staticmethod
    def normalize_code(frame: pd.DataFrame) -> pd.DataFrame:
        """Unify line endings and drop a byte order mark.

        Sources differ here (a third of PROGpedia uses CRLF), which would otherwise
        surface as a feature of the source rather than of the author.
        """
        out = frame.copy()
        out["code"] = (out["code"].fillna("").str.replace("\r\n", "\n", regex=False)
                       .str.replace("\r", "\n", regex=False).str.lstrip("﻿"))
        return out

    def drop_empty(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Drop samples with no code or with fewer characters than the threshold."""
        return frame[frame["code"].str.strip().str.len() >= self.min_code_chars]

    def filter_languages(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep only the languages listed in the configuration."""
        return frame[frame["language"].isin(self.languages)]

    @staticmethod
    def drop_exact_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
        """Drop identical code (ignoring whitespace) for the same task, keeping the first."""
        key = frame["code"].map(_compact)
        return frame[~pd.concat([frame["task_id"], key], axis=1).duplicated()]

    def drop_lecture_copies(self, frame: pd.DataFrame,
                            examples: dict[str, list[str]]) -> pd.DataFrame:
        """Drop code that copies a code example from its own lesson's lecture.

        Copied examples say nothing about who wrote the code, and the examples may
        themselves be AI-generated. Applied to both classes: generators saw the lecture too.
        """
        def copied(task_id: str, code: str) -> bool:
            candidates = examples.get(task_id)
            if not candidates:
                return False
            compact = _compact(code)
            return any(self._similar(compact, example) for example in candidates)

        mask = [copied(t, c) for t, c in zip(frame["task_id"], frame["code"])]
        return frame[~pd.Series(mask, index=frame.index)]

    def _similar(self, a: str, b: str) -> bool:
        if a == b:
            return True
        matcher = SequenceMatcher(None, a, b, autojunk=False)
        return (matcher.real_quick_ratio() >= self.example_similarity
                and matcher.quick_ratio() >= self.example_similarity
                and matcher.ratio() >= self.example_similarity)

    @staticmethod
    def fill_missing(frame: pd.DataFrame) -> pd.DataFrame:
        """Make not-applicable values explicit; unknown dates stay missing (not imputed)."""
        out = frame.copy()
        for column, value in NOT_APPLICABLE.items():
            out[column] = out[column].fillna(value)
        out["author_id"] = out["author_id"].fillna("unknown")
        return out

    @staticmethod
    def fix_types(frame: pd.DataFrame) -> pd.DataFrame:
        """Set categorical and datetime dtypes."""
        out = frame.copy()
        for column in CATEGORY_COLUMNS:
            out[column] = out[column].astype("category")
        out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce", utc=True)
        return out


def _missing(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame.isna().sum()
    return {column: int(n) for column, n in counts.items() if n}
