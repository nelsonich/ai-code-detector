"""Cleaning of the unified table before feature extraction."""

import re

import pandas as pd

_WHITESPACE = re.compile(r"\s+")


class CodePreprocessor:
    """Remove unusable samples and derive normalised views of the code."""

    def __init__(self, min_code_chars: int, languages: list[str]) -> None:
        """Store cleaning thresholds."""
        self.min_code_chars = min_code_chars
        self.languages = languages

    def run(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps and return a new table."""
        cleaned = frame.copy()
        cleaned = self.drop_empty(cleaned)
        cleaned = self.filter_languages(cleaned)
        cleaned = self.drop_exact_duplicates(cleaned)
        cleaned = self.fix_types(cleaned)
        return cleaned.reset_index(drop=True)

    def drop_empty(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Drop samples with no code or with fewer characters than the threshold."""
        code = frame["code"].fillna("").str.strip()
        return frame[code.str.len() >= self.min_code_chars]

    def filter_languages(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep only the languages listed in the configuration."""
        return frame[frame["language"].isin(self.languages)]

    @staticmethod
    def drop_exact_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
        """Drop identical code submitted for the same task, keeping the first one."""
        key = frame["code"].map(lambda c: _WHITESPACE.sub("", c))
        return frame[~pd.concat([frame["task_id"], key], axis=1).duplicated()]

    @staticmethod
    def fix_types(frame: pd.DataFrame) -> pd.DataFrame:
        """Set categorical and datetime dtypes."""
        out = frame.copy()
        for column in ("language", "label", "source", "dataset"):
            out[column] = out[column].astype("category")
        out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
        return out
