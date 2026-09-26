"""Grouping, statistical summaries and correlation analysis of code features."""

import pandas as pd

from ai_code_detector.data.schema import RECORD_COLUMNS


class CodeAnalyzer:
    """Answer analytical questions on a feature table."""

    def __init__(self, frame: pd.DataFrame) -> None:
        """Keep the feature table and detect which columns are features."""
        self.frame = frame
        self.feature_columns = [c for c in frame.columns if c not in RECORD_COLUMNS]

    def summary_by_label(self) -> pd.DataFrame:
        """Descriptive statistics of every feature per label."""
        return self.frame.groupby("label", observed=True)[self.feature_columns].describe().T

    def group_means(self, by: list[str]) -> pd.DataFrame:
        """Mean of every feature grouped by the given columns."""
        return self.frame.groupby(by, observed=True)[self.feature_columns].mean()

    def label_correlation(self) -> pd.Series:
        """Point-biserial correlation of each feature with the ``ai`` label."""
        target = (self.frame["label"] == "ai").astype(float)
        return self.frame[self.feature_columns].corrwith(target).sort_values(ascending=False)

    def feature_correlation(self) -> pd.DataFrame:
        """Pairwise correlation matrix between features."""
        return self.frame[self.feature_columns].corr()
