"""Statistical summaries, group comparisons, correlation and PCA of code features."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ai_code_detector.data.schema import RECORD_COLUMNS

AI = "ai"
# Groups of the human class by how much its label can be trusted, plus the AI class.
GROUPS = {
    "human (verified)": ("human", "verified"),
    "human (unverified)": ("human", "unverified"),
    "ai": ("ai", "verified"),
}


class CodeAnalyzer:
    """Answer the analytical questions of the project on a feature table.

    Class comparisons use verified labels only (pre-AI human code against generated
    code); unverified human code is compared separately, as its label may be wrong.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        """Keep the feature table and detect which columns are features."""
        self.frame = frame
        self.feature_columns = [c for c in frame.columns if c not in RECORD_COLUMNS]

    @property
    def verified(self) -> pd.DataFrame:
        """Rows whose label is trusted."""
        return self.frame[self.frame["label_status"] == "verified"]

    def summary_by_label(self) -> pd.DataFrame:
        """Descriptive statistics of every feature per class (verified labels)."""
        stats = self.verified.groupby("label", observed=True)[self.feature_columns].describe()
        return stats.stack(level=0, future_stack=True).swaplevel().sort_index()

    def class_separation(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        """How well each feature alone separates AI from human code.

        ``auc`` is the probability that a random AI sample has a higher value than a
        random human one (0.5: no separation; far from 0.5 in either direction: strong).
        ``separation`` is ``|auc - 0.5| * 2`` for ranking; ``cohens_d`` is the effect
        size; ``p_value`` comes from the Mann-Whitney U test.
        """
        data = self.verified if frame is None else frame
        is_ai = (data["label"] == AI).to_numpy()
        rows = []
        for feature in self.feature_columns:
            values = data[feature].to_numpy(dtype=float)
            ai, human = values[is_ai], values[~is_ai]
            auc = roc_auc_score(is_ai, values)
            pooled = np.sqrt((ai.var(ddof=1) + human.var(ddof=1)) / 2)
            rows.append({
                "feature": feature,
                "median_ai": float(np.median(ai)),
                "median_human": float(np.median(human)),
                "auc": auc,
                "separation": abs(auc - 0.5) * 2,
                "cohens_d": (ai.mean() - human.mean()) / pooled if pooled else 0.0,
                "p_value": mannwhitneyu(ai, human).pvalue,
            })
        return pd.DataFrame(rows).set_index("feature").sort_values("separation", ascending=False)

    def separation_by(self, column: str, min_per_class: int = 30) -> pd.DataFrame:
        """Single-feature separation within each group of ``column`` (e.g. language).

        Shows whether a difference holds inside every group or only appears because
        groups have different shares of AI code. Groups with fewer than
        ``min_per_class`` samples of either class are skipped.
        """
        result = {}
        for group, data in self.verified.groupby(column, observed=True):
            n_ai = int((data["label"] == AI).sum())
            if min(n_ai, len(data) - n_ai) >= min_per_class:
                result[group] = self.class_separation(data)["auc"]
        return pd.DataFrame(result)

    def group_profile(self, by: list[str], statistic: str = "median") -> pd.DataFrame:
        """Compute a statistic of every feature per group, with the group sizes."""
        grouped = self.frame.groupby(by, observed=True)
        return grouped[self.feature_columns].agg(statistic).assign(n=grouped.size())

    def trust_groups(self) -> pd.DataFrame:
        """Feature medians of verified human, unverified human and AI code side by side.

        Answers where unverified student code lies: closer to pre-AI human code or to AI.
        """
        columns = {}
        for name, (label, status) in GROUPS.items():
            data = self.frame[(self.frame["label"] == label)
                              & (self.frame["label_status"] == status)]
            columns[name] = data[self.feature_columns].median()
        return pd.DataFrame(columns)

    def label_correlation(self) -> pd.Series:
        """Spearman correlation of each feature with the AI label (verified labels)."""
        data = self.verified
        target = (data["label"] == AI).astype(float)
        return (data[self.feature_columns].corrwith(target, method="spearman")
                .sort_values(ascending=False))

    def feature_correlation(self) -> pd.DataFrame:
        """Pairwise Spearman correlation between features (robust to skewed counts)."""
        return self.frame[self.feature_columns].corr(method="spearman")

    def pca(self, n_components: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
        """Project all samples onto principal components of the standardised features.

        Counts are log-transformed first so a few very long samples do not dominate.
        Returns the projection with label, status, dataset, language and generator,
        and the explained variance ratio of each component.
        """
        values = self.frame[self.feature_columns].to_numpy(dtype=float)
        values = np.sign(values) * np.log1p(np.abs(values))
        scaled = StandardScaler().fit_transform(values)
        model = PCA(n_components=n_components, random_state=0)
        projection = model.fit_transform(scaled)
        columns = [f"pc{i + 1}" for i in range(n_components)]
        meta = ["label", "label_status", "dataset", "language", "generator"]
        frame = pd.DataFrame(projection, columns=columns, index=self.frame.index)
        return pd.concat([self.frame[meta], frame], axis=1), model.explained_variance_ratio_

    def export(self, directory: Path) -> dict[str, Path]:
        """Write every analysis table to ``directory`` and return the written paths."""
        directory.mkdir(parents=True, exist_ok=True)
        projection, variance = self.pca()
        tables = {
            "summary_by_label.csv": self.summary_by_label(),
            "class_separation.csv": self.class_separation(),
            "separation_by_language.csv": self.separation_by("language"),
            "trust_groups.csv": self.trust_groups(),
            "profile_by_language_label.csv": self.group_profile(["language", "label"]),
            "profile_by_generator.csv": self.group_profile(["generator"]),
            "label_correlation.csv": self.label_correlation().rename("spearman"),
            "feature_correlation.csv": self.feature_correlation(),
            "pca_projection.csv": projection,
        }
        paths = {}
        for name, table in tables.items():
            paths[name] = directory / name
            table.to_csv(paths[name])
        paths["pca_variance.json"] = directory / "pca_variance.json"
        paths["pca_variance.json"].write_text(json.dumps(
            {f"pc{i + 1}": round(float(v), 4) for i, v in enumerate(variance)}, indent=2))
        return paths
