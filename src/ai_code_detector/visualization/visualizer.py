"""Matplotlib and seaborn plots saved as image files."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


class Visualizer:
    """Produce the distribution, categorical and relationship plots of the project."""

    def __init__(self, output_dir: Path) -> None:
        """Remember where figures are written."""
        self.output_dir = output_dir
        sns.set_theme(style="whitegrid")

    def feature_distributions(self, frame: pd.DataFrame, features: list[str]) -> Path:
        """Boxplots of selected features split by label."""
        fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 4))
        for ax, feature in zip(axes.flat, features):
            sns.boxplot(data=frame, x="label", y=feature, ax=ax)
            ax.set_title(feature)
        return self._save(fig, "feature_distributions.png")

    def feature_by_language(self, frame: pd.DataFrame, feature: str) -> Path:
        """Bar chart of a feature's mean per language and label."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=frame, x="language", y=feature, hue="label", ax=ax)
        ax.set_title(f"{feature} by language")
        return self._save(fig, f"{feature}_by_language.png")

    def correlation_heatmap(self, matrix: pd.DataFrame) -> Path:
        """Heatmap of a correlation matrix."""
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Feature correlation")
        return self._save(fig, "correlation_heatmap.png")

    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self.output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
