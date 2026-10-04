"""Matplotlib figures for the report and the presentation, saved as PNG files.

Every figure uses the same colour roles, so an entity keeps its colour across charts.
The categorical colours were checked with a colour-vision-deficiency validator; the
values behind each figure are exported as tables by ``CodeAnalyzer``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

GROUP_COLORS = {
    "human (verified)": "#2a78d6",
    "ai": "#eb6834",
    "human (unverified)": "#1baf7a",
}
LABEL_COLORS = {"human": GROUP_COLORS["human (verified)"], "ai": GROUP_COLORS["ai"]}
SOURCE_COLORS = {"progpedia": GROUP_COLORS["human (verified)"],
                 "codenet": GROUP_COLORS["human (verified)"],
                 "webdproc": GROUP_COLORS["human (unverified)"],
                 "generated": GROUP_COLORS["ai"]}
INK, INK_SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8f8e", "#b8302f"])

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY, "legend.frameon": False,
}


def trust_group(frame: pd.DataFrame) -> pd.Series:
    """Name each row's group: verified human, unverified human or AI."""
    return np.where(frame["label"] == "ai", "ai",
                    np.where(frame["label_status"] == "verified",
                             "human (verified)", "human (unverified)"))


class Visualizer:
    """Produce the distribution, categorical and relationship figures of the project."""

    def __init__(self, output_dir: Path) -> None:
        """Remember where figures are written and apply the shared style."""
        self.output_dir = output_dir
        plt.rcParams.update(STYLE)

    def dataset_composition(self, frame: pd.DataFrame) -> Path:
        """Draw grouped bars of the number of samples per language and trust group."""
        counts = (frame.assign(group=trust_group(frame))
                  .groupby(["language", "group"], observed=True).size()
                  .unstack(fill_value=0).reindex(columns=list(GROUP_COLORS), fill_value=0))
        counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(counts))
        width = 0.27
        for i, group in enumerate(GROUP_COLORS):
            ax.bar(x + (i - 1) * width, counts[group], width * 0.92,
                   color=GROUP_COLORS[group], label=group)
        ax.set_xticks(x, counts.index)
        ax.set_ylabel("samples")
        ax.grid(axis="x", visible=False)
        ax.set_title("Dataset after cleaning: samples per language and group")
        ax.legend(ncols=3, loc="upper right")
        return self._save(fig, "01_dataset_composition.png")

    def feature_separation(self, separation: pd.DataFrame) -> Path:
        """Draw horizontal bars of how strongly each feature separates AI from human code."""
        data = separation.assign(signed=(separation["auc"] - 0.5) * 2).sort_values("signed")
        colors = [LABEL_COLORS["ai"] if v > 0 else LABEL_COLORS["human"] for v in data["signed"]]
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        ax.barh(data.index, data["signed"], color=colors, height=0.72)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("separation  (AUC − 0.5) × 2:  ← higher in human code   "
                      "higher in AI code →")
        ax.grid(axis="y", visible=False)
        ax.set_title("Which features separate AI from human code (verified labels)")
        for name, value in zip(data.index, data["signed"]):
            if abs(value) >= 0.25:
                ax.text(value + (0.02 if value > 0 else -0.02), name, f"{value:+.2f}",
                        va="center", ha="left" if value > 0 else "right",
                        color=INK_SECONDARY, fontsize=8)
        handles = [plt.Rectangle((0, 0), 1, 1, color=LABEL_COLORS[k]) for k in ("human", "ai")]
        ax.legend(handles, ["higher in human code", "higher in AI code"], loc="lower right")
        return self._save(fig, "02_feature_separation.png")

    def feature_distributions(self, frame: pd.DataFrame, features: list[str]) -> Path:
        """Draw each feature's distribution per class (verified labels) as small multiples."""
        data = frame[frame["label_status"] == "verified"]
        cols = 3
        rows = int(np.ceil(len(features) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3.4 * rows))
        for ax, feature in zip(np.ravel(axes), features):
            upper = data[feature].quantile(0.98)
            bins = np.linspace(data[feature].min(), upper if upper > 0 else 1, 30)
            for label in ("human", "ai"):
                values = data.loc[data["label"] == label, feature].clip(upper=bins[-1])
                ax.hist(values, bins=bins, density=True, histtype="step", linewidth=2,
                        color=LABEL_COLORS[label], label=label)
            ax.set_title(feature, fontsize=10)
            ax.grid(axis="x", visible=False)
            ax.set_yticks([])
        for ax in np.ravel(axes)[len(features):]:
            ax.set_visible(False)
        np.ravel(axes)[0].legend(loc="upper right")
        fig.suptitle("Distributions of the most separating features (values above the 98th "
                     "percentile are pooled in the last bin)", x=0.01, ha="left",
                     fontsize=12, fontweight="bold")
        return self._save(fig, "03_feature_distributions.png")

    def habits_by_source(self, frame: pd.DataFrame) -> Path:
        """Draw the share of samples with each formatting habit, per source dataset."""
        habits = {
            "trailing whitespace": frame["trailing_whitespace_ratio"] > 0,
            "tab indentation": frame["tab_indent_ratio"] > 0,
            "any comment": frame["has_comments"] > 0,
        }
        shares = pd.DataFrame(habits).groupby(frame["dataset"].astype(str)).mean()
        shares = shares.loc[[d for d in SOURCE_COLORS if d in shares.index]]
        colors = [SOURCE_COLORS[d] for d in shares.index]
        fig, axes = plt.subplots(1, len(habits), figsize=(12, 3.8), sharey=True)
        for ax, habit in zip(axes, habits):
            ax.bar(shares.index, shares[habit], color=colors, width=0.7)
            for i, value in enumerate(shares[habit]):
                ax.text(i, value + 0.02, f"{value:.0%}", ha="center", color=INK_SECONDARY,
                        fontsize=9)
            ax.set_title(habit, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="x", visible=False)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[0].set_ylabel("share of samples")
        fig.suptitle("Formatting habits by source: verified human (blue), unverified "
                     "student (green), AI (orange)", x=0.01, ha="left", fontsize=12,
                     fontweight="bold")
        return self._save(fig, "04_habits_by_source.png")

    def separation_by_language(self, auc: pd.DataFrame) -> Path:
        """Draw a heatmap of single-feature AUC within each language (0.5: no separation)."""
        fig, ax = plt.subplots(figsize=(9, 0.38 * len(auc) + 1.5))
        image = ax.imshow(auc.to_numpy(), cmap=DIVERGING, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(auc.columns)), auc.columns)
        ax.set_yticks(range(len(auc.index)), auc.index)
        ax.grid(False)
        for (i, j), value in np.ndenumerate(auc.to_numpy()):
            if abs(value - 0.5) >= 0.2:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(value - 0.5) >= 0.35 else INK)
        bar = fig.colorbar(image, ax=ax, shrink=0.6)
        bar.set_label("AUC  (red: higher in AI, blue: higher in human)")
        ax.set_title("Separation of each feature within each language (verified labels)")
        return self._save(fig, "05_separation_by_language.png")

    def pca_projection(self, projection: pd.DataFrame, variance: np.ndarray) -> Path:
        """Draw the first two principal components, coloured by trust group."""
        groups = trust_group(projection)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        for group, color in GROUP_COLORS.items():
            points = projection[groups == group]
            ax.scatter(points["pc1"], points["pc2"], s=9, alpha=0.35, color=color,
                       linewidths=0, label=f"{group} (n={len(points)})")
        ax.set_xlabel(f"PC1 ({variance[0]:.0%} of variance)")
        ax.set_ylabel(f"PC2 ({variance[1]:.0%} of variance)")
        ax.set_title("All samples projected on two principal components")
        legend = ax.legend(loc="upper right", markerscale=2.5)
        for handle in legend.legend_handles:
            handle.set_alpha(1)
        return self._save(fig, "06_pca_projection.png")

    def correlation_heatmap(self, matrix: pd.DataFrame) -> Path:
        """Draw a heatmap of the Spearman correlation between features."""
        fig, ax = plt.subplots(figsize=(10, 8.5))
        image = ax.imshow(matrix.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(len(matrix)), matrix.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(matrix)), matrix.index, fontsize=8)
        ax.grid(False)
        fig.colorbar(image, ax=ax, shrink=0.7).set_label("Spearman correlation")
        ax.set_title("Correlation between features")
        return self._save(fig, "07_feature_correlation.png")

    def _save(self, fig: plt.Figure, name: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path
