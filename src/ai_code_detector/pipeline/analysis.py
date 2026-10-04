"""Analysis stages: extract features, summarise them and draw figures."""

import pandas as pd

from ai_code_detector.analysis.analyzer import CodeAnalyzer
from ai_code_detector.config import Config
from ai_code_detector.data.loader import DatasetLoader
from ai_code_detector.features.extractor import FeatureExtractor
from ai_code_detector.pipeline import PROCESSED_CLEAN, PROCESSED_FEATURES
from ai_code_detector.visualization.visualizer import Visualizer


def features(config: Config) -> None:
    """Append feature columns to the clean table."""
    frame = DatasetLoader.read(config.processed_dir / PROCESSED_CLEAN)
    table = FeatureExtractor().transform(frame)
    DatasetLoader.save(table, config.processed_dir / PROCESSED_FEATURES)
    print(f"extracted features for {len(table)} records")


def analyze(config: Config) -> None:
    """Export every analysis table to ``reports/tables`` and print the key results."""
    analyzer = CodeAnalyzer(DatasetLoader.read(config.processed_dir / PROCESSED_FEATURES))
    paths = analyzer.export(config.tables_dir)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("Features that separate AI from human code best (verified labels):")
        print(analyzer.class_separation().head(10)[
            ["median_ai", "median_human", "auc", "cohens_d"]].round(3))
        print("\nWhere unverified student code lies (medians):")
        print(analyzer.trust_groups().round(3))
    _, variance = analyzer.pca()
    print(f"\nPCA explained variance: {[round(float(v), 3) for v in variance]}")
    print(f"tables written to {config.tables_dir}: {sorted(p.name for p in paths.values())}")


def plot(config: Config, top_features: int = 6) -> None:
    """Write all figures to ``reports/figures``."""
    frame = DatasetLoader.read(config.processed_dir / PROCESSED_FEATURES)
    analyzer = CodeAnalyzer(frame)
    separation = analyzer.class_separation()
    top = list(separation.index[:top_features])
    projection, variance = analyzer.pca()
    viz = Visualizer(config.figures_dir)
    paths = [
        viz.dataset_composition(frame),
        viz.feature_separation(separation),
        viz.feature_distributions(frame, top),
        viz.habits_by_source(frame),
        viz.separation_by_language(analyzer.separation_by("language").loc[
            list(separation.index[:12])]),
        viz.pca_projection(projection, variance),
        viz.correlation_heatmap(analyzer.feature_correlation()),
    ]
    print(f"figures written to {config.figures_dir}: {[p.name for p in paths]}")


STAGES = {"features": features, "analyze": analyze, "plot": plot}
