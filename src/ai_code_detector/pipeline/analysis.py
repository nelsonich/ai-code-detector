"""Analysis stages: extract features, summarise them and draw figures."""

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
    """Print the analytical summaries."""
    analyzer = CodeAnalyzer(DatasetLoader.read(config.processed_dir / PROCESSED_FEATURES))
    print(analyzer.summary_by_label())
    print(analyzer.label_correlation())


def plot(config: Config) -> None:
    """Write all figures to ``reports/figures``."""
    frame = DatasetLoader.read(config.processed_dir / PROCESSED_FEATURES)
    analyzer = CodeAnalyzer(frame)
    viz = Visualizer(config.figures_dir)
    viz.feature_distributions(frame, analyzer.feature_columns[:4])
    viz.feature_by_language(frame, "comment_line_ratio")
    viz.correlation_heatmap(analyzer.feature_correlation())
    print(f"figures written to {config.figures_dir}")


STAGES = {"features": features, "analyze": analyze, "plot": plot}
