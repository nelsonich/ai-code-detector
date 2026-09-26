"""Command line interface: run the pipeline end to end or one stage at a time."""

import argparse

from ai_code_detector.analysis.analyzer import CodeAnalyzer
from ai_code_detector.config import Config
from ai_code_detector.data.loader import DatasetLoader
from ai_code_detector.data.preprocessor import CodePreprocessor
from ai_code_detector.data.sources import WebdprocSource
from ai_code_detector.features.extractor import FeatureExtractor
from ai_code_detector.visualization.visualizer import Visualizer

RECORDS_FILE = "records.parquet"
CLEAN_FILE = "clean.parquet"
FEATURES_FILE = "features.parquet"


def load(config: Config) -> None:
    """Read all sources into ``data/interim``."""
    sources = [WebdprocSource(config.raw_dir / "webdproc")]
    frame = DatasetLoader(sources).load()
    DatasetLoader.save(frame, config.interim_dir / RECORDS_FILE)
    print(f"loaded {len(frame)} records")


def preprocess(config: Config) -> None:
    """Clean the interim table into ``data/processed``."""
    frame = DatasetLoader.read(config.interim_dir / RECORDS_FILE)
    clean = CodePreprocessor(**config.preprocessing).run(frame)
    DatasetLoader.save(clean, config.processed_dir / CLEAN_FILE)
    print(f"kept {len(clean)} of {len(frame)} records")


def features(config: Config) -> None:
    """Append feature columns to the clean table."""
    frame = DatasetLoader.read(config.processed_dir / CLEAN_FILE)
    table = FeatureExtractor(config.features["comment_markers"]).transform(frame)
    DatasetLoader.save(table, config.processed_dir / FEATURES_FILE)
    print(f"extracted features for {len(table)} records")


def analyze(config: Config) -> None:
    """Print the analytical summaries."""
    analyzer = CodeAnalyzer(DatasetLoader.read(config.processed_dir / FEATURES_FILE))
    print(analyzer.summary_by_label())
    print(analyzer.label_correlation())


def plot(config: Config) -> None:
    """Write all figures to ``reports/figures``."""
    frame = DatasetLoader.read(config.processed_dir / FEATURES_FILE)
    analyzer = CodeAnalyzer(frame)
    viz = Visualizer(config.figures_dir)
    viz.feature_distributions(frame, analyzer.feature_columns[:4])
    viz.feature_by_language(frame, "comment_line_ratio")
    viz.correlation_heatmap(analyzer.feature_correlation())
    print(f"figures written to {config.figures_dir}")


STAGES = {
    "load": load,
    "preprocess": preprocess,
    "features": features,
    "analyze": analyze,
    "plot": plot,
}


def main() -> None:
    """Parse the command line and run the requested stage(s)."""
    parser = argparse.ArgumentParser(prog="ai-code-detector")
    parser.add_argument("stage", choices=[*STAGES, "run"], help="'run' executes every stage")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    args = parser.parse_args()

    config = Config.load(args.config) if args.config else Config.load()
    config.ensure_dirs()
    for name in STAGES if args.stage == "run" else [args.stage]:
        print(f"== {name}")
        STAGES[name](config)


if __name__ == "__main__":
    main()
