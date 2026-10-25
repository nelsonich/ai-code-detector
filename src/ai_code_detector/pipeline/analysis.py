"""Analysis stages: extract features, summarise them and draw figures."""

import json

import pandas as pd

from ai_code_detector.analysis.analyzer import CodeAnalyzer
from ai_code_detector.analysis.evaluation import (
    ALL_FEATURES,
    NORMALIZED,
    FeatureEvaluator,
    feature_sets,
    select_features,
)
from ai_code_detector.config import Config
from ai_code_detector.data.loader import DatasetLoader
from ai_code_detector.features.extractor import FeatureExtractor
from ai_code_detector.features.language_model import normalize_layout
from ai_code_detector.pipeline import PROCESSED_CLEAN, PROCESSED_FEATURES
from ai_code_detector.pipeline.language_model import scored_features
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

    scored = scored_features(config)
    if scored is not None:
        robust = CodeAnalyzer(scored).robust_features()
        robust.to_csv(config.tables_dir / "robust_features_lm.csv")
        print(f"\nRobustness on the {len(scored)} language-model scored records:")
        print(robust.round(2).head(12))


EVALUATION = "feature_evaluation.csv"
IMPORTANCE = "feature_importance.csv"
SELECTED = "selected_features.json"


def with_normalized_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add every feature recomputed on code whose editor-dependent layout is removed."""
    normalized = FeatureExtractor().transform(
        frame[["language"]].assign(code=frame["code"].map(normalize_layout)))
    renamed = normalized[ALL_FEATURES].add_suffix(NORMALIZED)
    return pd.concat([frame, renamed], axis=1)


def evaluate(config: Config) -> None:
    """Score the features as one classifier, rank them and pick a smaller robust set.

    Every set is trained on part of the verified data and scored on new tasks, an unseen
    language and an unseen origin. Features are ranked by permutation importance on the
    unseen splits; the selected set drops features that do not help or duplicate others.
    """
    frame = with_normalized_features(
        DatasetLoader.read(config.processed_dir / PROCESSED_FEATURES))
    evaluator = FeatureEvaluator(frame, config.preprocessing["languages"])
    importance = evaluator.importance(ALL_FEATURES)
    importance.to_csv(config.tables_dir / IMPORTANCE)
    correlation = evaluator.frame[ALL_FEATURES].corr(method="spearman")
    selected = select_features(importance, correlation)
    (config.tables_dir / SELECTED).write_text(json.dumps(selected, indent=2))

    sets = feature_sets() | {
        f"selected ({len(selected)})": selected,
        "all features, normalized code": [c + NORMALIZED for c in ALL_FEATURES],
        "selected, normalized code": [c + NORMALIZED for c in selected],
    }
    results = evaluator.run(sets)
    results.to_csv(config.tables_dir / EVALUATION, index=False)
    with pd.option_context("display.width", 160):
        print(f"Selected {len(selected)} of {len(ALL_FEATURES)} features: {selected}")
        print("Mean AUC of a classifier on the features:")
        print(FeatureEvaluator.summary(results).round(3))
    print(f"tables written to {config.tables_dir}: {EVALUATION}, {IMPORTANCE}, {SELECTED}")


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
        viz.robustness(analyzer.separation_by_origin(), list(analyzer.robust_features().index)),
    ]
    evaluation_path = config.tables_dir / EVALUATION
    if evaluation_path.exists():
        summary = FeatureEvaluator.summary(pd.read_csv(evaluation_path))
        paths.append(viz.model_evaluation(summary))
        paths.append(viz.group_ablation(summary))
        selected = json.loads((config.tables_dir / SELECTED).read_text())
        importance = pd.read_csv(config.tables_dir / IMPORTANCE, index_col=0)
        paths.append(viz.feature_importance(importance, selected))
    scored = scored_features(config)
    if scored is not None:
        lm = CodeAnalyzer(scored)
        paths.append(viz.robustness(
            lm.separation_by_origin(), list(lm.robust_features().index),
            name="12_language_model_signals.png",
            title=f"Style and language-model signals on {len(scored)} scored samples "
                  "(rows: most consistent first)"))
    print(f"figures written to {config.figures_dir}: {[p.name for p in paths]}")


STAGES = {"features": features, "analyze": analyze, "evaluate": evaluate, "plot": plot}
