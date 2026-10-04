import numpy as np
import pandas as pd

from ai_code_detector.analysis.analyzer import CodeAnalyzer
from ai_code_detector.visualization.visualizer import Visualizer, source_group, trust_group


def _frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    groups = [("ai", "verified", "generated"), ("human", "verified", "progpedia"),
              ("human", "unverified", "webdproc")]
    rows = []
    for label, status, dataset in groups:
        for i in range(n):
            rows.append({
                "label": label, "label_status": status, "dataset": dataset,
                "language": ["python", "java"][i % 2], "generator": "none",
                "origin": "a" if i < n // 2 else "b",
                "trailing_whitespace_ratio": rng.uniform(0, 0.2) if label == "human" else 0.0,
                "tab_indent_ratio": rng.uniform(0, 1) * (label == "human"),
                "has_comments": float(rng.random() < 0.5),
                "token_diversity": rng.normal(0.7 + 0.05 * (label == "ai"), 0.02),
            })
    return pd.DataFrame(rows)


def test_trust_group_names_the_three_groups():
    groups = trust_group(_frame(n=1))
    assert list(groups) == ["ai", "human (verified)", "human (unverified)"]


def test_source_group_splits_datasets_with_both_classes():
    frame = pd.DataFrame({"dataset": ["droid", "droid", "progpedia"],
                          "label": ["ai", "human", "human"]})
    assert list(source_group(frame)) == ["droid ai", "droid human", "progpedia"]


def test_every_figure_is_written(tmp_path):
    frame = _frame()
    analyzer = CodeAnalyzer(frame)
    separation = analyzer.class_separation()
    projection, variance = analyzer.pca()
    viz = Visualizer(tmp_path)
    paths = [
        viz.dataset_composition(frame),
        viz.feature_separation(separation),
        viz.feature_distributions(frame, list(separation.index[:3])),
        viz.habits_by_source(frame),
        viz.separation_by_language(analyzer.separation_by("language", min_per_class=10)),
        viz.pca_projection(projection, variance),
        viz.correlation_heatmap(analyzer.feature_correlation()),
        viz.robustness(analyzer.separation_by_origin(min_per_class=5),
                       list(analyzer.robust_features(min_per_class=5).index)),
    ]
    assert len({p.name for p in paths}) == 8
    assert all(p.exists() and p.stat().st_size > 10_000 for p in paths)
