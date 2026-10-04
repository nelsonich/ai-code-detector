import numpy as np
import pandas as pd
import pytest

from ai_code_detector.analysis.analyzer import CodeAnalyzer


def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for label, status, shift in [("ai", "verified", 1.0), ("human", "verified", 0.0),
                                 ("human", "unverified", 0.5)]:
        for i in range(n):
            rows.append({
                "label": label, "label_status": status,
                "language": "python" if i % 2 else "java",
                "dataset": f"{label}-{status}", "origin": "a" if i < n // 2 else "b",
                "generator": "g" if label == "ai" else "none",
                "separating": shift + rng.normal(0, 0.1),
                "noise": rng.normal(0, 1),
            })
    return pd.DataFrame(rows)


def test_features_are_the_non_record_columns():
    assert CodeAnalyzer(_frame()).feature_columns == ["separating", "noise"]


def test_class_separation_ranks_the_separating_feature_first():
    table = CodeAnalyzer(_frame()).class_separation()
    assert list(table.index) == ["separating", "noise"]
    assert table.loc["separating", "auc"] == pytest.approx(1.0)
    assert table.loc["separating", "cohens_d"] > 5
    assert table.loc["separating", "p_value"] < 1e-10
    assert table.loc["noise", "separation"] < 0.3


def test_separation_by_language_skips_groups_with_too_few_samples():
    table = CodeAnalyzer(_frame()).separation_by("language", min_per_class=30)
    assert set(table.columns) == {"java", "python"}
    assert (table.loc["separating"] > 0.99).all()
    assert CodeAnalyzer(_frame()).separation_by("language", min_per_class=31).empty


def test_robust_features_rank_a_source_artifact_below_a_real_difference():
    frame = _frame()
    is_ai = frame["label"] == "ai"
    # Higher in AI code within origin "a", higher in human code within origin "b".
    flip = np.where(frame["origin"] == "a", 1.0, -1.0)
    frame["artifact"] = np.where(is_ai, flip, -flip) + np.random.default_rng(1).normal(
        0, 0.1, len(frame))
    analyzer = CodeAnalyzer(frame)
    cells = analyzer.separation_by_origin(min_per_class=10)
    assert set(cells.columns) == {"java/a", "java/b", "python/a", "python/b"}
    table = analyzer.robust_features(min_per_class=10)
    assert table.index[0] == "separating"
    assert table.loc["separating", "agree"] == 1.0 and table.loc["separating", "weakest"] > 0.9
    assert table.loc["artifact", "weakest"] < -0.9


def test_trust_groups_place_unverified_between_the_classes():
    table = CodeAnalyzer(_frame()).trust_groups()
    row = table.loc["separating"]
    assert row["human (verified)"] < row["human (unverified)"] < row["ai"]


def test_pca_returns_projection_with_metadata_and_variance():
    projection, variance = CodeAnalyzer(_frame()).pca()
    assert {"pc1", "pc2", "label", "dataset", "generator"} <= set(projection.columns)
    assert len(projection) == 180 and len(variance) == 2
    assert variance[0] >= variance[1] > 0


def test_export_writes_every_table(tmp_path):
    paths = CodeAnalyzer(_frame()).export(tmp_path)
    assert all(path.exists() for path in paths.values())
    assert "class_separation.csv" in paths and "pca_variance.json" in paths
    summary = pd.read_csv(paths["summary_by_label.csv"], index_col=[0, 1])
    assert ("separating", "ai") in summary.index
