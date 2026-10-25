import numpy as np
import pandas as pd

from ai_code_detector.analysis.evaluation import (
    ALL,
    ALL_FEATURES,
    BASELINE,
    IN_DOMAIN,
    UNSEEN_LANGUAGE,
    UNSEEN_ORIGIN,
    FeatureEvaluator,
    feature_sets,
    select_features,
    tpr_at_fpr,
)


def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for origin in ("a", "b"):
        for language in ("python", "java"):
            for label in ("ai", "human"):
                for i in range(n):
                    row = {c: rng.normal() for c in ALL_FEATURES}
                    row["tab_indent_ratio"] += 2.0 if label == "human" else 0.0
                    rows.append(row | {"label": label, "label_status": "verified",
                                       "origin": origin, "language": language,
                                       "task_id": f"{origin}:{language}:{i % 20}"})
    rows.append(rows[0] | {"label_status": "unverified"})
    rows.append(rows[1] | {"origin": "webdproc"})
    return pd.DataFrame(rows)


def test_feature_sets_cover_baseline_all_and_one_ablation_per_group():
    sets = feature_sets()
    assert sets[ALL] == ALL_FEATURES and len(ALL_FEATURES) == len(set(ALL_FEATURES)) == 25
    assert BASELINE in sets and "without indentation" in sets and "without structure" in sets
    assert "tab_indent_ratio" not in sets["without formatting"]


def test_tpr_at_fpr_counts_ai_caught_below_the_false_positive_limit():
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95])
    assert tpr_at_fpr(y, scores, 0.0) == 0.5
    assert tpr_at_fpr(y, scores, 0.25) == 1.0


def test_evaluator_drops_unverified_and_webdproc_and_builds_three_kinds_of_split():
    evaluator = FeatureEvaluator(_frame(), ["python", "java"], min_per_class=50)
    assert len(evaluator.frame) == 480
    kinds = {split for split, *_ in evaluator.splits()}
    assert kinds == {IN_DOMAIN, UNSEEN_LANGUAGE, UNSEEN_ORIGIN}


def test_informative_feature_beats_the_length_baseline_everywhere():
    evaluator = FeatureEvaluator(_frame(), ["python", "java"], min_per_class=50, n_folds=3)
    sets = {k: v for k, v in feature_sets().items()
            if k in (BASELINE, ALL)}
    summary = FeatureEvaluator.summary(evaluator.run(sets))
    assert list(summary.columns) == [IN_DOMAIN, UNSEEN_LANGUAGE, UNSEEN_ORIGIN]
    assert (summary.loc[ALL] > 0.8).all()
    assert (summary.loc[BASELINE] < 0.7).all()


def test_importance_ranks_the_informative_feature_first():
    evaluator = FeatureEvaluator(_frame(), ["python", "java"], min_per_class=50)
    table = evaluator.importance(["tab_indent_ratio", "n_chars"], n_repeats=2)
    assert table.index[0] == "tab_indent_ratio"
    assert table.loc["tab_indent_ratio"].min() > table.loc["n_chars"].max()


def test_select_features_drops_useless_and_near_duplicate_features():
    importance = pd.DataFrame({"a": [0.05, 0.04], "b": [0.03, 0.02], "c": [0.02, -0.01]},
                              index=["unseen language", "unseen origin"]).T
    correlation = pd.DataFrame(np.eye(3), index=list("abc"), columns=list("abc"))
    correlation.loc["a", "b"] = correlation.loc["b", "a"] = 0.95
    assert select_features(importance, correlation) == ["a"]
    correlation.loc["a", "b"] = correlation.loc["b", "a"] = 0.5
    assert select_features(importance, correlation) == ["a", "b"]
