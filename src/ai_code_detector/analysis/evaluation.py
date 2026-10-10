"""How much the hand-crafted features are worth together, as one classifier.

Single-feature statistics miss interactions: a feature can mean one thing in Python and
another in Java. Here a gradient-boosting model sees all features plus the language and
is scored on data it has not seen in three ways, from easiest to closest to real use:
new tasks, an unseen language, and an unseen origin (another population of people).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

AI = "ai"
IN_DOMAIN, UNSEEN_LANGUAGE, UNSEEN_ORIGIN = "new tasks", "unseen language", "unseen origin"
SPLITS = (IN_DOMAIN, UNSEEN_LANGUAGE, UNSEEN_ORIGIN)

FEATURE_GROUPS = {
    "size": ["n_chars", "n_lines", "avg_line_length", "max_line_length"],
    "layout": ["blank_line_ratio", "trailing_whitespace_ratio"],
    "indentation": ["avg_indent", "indent_std", "tab_indent_ratio", "mixed_indentation",
                    "even_indent_ratio"],
    "comments": ["has_comments", "comment_line_ratio", "comment_char_ratio",
                 "comment_armenian_ratio", "comment_cyrillic_ratio"],
    "identifiers": ["n_identifiers", "avg_identifier_length", "short_identifier_ratio",
                    "snake_case_ratio", "camel_case_ratio"],
    "lexical": ["token_diversity", "modern_constructs_per_line", "legacy_constructs_per_line"],
}
ALL_FEATURES = [c for columns in FEATURE_GROUPS.values() for c in columns]
# Features an editor or auto-formatter can change without touching the program.
FORMATTING = ["blank_line_ratio", "trailing_whitespace_ratio", "tab_indent_ratio",
              "mixed_indentation", "even_indent_ratio"]


def feature_sets() -> dict[str, list[str]]:
    """Feature sets compared in the report: a length baseline, all features, ablations."""
    sets = {
        "length only (baseline)": ["n_chars", "n_lines"],
        "all 24 features": ALL_FEATURES,
        "without formatting": [c for c in ALL_FEATURES if c not in FORMATTING],
    }
    for group, columns in FEATURE_GROUPS.items():
        sets[f"without {group}"] = [c for c in ALL_FEATURES if c not in columns]
    return sets


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Share of AI code caught when at most ``max_fpr`` of human code is flagged."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(tpr[fpr <= max_fpr].max())


class FeatureEvaluator:
    """Train and score a classifier per feature set on several train/test splits.

    Uses verified labels only and leaves out AI code generated for webdproc tasks, whose
    human counterpart is unverified. Splits with fewer than ``min_per_class`` test
    samples of either class are skipped.
    """

    def __init__(self, frame: pd.DataFrame, languages: list[str], min_per_class: int = 50,
                 n_folds: int = 5, seed: int = 0) -> None:
        """Keep the verified, comparable rows and the language list used for encoding."""
        keep = (frame["label_status"] == "verified") & (frame["origin"].astype(str) != "webdproc")
        self.frame = frame[keep].reset_index(drop=True)
        self.languages = languages
        self.min_per_class = min_per_class
        self.n_folds = n_folds
        self.seed = seed
        self.y = (self.frame["label"] == AI).to_numpy().astype(int)

    def splits(self):
        """Yield ``(split, held_out, train_index, test_index)`` for every evaluation."""
        groups = self.frame["task_id"].astype(str)
        for i, (train, test) in enumerate(GroupKFold(self.n_folds).split(self.frame, self.y,
                                                                         groups)):
            yield IN_DOMAIN, f"fold {i + 1}", train, test
        for split, column in ((UNSEEN_LANGUAGE, "language"), (UNSEEN_ORIGIN, "origin")):
            values = self.frame[column].astype(str)
            for held_out in sorted(values.unique()):
                test = np.where(values == held_out)[0]
                n_ai = int(self.y[test].sum())
                if min(n_ai, len(test) - n_ai) >= self.min_per_class:
                    yield split, held_out, np.where(values != held_out)[0], test

    def run(self, sets: dict[str, list[str]]) -> pd.DataFrame:
        """Return one row per split, held-out part and feature set with AUC and TPRs."""
        rows = []
        for split, held_out, train, test in self.splits():
            for name, columns in sets.items():
                x = self._matrix(columns)
                model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                       random_state=self.seed)
                model.fit(x[train], self.y[train])
                scores = model.predict_proba(x[test])[:, 1]
                rows.append({
                    "split": split, "held_out": held_out, "feature_set": name,
                    "auc": roc_auc_score(self.y[test], scores),
                    "tpr_at_1pct_fpr": tpr_at_fpr(self.y[test], scores, 0.01),
                    "tpr_at_5pct_fpr": tpr_at_fpr(self.y[test], scores, 0.05),
                    "n_test": len(test),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def summary(results: pd.DataFrame) -> pd.DataFrame:
        """Mean AUC per feature set and split, in the order of ``SPLITS``."""
        table = results.pivot_table(index="feature_set", columns="split", values="auc",
                                    sort=False)
        return table[[s for s in SPLITS if s in table.columns]]

    def _matrix(self, columns: list[str]) -> np.ndarray:
        values = self.frame[columns].to_numpy(dtype=float)
        language = self.frame["language"].astype(str).to_numpy()
        onehot = np.stack([(language == lang) for lang in self.languages], axis=1)
        return np.hstack([values, onehot.astype(float)])
