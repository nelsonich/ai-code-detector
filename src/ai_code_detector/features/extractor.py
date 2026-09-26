"""Numeric features describing the style and structure of a code sample."""

import numpy as np
import pandas as pd


class FeatureExtractor:
    """Compute one row of numeric features per code sample.

    The same extractor is used for the exploratory analysis and, later, as the
    input of the model, so both always see identical features.
    """

    def __init__(self, comment_markers: dict[str, list[str]]) -> None:
        """Store per-language comment markers."""
        self.comment_markers = comment_markers

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the input table with feature columns appended."""
        features = pd.DataFrame(
            [self.extract(code, lang) for code, lang in zip(frame["code"], frame["language"])],
            index=frame.index,
        )
        return pd.concat([frame, features], axis=1)

    def extract(self, code: str, language: str) -> dict[str, float]:
        """Compute features for a single code sample."""
        lines = code.splitlines() or [""]
        stripped = [line.strip() for line in lines]
        non_blank = [line for line in lines if line.strip()]
        indents = [len(line) - len(line.lstrip(" \t")) for line in non_blank]
        markers = tuple(self.comment_markers.get(language, []))
        comment_lines = [line for line in stripped if markers and line.startswith(markers)]
        return {
            "n_chars": len(code),
            "n_lines": len(lines),
            "avg_line_length": float(np.mean([len(line) for line in lines])),
            "blank_line_ratio": 1 - len(non_blank) / len(lines),
            "avg_indent": float(np.mean(indents)) if indents else 0.0,
            "indent_std": float(np.std(indents)) if indents else 0.0,
            "comment_line_ratio": len(comment_lines) / len(lines),
            "trailing_whitespace_ratio": sum(line != line.rstrip() for line in lines) / len(lines),
        }
