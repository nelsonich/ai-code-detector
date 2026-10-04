"""Numeric features describing the style and structure of a code sample."""

import math
import re

import numpy as np
import pandas as pd

from ai_code_detector.features import syntax

_ARMENIAN = re.compile(r"[Ա-և]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LETTER = re.compile(r"[^\W\d_]")
_SNAKE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_CAMEL = re.compile(r"^[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+$")


class FeatureExtractor:
    """Compute one row of numeric features per code sample.

    The same extractor is used for the exploratory analysis and, later, as the
    input of the model, so both always see identical features. Feature groups:
    size, layout and indentation, comments (amount and natural language),
    identifiers, token diversity and modern versus legacy constructs.
    """

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the input table with feature columns appended."""
        features = pd.DataFrame(
            [self.extract(code, lang) for code, lang in zip(frame["code"], frame["language"])],
            index=frame.index,
        )
        return pd.concat([frame, features], axis=1)

    def extract(self, code: str, language: str) -> dict[str, float]:
        """Compute all features for a single code sample."""
        comments, without_comments = syntax.split_comments(code, language)
        lines = code.splitlines() or [""]
        return (self._layout(code, lines)
                | self._indentation(lines)
                | self._comments(code, comments, without_comments)
                | self._identifiers(syntax.strip_strings(without_comments), language)
                | self._constructs(without_comments, language, len(lines)))

    @staticmethod
    def _layout(code: str, lines: list[str]) -> dict[str, float]:
        non_blank = [line for line in lines if line.strip()]
        return {
            "n_chars": len(code),
            "n_lines": len(lines),
            "avg_line_length": float(np.mean([len(line) for line in lines])),
            "max_line_length": max(len(line) for line in lines),
            "blank_line_ratio": 1 - len(non_blank) / len(lines),
            "trailing_whitespace_ratio": sum(line != line.rstrip() for line in lines) / len(lines),
        }

    @staticmethod
    def _indentation(lines: list[str]) -> dict[str, float]:
        indents = [line[:len(line) - len(line.lstrip(" \t"))] for line in lines if line.strip()]
        widths = [len(indent.expandtabs(4)) for indent in indents]
        indented = [indent for indent in indents if indent]
        spaces = [len(indent) for indent in indented if "\t" not in indent]
        with_tabs = sum("\t" in indent for indent in indented)
        return {
            "avg_indent": float(np.mean(widths)) if widths else 0.0,
            "indent_std": float(np.std(widths)) if widths else 0.0,
            "tab_indent_ratio": with_tabs / len(indented) if indented else 0.0,
            "mixed_indentation": float(bool(with_tabs) and bool(spaces)),
            "even_indent_ratio": (sum(width % 2 == 0 for width in spaces) / len(spaces)
                                  if spaces else 1.0),
        }

    @staticmethod
    def _comments(code: str, comments: list[str], without_comments: str) -> dict[str, float]:
        text = " ".join(comments)
        letters = len(_LETTER.findall(text))
        # split("\n") keeps a trailing empty line, so both sides stay aligned line by line.
        code_lines = code.split("\n")
        stripped_lines = without_comments.split("\n")
        commented = sum(original.strip() != stripped.strip()
                        for original, stripped in zip(code_lines, stripped_lines))
        n_lines = len(code.splitlines()) or 1
        return {
            "has_comments": float(bool(comments)),
            "comment_line_ratio": commented / n_lines,
            "comment_char_ratio": len(text) / len(code) if code else 0.0,
            "comment_armenian_ratio": len(_ARMENIAN.findall(text)) / letters if letters else 0.0,
            "comment_cyrillic_ratio": len(_CYRILLIC.findall(text)) / letters if letters else 0.0,
        }

    @staticmethod
    def _identifiers(code: str, language: str) -> dict[str, float]:
        names = set(syntax.identifiers(code, language))
        all_tokens = syntax.tokens(code)
        vocabulary = len(set(all_tokens))
        return {
            "n_identifiers": len(names),
            "avg_identifier_length": float(np.mean([len(n) for n in names])) if names else 0.0,
            "short_identifier_ratio": (sum(len(n) <= 2 for n in names) / len(names)
                                       if names else 0.0),
            "snake_case_ratio": sum(bool(_SNAKE.match(n)) for n in names) / len(names)
            if names else 0.0,
            "camel_case_ratio": sum(bool(_CAMEL.match(n)) for n in names) / len(names)
            if names else 0.0,
            # Herdan's C: vocabulary growth that, unlike the type-token ratio, does not
            # shrink just because a sample is longer.
            "token_diversity": (math.log(vocabulary) / math.log(len(all_tokens))
                                if len(all_tokens) > 1 and vocabulary > 1 else 0.0),
        }

    @staticmethod
    def _constructs(code: str, language: str, n_lines: int) -> dict[str, float]:
        modern, legacy = syntax.construct_counts(code, language)
        return {
            "modern_constructs_per_line": modern / n_lines,
            "legacy_constructs_per_line": legacy / n_lines,
        }
