import pandas as pd
import pytest

from ai_code_detector.data.schema import RECORD_COLUMNS
from ai_code_detector.features import syntax
from ai_code_detector.features.extractor import FeatureExtractor

extract = FeatureExtractor().extract


def test_layout_counts():
    result = extract("def f():\n    return 1\n\n# done\n", "python")
    assert result["n_lines"] == 4
    assert result["blank_line_ratio"] == 0.25
    assert result["comment_line_ratio"] == 0.25
    assert result["avg_indent"] == 4 / 3
    assert result["max_line_length"] == len("    return 1")


def test_indentation_tabs_mixing_and_odd_widths():
    mixed = "int main() {\n\tint a = 1;\n   int b = 2;\n  return 0;\n}"
    result = extract(mixed, "c")
    assert result["tab_indent_ratio"] == 1 / 3
    assert result["mixed_indentation"] == 1.0
    assert result["even_indent_ratio"] == 0.5
    clean = extract("if x:\n    y = 1\n    z = 2", "python")
    assert clean["mixed_indentation"] == 0.0 and clean["even_indent_ratio"] == 1.0


def test_comments_block_inline_and_url_not_a_comment():
    code = ('/* header */\nlet url = "http://x.y"; // inline note\n'
            "const a = 1;\n/* multi\n line */")
    comments, rest = syntax.split_comments(code, "javascript")
    assert comments == ["/* header */", "/* multi\n line */", "// inline note"]
    assert "http://x.y" in rest and rest.count("\n") == code.count("\n")
    result = extract(code, "javascript")
    assert result["has_comments"] == 1.0
    assert result["comment_line_ratio"] == 4 / 5


def test_comment_natural_language_shares():
    armenian = extract("// Բարև աշխարհ\nlet a = 1;", "javascript")
    assert armenian["comment_armenian_ratio"] == 1.0 and armenian["comment_cyrillic_ratio"] == 0
    punctuated = extract("// Բարև, աշխարհ՝ ողջույն։\nlet a = 1;", "javascript")
    assert punctuated["comment_armenian_ratio"] == 1.0
    russian = extract("# привет world\nx = 1", "python")
    assert russian["comment_cyrillic_ratio"] == pytest.approx(6 / 11)
    html = extract("<!-- note -->\n<p>Hi</p>", "html")
    assert html["has_comments"] == 1.0 and html["comment_armenian_ratio"] == 0.0


def test_php_hash_comment_but_not_attribute_or_array_key():
    comments, _ = syntax.split_comments("<?php\n# note\n$a['#'] = 1;\n#[Attr]\n", "php")
    assert comments == ["# note"]


def test_identifiers_skip_keywords_strings_and_comments():
    code = 'const userName = "not_an_identifier";\nlet i = getUser(user_id); // commentWord'
    result = extract(code, "javascript")
    # userName, i, getUser, user_id
    assert result["n_identifiers"] == 4
    assert result["short_identifier_ratio"] == 1 / 4
    assert result["camel_case_ratio"] == 2 / 4 and result["snake_case_ratio"] == 1 / 4


def test_token_diversity_is_between_zero_and_one():
    repetitive = extract("a = a + a + a + a + a + a", "python")["token_diversity"]
    varied = extract("total = price * count - discount / rate", "python")["token_diversity"]
    assert 0 < repetitive < varied <= 1
    assert extract("x", "python")["token_diversity"] == 0.0


@pytest.mark.parametrize(("language", "code", "modern", "legacy"), [
    ("javascript", "const a = 1;\nvar b = 2;\nif (a === b) {}\nif (a == b) {}", 2, 2),
    ("python", "with open(p) as f:\n    print(f'{x}')\nfor i in range(len(xs)): pass", 2, 1),
    ("cpp", "auto p = nullptr;\nint *q = NULL;\nfor (auto x : v) {}", 4, 1),
    ("html", "<!DOCTYPE html>\n<header></header>\n<center>old</center>", 2, 1),
    ("css", ".a { display: flex; float: left; color: red !important; }", 1, 2),
])
def test_modern_and_legacy_constructs(language, code, modern, legacy):
    assert syntax.construct_counts(code, language) == (modern, legacy)


def test_unknown_language_and_empty_code_do_not_fail():
    result = extract("", "cobol")
    assert result["n_chars"] == 0 and result["has_comments"] == 0.0
    assert result["modern_constructs_per_line"] == 0.0


def test_transform_appends_features_without_clashing_with_record_columns():
    frame = pd.DataFrame({"code": ["let a = 1;", "x = 2"], "language": ["javascript", "python"]})
    table = FeatureExtractor().transform(frame)
    features = [c for c in table.columns if c not in frame.columns]
    assert len(features) >= 20
    assert not set(features) & set(RECORD_COLUMNS)
    assert table[features].notna().all().all()
