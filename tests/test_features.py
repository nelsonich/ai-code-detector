from ai_code_detector.features.extractor import FeatureExtractor

MARKERS = {"python": ["#"], "javascript": ["//", "/*"]}


def test_basic_counts():
    code = "def f():\n    return 1\n\n# done\n"
    result = FeatureExtractor(MARKERS).extract(code, "python")
    assert result["n_lines"] == 4
    assert result["blank_line_ratio"] == 0.25
    assert result["comment_line_ratio"] == 0.25
    assert result["avg_indent"] == 4 / 3


def test_unknown_language_has_no_comments():
    result = FeatureExtractor(MARKERS).extract("// x\n", "css")
    assert result["comment_line_ratio"] == 0
