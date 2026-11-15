import pandas as pd

from ai_code_detector.data.preprocessor import (
    CodePreprocessor,
    lecture_examples,
    realign_cut_snippet,
)

LESSON = "webdproc:lesson:1"
EXAMPLE = "<!DOCTYPE html>\n<html>\n  <body>\n    <h1>Heading</h1>\n  </body>\n</html>"


def _frame(rows):
    base = {"language": "python", "task_id": "t1", "label": "human",
            "label_status": "unverified", "source": "lesson", "dataset": "x",
            "origin": None, "created_at": None, "author_id": "a", "generator": None,
            "prompt_style": None}
    return pd.DataFrame([base | r for r in rows])


def _tasks():
    escaped = EXAMPLE.replace("<", "&lt;").replace(">", "&gt;")
    return pd.DataFrame({
        "task_id": [LESSON, "progpedia:1"],
        "statement": [f'<p>Look:</p><pre class="language-markup"><code>{escaped}</code></pre>',
                      "<pre>not a lesson</pre>"],
    })


def test_drops_short_other_languages_and_duplicates():
    frame = _frame([
        {"code": "x = 1"},
        {"code": "def f():\n    return 42\n" * 3},
        {"code": "def f():\n\n    return 42\n" * 3},
        {"code": "DISPLAY 'HELLO WORLD FROM COBOL'.", "language": "cobol"},
    ])
    preprocessor = CodePreprocessor(min_code_chars=20, languages=["python"])
    clean = preprocessor.run(frame)
    assert len(clean) == 1
    assert str(clean["language"].dtype) == "category"
    assert [s["removed"] for s in preprocessor.report["steps"]] == [1, 1, 1]


def test_normalizes_line_endings_and_bom():
    frame = _frame([{"code": "﻿def f():\r\n    return 1\r\n" * 3}])
    clean = CodePreprocessor(min_code_chars=10, languages=["python"]).run(frame)
    assert "\r" not in clean.loc[0, "code"] and not clean.loc[0, "code"].startswith("﻿")


def test_lecture_examples_are_unescaped_and_only_from_lessons():
    examples = lecture_examples(_tasks())
    assert list(examples) == [LESSON]
    assert examples[LESSON][0].startswith("<!DOCTYPEhtml>")


def test_drops_copies_of_lecture_examples_in_both_classes():
    own = "<html>\n<body>\n<h2>My own page about cats and dogs</h2>\n<p>Text</p>\n</body>\n</html>"
    # Close to lesson 1's example, but lesson 2 has no examples, so it is not a copy.
    similar = EXAMPLE.replace("Heading", "Title")
    frame = _frame([
        {"code": EXAMPLE, "language": "html", "task_id": LESSON},
        {"code": EXAMPLE.replace("Heading", "Headings"), "language": "html", "task_id": LESSON,
         "label": "ai", "label_status": "verified", "generator": "g", "prompt_style": "standard"},
        {"code": own, "language": "html", "task_id": LESSON},
        {"code": similar, "language": "html", "task_id": "webdproc:lesson:2"},
    ])
    preprocessor = CodePreprocessor(min_code_chars=10, languages=["html"])
    clean = preprocessor.run(frame, _tasks())

    assert sorted(clean["code"]) == sorted([own, similar])
    assert preprocessor.report["steps"][-1] == {
        "step": "drop_lecture_copies", "removed": 2,
        "removed_by_label": {"human": 1, "ai": 1}, "remaining": 2}


def test_missing_values_are_made_explicit_and_reported():
    frame = _frame([{"code": "print('hello world')", "author_id": None}])
    preprocessor = CodePreprocessor(min_code_chars=5, languages=["python"])
    clean = preprocessor.run(frame)

    assert clean.loc[0, "generator"] == "none" and clean.loc[0, "prompt_style"] == "none"
    assert clean.loc[0, "author_id"] == "unknown"
    assert preprocessor.report["missing_before"]["generator"] == 1
    assert "generator" not in preprocessor.report["missing_after"]
    assert preprocessor.report["missing_after"]["created_at"] == 1
    assert preprocessor.report["balance"] == {"python": {"human": 1}}


def test_realigns_snippets_cut_out_of_a_class():
    braces = "bool Equals(X x)\n        {\n            return x;\n        }"
    assert realign_cut_snippet(braces, "csharp") == "bool Equals(X x)\n{\n    return x;\n}"
    python = "def f(self):\n        if x:\n            return 1\n\n        return 2"
    expected = "def f(self):\n    if x:\n        return 1\n\n    return 2"
    assert realign_cut_snippet(python, "python") == expected
    tabs = "def f(self):\n\t\treturn 1"
    assert realign_cut_snippet(tabs, "python") == "def f(self):\n\treturn 1"


def test_leaves_ordinary_code_alone():
    for code, language in (
            ("def f():\n    if x:\n        return 1\n    return 2", "python"),
            ("int f() {\n    return 1;\n}", "cpp"),
            ("class A:\n    def f(self):\n        g(1,\n      2)\n        return 1", "python"),
            ("def f(self):\n    d = {\n        'a': 1,\n    }", "python"),
            ("app.module()\n  .run(function() {\n    go();\n  });", "javascript"),
            ("    indented_first_line()\n        more()", "python"),
            ("x = 1\ny = 2", "python")):
        assert realign_cut_snippet(code, language) == code
