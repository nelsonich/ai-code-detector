import pandas as pd

from ai_code_detector.data.preprocessor import CodePreprocessor


def _frame(rows):
    base = {"language": "python", "task_id": "t1", "label": "student", "source": "lesson",
            "dataset": "x", "created_at": None}
    return pd.DataFrame([base | r for r in rows])


def test_drops_short_and_duplicate_code():
    frame = _frame([
        {"code": "x = 1"},
        {"code": "def f():\n    return 42\n" * 3},
        {"code": "def f():\n\n    return 42\n" * 3},
        {"code": "print('a' * 100)" + " " * 40, "language": "cobol"},
    ])
    clean = CodePreprocessor(min_code_chars=20, languages=["python"]).run(frame)
    assert len(clean) == 1
    assert str(clean["language"].dtype) == "category"
