import pandas as pd

from ai_code_detector.features.language_model import normalize_layout
from ai_code_detector.pipeline.language_model import select


def test_normalize_layout_removes_editor_dependent_layout():
    code = "\n\nint main() {  \n\tint a = 1;\t\n\n\n\n\treturn a;\n}\n\n"
    assert normalize_layout(code) == "int main() {\n    int a = 1;\n\n    return a;\n}"


def test_normalize_layout_keeps_code_without_such_layout():
    code = "def f():\n    return 1"
    assert normalize_layout(code) == code


def test_select_takes_at_most_per_group_and_is_reproducible():
    frame = pd.DataFrame({
        "record_id": range(40),
        "language": ["python", "java"] * 20,
        "origin": "x",
        "label": ["ai"] * 20 + ["human"] * 20,
        "label_status": "verified",
    })
    first, second = select(frame, 3, seed=1), select(frame, 3, seed=1)
    assert len(first) == 12
    assert list(first["record_id"]) == list(second["record_id"])
    assert select(frame, 2, seed=1)["record_id"].isin(first["record_id"]).all()
