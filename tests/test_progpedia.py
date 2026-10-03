from ai_code_detector.data.sources.progpedia import ProgpediaSource


def _write(root, rel, text="x = 1\n"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_keeps_latest_accepted_attempt_per_author_and_language(tmp_path):
    _write(tmp_path, "00000001/statement.md", "Sort numbers\n============\n\nTask")
    _write(tmp_path, "00000001/ACCEPTED/00007_00001/a.py", "first")
    _write(tmp_path, "00000001/ACCEPTED/00007_00002/a.py", "accepted")
    _write(tmp_path, "00000001/WRONG_ANSWER/00007_00003/a.py", "later but wrong")
    _write(tmp_path, "00000001/WRONG_ANSWER/00007_00004/A.java", "java only wrong")
    _write(tmp_path, "00000001/ACCEPTED/00007_00002/a_py.cpg.csv", "ignored")

    source = ProgpediaSource(tmp_path)
    records = {r.language: r for r in source.load()}

    assert set(records) == {"python", "java"}
    assert records["python"].code == "accepted"
    assert records["java"].extra["meta_status"] == "WRONG_ANSWER"
    assert records["python"].label_status == "verified"
    [task] = source.tasks()
    assert task.task_id == "progpedia:00000001"
    assert task.title == "Sort numbers"
    assert task.statement_format == "markdown"
