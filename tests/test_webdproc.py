import json
from datetime import datetime
from zoneinfo import ZoneInfo

from ai_code_detector.data.sources.webdproc import WebdprocSource


def _dump(root, name, rows):
    (root / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")


def test_loads_samples_from_navicat_export(tmp_path):
    _dump(tmp_path, "samples_lessons", [{
        "id": "1:css", "code": ".a { color: red }", "language": "css", "source": "lesson",
        "task_id": "lesson:7", "author_id": "abc", "created_at": "5/3/2026 09:07:01",
        "updated_at": "None", "meta_course_id": 1,
    }])
    _dump(tmp_path, "tasks_lessons", [])

    [record] = WebdprocSource(tmp_path).load()

    assert record.record_id == "webdproc:lesson:1:css"
    assert record.task_id == "webdproc:lesson:7"
    assert record.created_at == datetime(2026, 3, 5, 9, 7, 1, tzinfo=ZoneInfo("Asia/Yerevan"))
    assert record.label_status == "unverified"
    assert record.extra == {"meta_course_id": 1}


def test_tasks_prefer_english_and_attach_sorted_tests(tmp_path):
    _dump(tmp_path, "tasks_lessons", [{
        "task_id": "lesson:7", "course_id": "1", "course_title": "HTML", "order": "2",
        "title": "Intro", "description_html": "<p>Lecture</p>", "homework": "None",
    }])
    _dump(tmp_path, "tasks_challenges", [{
        "task_id": "challenge:3", "language_mode": "python", "difficulty": "easy",
        "execution_mode": "function", "evaluation_mode": "auto",
        "function_entry_names": "None",
        "title_am": "Գումար", "description_am": "<p>am</p>",
        "title_en": "Sum", "description_en": "<p>Add two numbers</p>",
        "title_ru": "None", "description_ru": "None",
    }])
    _dump(tmp_path, "tasks_challenge_tests", [
        {"task_id": "challenge:3", "input_data": "2 3", "expected_output": "5",
         "is_hidden": "1", "sort_order": "1"},
        {"task_id": "challenge:3", "input_data": "1 1", "expected_output": "2",
         "is_hidden": "0", "sort_order": "0"},
    ])

    tasks = {t.task_id: t for t in WebdprocSource(tmp_path).tasks()}

    lesson = tasks["webdproc:lesson:7"]
    assert lesson.statement == "<p>Lecture</p>" and lesson.language is None
    challenge = tasks["webdproc:challenge:3"]
    assert challenge.title == "Sum"
    assert challenge.language == "python"
    assert challenge.extra["function_entry_names"] is None
    assert [t["input"] for t in challenge.extra["tests"]] == ["1 1", "2 3"]
    assert challenge.extra["tests"][1]["is_hidden"] is True


def test_web_challenge_has_no_single_language(tmp_path):
    _dump(tmp_path, "tasks_challenges", [{
        "task_id": "challenge:9", "language_mode": "web",
        "title_en": "Page", "description_en": "<p>Build a page</p>",
    }])

    [task] = WebdprocSource(tmp_path).tasks()

    assert task.language is None
    assert task.extra["tests"] == []
