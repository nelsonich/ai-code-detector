from ai_code_detector.data.loader import DatasetLoader
from ai_code_detector.data.schema import CodeRecord, Label, LabelStatus, SourceKind, TaskRecord
from ai_code_detector.data.sources.base import DataSource


class _FakeSource(DataSource):
    name = "fake"

    def __init__(self, extra):
        self.extra = extra

    def load(self):
        for i, extra in enumerate(self.extra):
            yield CodeRecord(
                record_id=f"r{i}", code="x = 1", language="python", label=Label.HUMAN,
                label_status=LabelStatus.VERIFIED, source=SourceKind.EXTERNAL,
                task_id="t1", dataset=self.name, extra=extra,
            )

    def tasks(self):
        task = TaskRecord(task_id="t1", dataset=self.name, title="T", statement="Do it",
                          statement_format="text", extra={"tests": [{"input": "1"}]})
        yield task
        yield task


def test_tasks_are_deduplicated_by_id():
    tasks = DatasetLoader([_FakeSource([{}])]).load_tasks()
    assert list(tasks["task_id"]) == ["t1"]


def test_save_and_read_keep_heterogeneous_extra(tmp_path):
    loader = DatasetLoader([_FakeSource([{"a": 1}, {"b": "x", "c": [1, 2]}, {}])])
    path = tmp_path / "records.parquet"

    DatasetLoader.save(loader.load(), path)
    frame = DatasetLoader.read(path)

    assert list(frame["extra"]) == [{"a": 1}, {"b": "x", "c": [1, 2]}, {}]

    DatasetLoader.save(loader.load_tasks(), tmp_path / "tasks.parquet")
    tasks = DatasetLoader.read(tmp_path / "tasks.parquet")
    assert tasks.loc[0, "extra"] == {"tests": [{"input": "1"}]}
