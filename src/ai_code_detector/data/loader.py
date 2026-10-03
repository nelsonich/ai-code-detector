"""Loading records and tasks from all configured sources into tables."""

import json
from pathlib import Path

import pandas as pd

from ai_code_detector.data.schema import records_to_frame, tasks_to_frame
from ai_code_detector.data.sources.base import DataSource

# Free-form metadata differs between sources and rows; it is stored as JSON text so
# Parquet does not need one fixed nested type for it.
JSON_COLUMNS = ("extra",)


class DatasetLoader:
    """Combine every data source into single DataFrames in the unified schema."""

    def __init__(self, sources: list[DataSource]) -> None:
        """Keep the sources to read from."""
        self.sources = sources

    def load(self) -> pd.DataFrame:
        """Read all sources and return their code records as one table."""
        frames = [records_to_frame(list(source.load())) for source in self.sources]
        return pd.concat(frames, ignore_index=True) if frames else records_to_frame([])

    def load_tasks(self) -> pd.DataFrame:
        """Read all sources and return their tasks as one table, one row per task id."""
        frames = [tasks_to_frame(list(source.tasks())) for source in self.sources]
        tasks = pd.concat(frames, ignore_index=True) if frames else tasks_to_frame([])
        return tasks.drop_duplicates("task_id").reset_index(drop=True)

    @staticmethod
    def save(frame: pd.DataFrame, path: Path) -> None:
        """Persist a table as Parquet."""
        out = frame.copy()
        for column in JSON_COLUMNS:
            if column in out:
                out[column] = out[column].map(lambda v: json.dumps(v, default=str))
        out.to_parquet(path, index=False)

    @staticmethod
    def read(path: Path) -> pd.DataFrame:
        """Read a table saved by ``save``."""
        frame = pd.read_parquet(path)
        for column in JSON_COLUMNS:
            if column in frame:
                frame[column] = frame[column].map(json.loads)
        return frame
