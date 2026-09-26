"""Loading records from all configured sources into one table."""

from pathlib import Path

import pandas as pd

from ai_code_detector.data.schema import records_to_frame
from ai_code_detector.data.sources.base import DataSource


class DatasetLoader:
    """Combine every data source into a single DataFrame in the unified schema."""

    def __init__(self, sources: list[DataSource]) -> None:
        """Keep the sources to read from."""
        self.sources = sources

    def load(self) -> pd.DataFrame:
        """Read all sources and return their records as one table."""
        frames = [records_to_frame(list(source.load())) for source in self.sources]
        return pd.concat(frames, ignore_index=True) if frames else records_to_frame([])

    @staticmethod
    def save(frame: pd.DataFrame, path: Path) -> None:
        """Persist a table as Parquet."""
        frame.to_parquet(path, index=False)

    @staticmethod
    def read(path: Path) -> pd.DataFrame:
        """Read a table saved by ``save``."""
        return pd.read_parquet(path)
