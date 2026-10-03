"""Common interface for data sources."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ai_code_detector.data.schema import CodeRecord, TaskRecord


class DataSource(ABC):
    """A provider of code samples and their tasks in the unified record format.

    Adding a new dataset means implementing this interface in one module;
    the rest of the pipeline stays untouched.
    """

    name: str

    @abstractmethod
    def load(self) -> Iterator[CodeRecord]:
        """Yield all records available in this source."""

    def tasks(self) -> Iterator[TaskRecord]:
        """Yield the tasks solved by this source's records. Sources without tasks yield none."""
        yield from ()
