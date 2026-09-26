"""Data sources. Each source converts its raw files into ``CodeRecord`` objects."""

from ai_code_detector.data.sources.base import DataSource
from ai_code_detector.data.sources.progpedia import ProgpediaSource
from ai_code_detector.data.sources.webdproc import WebdprocSource

__all__ = ["DataSource", "ProgpediaSource", "WebdprocSource"]
