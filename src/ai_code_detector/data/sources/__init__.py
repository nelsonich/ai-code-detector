"""Data sources. Each source converts its raw files into ``CodeRecord`` objects."""

from ai_code_detector.data.sources.base import DataSource
from ai_code_detector.data.sources.codenet import CodenetSource
from ai_code_detector.data.sources.droid import DroidSource
from ai_code_detector.data.sources.generated import GeneratedSource
from ai_code_detector.data.sources.progpedia import ProgpediaSource
from ai_code_detector.data.sources.webdproc import WebdprocSource

__all__ = ["CodenetSource", "DataSource", "DroidSource", "GeneratedSource", "ProgpediaSource",
           "WebdprocSource"]
