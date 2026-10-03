"""Data stages: collect every source into one table and clean it."""

from ai_code_detector.config import PROJECT_ROOT, Config
from ai_code_detector.data.loader import DatasetLoader
from ai_code_detector.data.preprocessor import CodePreprocessor
from ai_code_detector.data.sources import (
    CodenetSource,
    DataSource,
    ProgpediaSource,
    WebdprocSource,
)
from ai_code_detector.pipeline import INTERIM_RECORDS, PROCESSED_CLEAN


def build_sources(config: Config) -> list[DataSource]:
    """Create the configured sources whose data directory exists."""
    settings = config.sources
    sources: list[DataSource] = []
    if (PROJECT_ROOT / settings["webdproc"]["dir"]).exists():
        sources.append(WebdprocSource(PROJECT_ROOT / settings["webdproc"]["dir"]))
    if (PROJECT_ROOT / settings["progpedia"]["dir"]).exists():
        sources.append(ProgpediaSource(PROJECT_ROOT / settings["progpedia"]["dir"]))
    codenet = settings["codenet"]
    if (PROJECT_ROOT / codenet["dir"]).exists():
        sources.append(CodenetSource(
            PROJECT_ROOT / codenet["dir"],
            per_language=codenet["per_language"],
            max_per_problem=codenet["max_per_problem"],
            seed=codenet["seed"],
        ))
    return sources


def load(config: Config) -> None:
    """Read all sources into ``data/interim``."""
    sources = build_sources(config)
    frame = DatasetLoader(sources).load()
    DatasetLoader.save(frame, config.interim_dir / INTERIM_RECORDS)
    counts = frame.groupby(["dataset", "label"]).size().to_dict() if len(frame) else {}
    print(f"loaded {len(frame)} records from {[s.name for s in sources]}: {counts}")


def preprocess(config: Config) -> None:
    """Clean the interim table into ``data/processed``."""
    frame = DatasetLoader.read(config.interim_dir / INTERIM_RECORDS)
    clean = CodePreprocessor(**config.preprocessing).run(frame)
    DatasetLoader.save(clean, config.processed_dir / PROCESSED_CLEAN)
    print(f"kept {len(clean)} of {len(frame)} records")


STAGES = {"load": load, "preprocess": preprocess}
