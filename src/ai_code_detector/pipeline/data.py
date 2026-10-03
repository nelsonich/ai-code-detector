"""Data stages: collect every source into one table and clean it."""

from ai_code_detector.config import PROJECT_ROOT, Config
from ai_code_detector.data.loader import DatasetLoader
from ai_code_detector.data.preprocessor import CodePreprocessor
from ai_code_detector.data.sources import (
    CodenetSource,
    DataSource,
    GeneratedSource,
    ProgpediaSource,
    WebdprocSource,
)
from ai_code_detector.pipeline import INTERIM_RECORDS, INTERIM_TASKS, PROCESSED_CLEAN


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
    if (PROJECT_ROOT / settings["generated"]["dir"]).exists():
        sources.append(GeneratedSource(PROJECT_ROOT / settings["generated"]["dir"]))
    return sources


def load(config: Config) -> None:
    """Read records and tasks of all sources into ``data/interim``."""
    sources = build_sources(config)
    loader = DatasetLoader(sources)
    records, tasks = loader.load(), loader.load_tasks()
    DatasetLoader.save(records, config.interim_dir / INTERIM_RECORDS)
    DatasetLoader.save(tasks, config.interim_dir / INTERIM_TASKS)

    print(f"sources: {[s.name for s in sources]}")
    if records.empty:
        print("loaded 0 records")
        return
    print(f"records: {records.groupby(['dataset', 'label']).size().to_dict()}")
    print(f"tasks: {tasks.groupby('dataset').size().to_dict()}")
    orphans = records.loc[~records["task_id"].isin(tasks["task_id"])]
    if not orphans.empty:
        by_kind = orphans.groupby(["dataset", "source"]).size().to_dict()
        print(f"warning: {len(orphans)} records have no task text: {by_kind}")


def preprocess(config: Config) -> None:
    """Clean the interim table into ``data/processed``."""
    frame = DatasetLoader.read(config.interim_dir / INTERIM_RECORDS)
    clean = CodePreprocessor(**config.preprocessing).run(frame)
    DatasetLoader.save(clean, config.processed_dir / PROCESSED_CLEAN)
    print(f"kept {len(clean)} of {len(frame)} records")


STAGES = {"load": load, "preprocess": preprocess}
