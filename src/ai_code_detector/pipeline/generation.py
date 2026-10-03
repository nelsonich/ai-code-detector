"""Generation stage: ask AI models to solve the tasks that the human code solves.

Not part of ``run``: it calls paid and rate-limited APIs. It needs ``load`` to have
written ``records.parquet`` and ``tasks.parquet``; run ``load`` again afterwards to
include the generated code.
"""

from ai_code_detector.config import PROJECT_ROOT, Config
from ai_code_detector.data.loader import DatasetLoader
from ai_code_detector.generation.clients import build_generator
from ai_code_detector.generation.prompts import PromptStyle
from ai_code_detector.generation.runner import GenerationRunner, plan_jobs
from ai_code_detector.pipeline import INTERIM_RECORDS, INTERIM_TASKS


def generate(config: Config, limit_tasks: int | None, generator_names: list[str] | None) -> None:
    """Generate solutions for up to ``limit_tasks`` tasks (all when None) with each generator."""
    settings = config.generation
    names = generator_names or list(settings["generators"])
    generators = {
        name: build_generator(name, settings["generators"][name], settings["max_tokens"])
        for name in names
    }
    records = DatasetLoader.read(config.interim_dir / INTERIM_RECORDS)
    tasks = DatasetLoader.read(config.interim_dir / INTERIM_TASKS)
    styles = [PromptStyle(style) for style in settings["styles"]]

    jobs = plan_jobs(records, tasks, names, styles, limit_tasks, settings["seed"])
    print(f"planned {len(jobs)} jobs for {len(jobs) // max(len(names), 1)} tasks: {names}")
    runner = GenerationRunner(generators, tasks, PROJECT_ROOT / settings["output_dir"],
                              settings["max_statement_chars"])
    for name, counts in runner.run(jobs).items():
        print(f"{name}: {counts}")
