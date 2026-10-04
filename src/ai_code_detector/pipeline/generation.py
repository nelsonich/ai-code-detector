"""Generation stage: ask AI models to solve the tasks that the human code solves.

Not part of ``run``: it calls paid and rate-limited APIs. It needs ``load`` to have
written ``records.parquet`` and ``tasks.parquet``; run ``load`` again afterwards to
include the generated code.
"""

from collections import Counter
from pathlib import Path

from ai_code_detector.config import PROJECT_ROOT, Config
from ai_code_detector.data.loader import DatasetLoader
from ai_code_detector.generation.clients import build_generator
from ai_code_detector.generation.prompts import PromptStyle
from ai_code_detector.generation.runner import (
    GenerationRunner,
    Job,
    Pricing,
    plan_jobs,
    read_results,
    spend_by_generator,
)
from ai_code_detector.pipeline import INTERIM_RECORDS, INTERIM_TASKS


def _pricing(settings: dict) -> dict[str, Pricing]:
    return {
        name: Pricing(float(entry.get("price_input", 0.0)), float(entry.get("price_output", 0.0)),
                      entry.get("budget_usd"))
        for name, entry in settings["generators"].items()
    }


def _plan(config: Config, names: list[str], limit_tasks: int | None) -> list[Job]:
    settings = config.generation
    records = DatasetLoader.read(config.interim_dir / INTERIM_RECORDS)
    tasks = DatasetLoader.read(config.interim_dir / INTERIM_TASKS)
    styles = [PromptStyle(style) for style in settings["styles"]]
    return plan_jobs(records, tasks, names, styles, limit_tasks, settings["seed"],
                     settings.get("exhaustive_datasets", ()))


def generate(config: Config, limit_tasks: int | None, generator_names: list[str] | None,
             dry_run: bool = False) -> None:
    """Generate solutions for up to ``limit_tasks`` tasks (all when None) with each generator."""
    settings = config.generation
    names = generator_names or list(settings["generators"])
    output_dir = PROJECT_ROOT / settings["output_dir"]
    jobs = _plan(config, names, limit_tasks)
    print(f"planned {len(jobs)} jobs for {len({job.task_id for job in jobs})} tasks: {names}")
    if dry_run:
        report(output_dir, jobs, _pricing(settings))
        return

    generators = {
        name: build_generator(name, settings["generators"][name], settings["max_tokens"])
        for name in names
    }
    tasks = DatasetLoader.read(config.interim_dir / INTERIM_TASKS)
    caps = {name: int(entry["max_results"]) for name, entry in settings["generators"].items()
            if entry.get("max_results") is not None}
    runner = GenerationRunner(generators, tasks, output_dir, settings["max_statement_chars"],
                              _pricing(settings), caps)
    for name, counts in runner.run(jobs).items():
        print(f"{name}: {counts}")


def report(output_dir: Path, jobs: list[Job], pricing: dict[str, Pricing]) -> None:
    """Print pending jobs, spend so far and the projected cost of the pending jobs."""
    rows = read_results(output_dir)
    done = {row["job_id"] for row in rows}
    pending = Counter(job.generator for job in jobs if job.job_id not in done)
    spent = spend_by_generator(rows, pricing)
    for name in sorted({job.generator for job in jobs}):
        measured = [row for row in rows if row["generator"] == name and row.get("usage")]
        price = pricing.get(name, Pricing())
        average = (sum(price.cost(r["usage"]["input_tokens"], r["usage"]["output_tokens"])
                       for r in measured) / len(measured)) if measured else None
        projected = f"${average * pending[name]:.2f}" if average is not None else "unknown"
        print(f"{name}: pending {pending[name]}, spent ${spent.get(name, 0.0):.2f}, "
              f"avg/solution {'$%.4f' % average if average is not None else 'unknown'} "
              f"(from {len(measured)} measured), projected for pending {projected}, "
              f"budget {price.budget_usd}")
