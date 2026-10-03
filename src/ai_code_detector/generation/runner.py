"""Planning and running generation jobs; results are appended to JSONL files and resumable."""

import hashlib
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ai_code_detector.generation.clients import GenerationError, Generator
from ai_code_detector.generation.extract import extract_code
from ai_code_detector.generation.prompts import WEB_LANGUAGES, PromptStyle, build_prompt

# A generator is skipped for the rest of the run after this many failures in a row,
# which usually means an exhausted quota rather than a bad task.
MAX_CONSECUTIVE_FAILURES = 5


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens and the most a generator may spend in total (None: no limit)."""

    input_per_million: float = 0.0
    output_per_million: float = 0.0
    budget_usd: float | None = None

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Cost in USD of the given token counts."""
        return (input_tokens * self.input_per_million
                + output_tokens * self.output_per_million) / 1_000_000


@dataclass(frozen=True)
class Job:
    """One solution to generate: a task, its languages, a generator and a style."""

    job_id: str
    task_id: str
    languages: tuple[str, ...]
    generator: str
    style: PromptStyle


def _stable_index(key: str, size: int) -> int:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % size


def plan_jobs(records: pd.DataFrame, tasks: pd.DataFrame, generators: Iterable[str],
              styles: list[PromptStyle], limit_tasks: int | None = None,
              seed: int = 42, exhaustive_datasets: Iterable[str] = ()) -> list[Job]:
    """Plan generation jobs for every eligible task and generator.

    A task is eligible when it has human code and a statement. Its languages are the
    languages of that human code, so AI and human samples solve the same task in the
    same languages. Web tasks (HTML, CSS, JS) are written together like in the editor;
    other tasks are solved by humans in one language, so a job asks for one language.

    By default a task gets one job per generator, with language and style fixed by a
    hash of task and generator. Tasks of ``exhaustive_datasets`` get a job for every
    style and every language instead, to balance the classes where human code is
    plentiful. The default job is one of the exhaustive ones, so switching a dataset to
    exhaustive never repeats finished work. With ``limit_tasks`` the tasks are drawn
    round-robin across (dataset, language) groups, so a small trial covers every kind.
    """
    human = records[records["label"] == "human"]
    languages = human.groupby("task_id")["language"].agg(lambda s: tuple(sorted(set(s))))
    eligible = tasks[tasks["task_id"].isin(languages.index)
                     & tasks["statement"].fillna("").str.strip().astype(bool)]
    task_ids = _select(eligible, languages, limit_tasks, seed)
    exhaustive = set(eligible.loc[eligible["dataset"].isin(set(exhaustive_datasets)), "task_id"])

    jobs = []
    for task_id in task_ids:
        for generator in generators:
            key = f"{task_id}|{generator}"
            if task_id in exhaustive:
                combos = [(langs, style) for langs in _language_options(languages[task_id])
                          for style in styles]
            else:
                options = _language_options(languages[task_id])
                combos = [(options[_stable_index(f"{key}|language", len(options))],
                           styles[_stable_index(key, len(styles))])]
            for job_languages, style in combos:
                job_id = f"{generator}|{task_id}|{'+'.join(job_languages)}|{style}"
                jobs.append(Job(job_id, task_id, job_languages, generator, style))
    return jobs


def _language_options(task_languages: tuple[str, ...]) -> list[tuple[str, ...]]:
    if set(task_languages) <= WEB_LANGUAGES:
        return [task_languages]
    return [(language,) for language in task_languages]


def _select(eligible: pd.DataFrame, languages: pd.Series, limit: int | None,
            seed: int) -> list[str]:
    if limit is None:
        return sorted(eligible["task_id"])
    frame = eligible.assign(group=[
        f"{dataset}|{languages[task_id][0]}"
        for dataset, task_id in zip(eligible["dataset"], eligible["task_id"])
    ]).sample(frac=1, random_state=seed)
    queues = [list(group["task_id"]) for _, group in frame.groupby("group")]
    picked: list[str] = []
    while len(picked) < limit and any(queues):
        for queue in queues:
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
    return picked


def read_results(output_dir: Path) -> list[dict]:
    """All generation results written so far."""
    rows = []
    for path in sorted(output_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def spend_by_generator(rows: list[dict], pricing: dict[str, Pricing]) -> dict[str, float]:
    """Estimated USD spent per generator, from the token usage stored with each result."""
    spent: dict[str, float] = {}
    for row in rows:
        usage = row.get("usage") or {}
        price = pricing.get(row["generator"], Pricing())
        cost = price.cost(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        spent[row["generator"]] = spent.get(row["generator"], 0.0) + cost
    return spent


class GenerationRunner:
    """Run jobs against generators and append each result to ``<generator>.jsonl``.

    A generator stops for the rest of the run once its estimated spend reaches its
    budget, or after repeated failures in a row; finished jobs are never repeated.
    """

    def __init__(self, generators: dict[str, Generator], tasks: pd.DataFrame,
                 output_dir: Path, max_statement_chars: int,
                 pricing: dict[str, Pricing] | None = None) -> None:
        """Keep generators, the task texts, where to write results and their prices."""
        self.generators = generators
        self.tasks = tasks.set_index("task_id")
        self.output_dir = output_dir
        self.max_statement_chars = max_statement_chars
        self.pricing = pricing or {}

    def done_job_ids(self) -> set[str]:
        """Ids of jobs already written by any earlier run."""
        return {row["job_id"] for row in read_results(self.output_dir)}

    def run(self, jobs: list[Job]) -> dict[str, dict[str, float]]:
        """Run every job not done yet; return per-generator counts and spend.

        Generators run in parallel, one thread each; every generator writes only its
        own file and keeps its own counters, so the threads share no mutable state.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = read_results(self.output_dir)
        done = {row["job_id"] for row in rows}
        spent = spend_by_generator(rows, self.pricing)
        queues: dict[str, list[Job]] = {name: [] for name in self.generators}
        for job in jobs:
            queues[job.generator].append(job)

        with ThreadPoolExecutor(max_workers=max(len(queues), 1)) as pool:
            futures = {
                name: pool.submit(self._run_generator, name, queue, done,
                                  spent.get(name, 0.0))
                for name, queue in queues.items()
            }
            return {name: future.result() for name, future in futures.items()}

    def _run_generator(self, name: str, jobs: list[Job], done: set[str],
                       spent: float) -> dict[str, float]:
        counts = {"done_before": 0, "written": 0, "failed": 0, "skipped": 0}
        price = self.pricing.get(name, Pricing())
        failures_in_row = 0
        for job in jobs:
            if job.job_id in done:
                counts["done_before"] += 1
                continue
            over_budget = price.budget_usd is not None and spent >= price.budget_usd
            if failures_in_row >= MAX_CONSECUTIVE_FAILURES or over_budget:
                counts["skipped"] += 1
                continue
            try:
                result = self._generate(job)
            except GenerationError as error:
                counts["failed"] += 1
                failures_in_row += 1
                print(f"  failed {job.job_id}: {str(error)[:200]}", flush=True)
                continue
            self._write(job, result)
            spent += price.cost(result["usage"]["input_tokens"], result["usage"]["output_tokens"])
            counts["written"] += 1
            failures_in_row = 0
            if counts["written"] % 50 == 0:
                print(f"  {name}: {counts['written']} written, ${spent:.2f} spent", flush=True)
        return counts | {"spent_usd": round(spent, 4)}

    def _generate(self, job: Job) -> dict:
        task = self.tasks.loc[job.task_id]
        prompt = build_prompt(job.task_id, task["title"], task["statement"],
                              task["statement_format"], list(job.languages), job.style,
                              self.max_statement_chars)
        generator = self.generators[job.generator]
        answer = generator.generate(prompt)
        return {
            "model": generator.model,
            "served_model": answer.served_model,
            "statement_truncated": prompt.truncated,
            "codes": extract_code(answer.text, list(job.languages)),
            "raw_answer": answer.text,
            "usage": {"input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens},
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _write(self, job: Job, result: dict) -> None:
        row = asdict(job) | {"style": str(job.style), "languages": list(job.languages)} | result
        with open(self.output_dir / f"{job.generator}.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
