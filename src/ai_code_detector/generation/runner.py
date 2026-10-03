"""Planning and running generation jobs; results are appended to JSONL files and resumable."""

import hashlib
import json
from collections.abc import Iterable
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
              seed: int = 42) -> list[Job]:
    """Plan one job per eligible task and generator.

    A task is eligible when it has human code and a statement. Its languages are the
    languages of that human code, so AI and human samples solve the same task in the
    same languages. Web tasks (HTML, CSS, JS) are written together like in the editor;
    other tasks are solved by humans in one language, so each job asks for one of the
    task's languages. With ``limit_tasks`` the tasks are drawn round-robin across
    (dataset, language) groups, so a small trial still covers every kind of task.
    Language and style of each job are fixed by a hash of task and generator, so
    re-planning never changes already finished jobs.
    """
    human = records[records["label"] == "human"]
    languages = human.groupby("task_id")["language"].agg(lambda s: tuple(sorted(set(s))))
    eligible = tasks[tasks["task_id"].isin(languages.index)
                     & tasks["statement"].fillna("").str.strip().astype(bool)]
    task_ids = _select(eligible, languages, limit_tasks, seed)

    jobs = []
    for task_id in task_ids:
        for generator in generators:
            key = f"{task_id}|{generator}"
            job_languages = _job_languages(languages[task_id], key)
            style = styles[_stable_index(key, len(styles))]
            job_id = f"{generator}|{task_id}|{'+'.join(job_languages)}|{style}"
            jobs.append(Job(job_id, task_id, job_languages, generator, style))
    return jobs


def _job_languages(task_languages: tuple[str, ...], key: str) -> tuple[str, ...]:
    if set(task_languages) <= WEB_LANGUAGES:
        return task_languages
    return (task_languages[_stable_index(f"{key}|language", len(task_languages))],)


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


class GenerationRunner:
    """Run jobs against generators and append each result to ``<generator>.jsonl``."""

    def __init__(self, generators: dict[str, Generator], tasks: pd.DataFrame,
                 output_dir: Path, max_statement_chars: int) -> None:
        """Keep generators, the task texts and where to write results."""
        self.generators = generators
        self.tasks = tasks.set_index("task_id")
        self.output_dir = output_dir
        self.max_statement_chars = max_statement_chars

    def done_job_ids(self) -> set[str]:
        """Ids of jobs already written by any earlier run."""
        done = set()
        for path in self.output_dir.glob("*.jsonl"):
            with open(path, encoding="utf-8") as handle:
                done.update(json.loads(line)["job_id"] for line in handle if line.strip())
        return done

    def run(self, jobs: list[Job]) -> dict[str, dict[str, int]]:
        """Run every job not done yet; return per-generator counts of results."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        done = self.done_job_ids()
        summary = {name: {"done_before": 0, "written": 0, "failed": 0, "skipped": 0}
                   for name in self.generators}
        failures_in_row = dict.fromkeys(self.generators, 0)

        for job in jobs:
            counts = summary[job.generator]
            if job.job_id in done:
                counts["done_before"] += 1
                continue
            if failures_in_row[job.generator] >= MAX_CONSECUTIVE_FAILURES:
                counts["skipped"] += 1
                continue
            try:
                self._write(job, self._generate(job))
            except GenerationError as error:
                counts["failed"] += 1
                failures_in_row[job.generator] += 1
                print(f"  failed {job.job_id}: {str(error)[:200]}")
                continue
            counts["written"] += 1
            failures_in_row[job.generator] = 0
        return summary

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
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _write(self, job: Job, result: dict) -> None:
        row = asdict(job) | {"style": str(job.style), "languages": list(job.languages)} | result
        with open(self.output_dir / f"{job.generator}.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
