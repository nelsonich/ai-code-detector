"""Language-model stage: score a balanced sample of code with a code language model.

Scoring runs on the CPU and takes about a second per sample, so it is a separate,
resumable command (``ai-code-detector lm-score``) and works on a sample: up to
``per_group`` records per language, origin and label. Each sample is scored as written
and after ``normalize_layout``, to see which signals survive reformatting.
"""

import time

import pandas as pd

from ai_code_detector.config import Config
from ai_code_detector.data.loader import DatasetLoader
from ai_code_detector.features.language_model import LanguageModelScorer, normalize_layout
from ai_code_detector.pipeline import LM_SCORES, PROCESSED_FEATURES

SAVE_EVERY = 50
NORMALIZED = "_normalized"


def select(frame: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    """Pick up to ``per_group`` records per language, origin and label, reproducibly."""
    shuffled = frame.sample(frac=1, random_state=seed)
    return shuffled.groupby(["language", "origin", "label", "label_status"],
                            observed=True).head(per_group)


def score(config: Config) -> None:
    """Score the selected records that have no score yet, saving progress regularly."""
    settings = config.features["language_model"]
    frame = DatasetLoader.read(config.processed_dir / PROCESSED_FEATURES)
    selected = select(frame, settings["per_group"], settings["seed"])
    path = config.processed_dir / LM_SCORES
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    pending = selected[~selected["record_id"].isin(done.get("record_id", []))]
    print(f"selected {len(selected)} records, {len(pending)} still to score")
    if pending.empty:
        return

    scorer = LanguageModelScorer(settings["observer"], settings["performer"],
                                 settings["max_tokens"], settings.get("threads"))
    rows, started = [], time.monotonic()
    for i, (record_id, code) in enumerate(zip(pending["record_id"], pending["code"]), 1):
        raw = scorer.score(code)
        normalized = scorer.score(normalize_layout(code))
        rows.append({"record_id": record_id} | raw
                    | {f"{k}{NORMALIZED}": v for k, v in normalized.items()})
        if i % SAVE_EVERY == 0 or i == len(pending):
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_parquet(path, index=False)
            rows = []
            rate = (time.monotonic() - started) / i
            print(f"  {i}/{len(pending)} scored, {rate:.1f}s each, "
                  f"~{rate * (len(pending) - i) / 60:.0f} min left", flush=True)


def scored_features(config: Config) -> pd.DataFrame | None:
    """Feature table restricted to scored records, with the language-model columns."""
    path = config.processed_dir / LM_SCORES
    if not path.exists():
        return None
    frame = DatasetLoader.read(config.processed_dir / PROCESSED_FEATURES)
    scores = pd.read_parquet(path).drop(columns=[f"lm_tokens{NORMALIZED}"])
    return frame.merge(scores.dropna(), on="record_id")

