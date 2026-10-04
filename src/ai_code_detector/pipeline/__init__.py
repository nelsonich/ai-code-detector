"""Pipeline stages. Each stage reads a table from disk and writes the next one.

``data``: raw sources -> unified records -> clean table.
``analysis``: clean table -> features -> statistics and figures.
"""

from pathlib import Path

INTERIM_RECORDS = Path("records.parquet")
INTERIM_TASKS = Path("tasks.parquet")
PROCESSED_CLEAN = Path("clean.parquet")
PROCESSED_FEATURES = Path("features.parquet")
LM_SCORES = Path("lm_scores.parquet")
