"""Project configuration loaded from a YAML file."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class Config:
    """Typed access to configuration sections with paths resolved against the project root."""

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    figures_dir: Path
    tables_dir: Path
    sources: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> "Config":
        """Read a YAML config and build a Config instance."""
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        paths = data["paths"]
        return cls(
            raw_dir=PROJECT_ROOT / paths["raw"],
            interim_dir=PROJECT_ROOT / paths["interim"],
            processed_dir=PROJECT_ROOT / paths["processed"],
            figures_dir=PROJECT_ROOT / paths["figures"],
            tables_dir=PROJECT_ROOT / paths["tables"],
            sources=data.get("sources", {}),
            preprocessing=data.get("preprocessing", {}),
            features=data.get("features", {}),
            generation=data.get("generation", {}),
        )

    def ensure_dirs(self) -> None:
        """Create all output directories if they are missing."""
        for directory in (self.interim_dir, self.processed_dir, self.figures_dir,
                          self.tables_dir):
            directory.mkdir(parents=True, exist_ok=True)
