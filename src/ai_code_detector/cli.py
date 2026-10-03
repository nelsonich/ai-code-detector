"""Command line interface: run the pipeline end to end or one stage at a time."""

import argparse

from ai_code_detector.config import Config
from ai_code_detector.pipeline import analysis, data

STAGES = data.STAGES | analysis.STAGES


def main() -> None:
    """Parse the command line and run the requested stage(s)."""
    parser = argparse.ArgumentParser(prog="ai-code-detector")
    parser.add_argument("stage", choices=[*STAGES, "run"], help="'run' executes every stage")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    args = parser.parse_args()

    config = Config.load(args.config) if args.config else Config.load()
    config.ensure_dirs()
    for name in STAGES if args.stage == "run" else [args.stage]:
        print(f"== {name}")
        STAGES[name](config)


if __name__ == "__main__":
    main()
