"""Command line interface: run the pipeline end to end or one stage at a time."""

import argparse

from ai_code_detector.config import Config
from ai_code_detector.pipeline import analysis, data
from ai_code_detector.pipeline.generation import generate

STAGES = data.STAGES | analysis.STAGES


def main() -> None:
    """Parse the command line and run the requested stage(s)."""
    parser = argparse.ArgumentParser(prog="ai-code-detector")
    parser.add_argument("stage", choices=[*STAGES, "run", "generate"],
                        help="'run' executes every stage; 'generate' calls AI APIs separately")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--tasks", type=int, default=None,
                        help="generate: number of tasks to use (default: all eligible)")
    parser.add_argument("--generators", default=None,
                        help="generate: comma-separated generator names (default: all configured)")
    args = parser.parse_args()

    config = Config.load(args.config) if args.config else Config.load()
    config.ensure_dirs()
    if args.stage == "generate":
        names = args.generators.split(",") if args.generators else None
        generate(config, args.tasks, names)
        return
    for name in STAGES if args.stage == "run" else [args.stage]:
        print(f"== {name}")
        STAGES[name](config)


if __name__ == "__main__":
    main()
