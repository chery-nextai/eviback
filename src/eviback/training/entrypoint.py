"""Parameter-complete VERL launcher with an inspectable dry-run mode."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace

from eviback.training.config import TrainingConfig


def build_command(config: TrainingConfig) -> list[str]:
    return [sys.executable, "-m", "verl.trainer.main_ppo", *config.as_verl_overrides()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--model")
    parser.add_argument("--train-data")
    parser.add_argument("--eval-data")
    parser.add_argument("--retriever-endpoint")
    parser.add_argument("--teacher-endpoint")
    parser.add_argument("--teacher-model")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--teacher-fallback-scale", type=float)
    parser.add_argument("--total-training-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args, verl_overrides = parser.parse_known_args(argv)
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
        overrides = {
            key: value
            for key, value in {
                "model": args.model,
                "train_data": args.train_data,
                "eval_data": args.eval_data,
                "retriever_endpoint": args.retriever_endpoint,
                "teacher_endpoint": args.teacher_endpoint,
                "teacher_model": args.teacher_model,
                "output_dir": args.output_dir,
                "seed": args.seed,
                "group_size": args.group_size,
                "teacher_fallback_scale": args.teacher_fallback_scale,
                "total_training_steps": args.total_training_steps,
            }.items()
            if value is not None
        }
        config = replace(config, **overrides)
    else:
        config = TrainingConfig(
            model=args.model or "",
            train_data=args.train_data or "",
            eval_data=args.eval_data or "",
            retriever_endpoint=args.retriever_endpoint or "",
            teacher_endpoint=args.teacher_endpoint or "",
            teacher_model=args.teacher_model or "",
            output_dir=args.output_dir or "",
            seed=args.seed if args.seed is not None else 42,
            group_size=args.group_size if args.group_size is not None else 8,
            teacher_fallback_scale=(
                args.teacher_fallback_scale if args.teacher_fallback_scale is not None else 0.1
            ),
            total_training_steps=(
                args.total_training_steps if args.total_training_steps is not None else 79
            ),
        )
    command = [*build_command(config), *verl_overrides]
    if args.dry_run or config.dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "retriever_endpoint": config.retriever_endpoint,
                    "teacher_endpoint": config.teacher_endpoint,
                    "teacher_model": config.teacher_model,
                    "teacher_fallback_scale": config.teacher_fallback_scale,
                    "inference_uses_teacher": False,
                },
                indent=2,
            )
        )
        return 0
    environment = {**os.environ, **config.runtime_environment()}
    return subprocess.run(command, check=False, env=environment).returncode



if __name__ == "__main__":
    raise SystemExit(main())