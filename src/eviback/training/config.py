"""Typed public training configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    train_data: str
    eval_data: str
    retriever_endpoint: str
    teacher_endpoint: str
    teacher_model: str
    output_dir: str
    seed: int = 42
    group_size: int = 8
    teacher_fallback_scale: float = 0.1
    total_training_steps: int = 79
    dry_run: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
        training = {
            key: os.path.expandvars(raw) if isinstance(raw, str) else raw
            for key, raw in dict(value.get("training") or value).items()
        }
        return cls(**training)

    def validate(self) -> None:
        for key in (
            "model",
            "train_data",
            "eval_data",
            "retriever_endpoint",
            "teacher_endpoint",
            "teacher_model",
            "output_dir",
        ):
            value = str(getattr(self, key)).strip()
            if not value:
                raise ValueError(f"{key} is required")
            if value.startswith("${"):
                raise ValueError(f"{key} contains an unresolved environment variable: {value}")
        if self.group_size <= 1:
            raise ValueError("group_size must be greater than one")
        if not 0 < self.teacher_fallback_scale <= 1:
            raise ValueError("teacher_fallback_scale must be in (0, 1]")

    def as_verl_overrides(self) -> list[str]:
        self.validate()
        reward_path = Path(__file__).with_name("verl_reward.py").resolve()
        package_root = Path(__file__).resolve().parents[3]
        tool_config = package_root / "configs" / "retriever" / "search_tool.yaml"
        return [
            f"actor_rollout_ref.model.path={self.model}",
            f"data.train_files={self.train_data}",
            f"data.val_files={self.eval_data}",
            "data.train_batch_size=64",
            "data.val_batch_size=8",
            "data.max_prompt_length=12000",
            "data.max_response_length=4096",
            f"data.seed={self.seed}",
            "actor_rollout_ref.actor.ppo_mini_batch_size=64",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
            "actor_rollout_ref.actor.optim.lr=1e-6",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
            "actor_rollout_ref.rollout.max_model_len=16096",
            f"actor_rollout_ref.rollout.n={self.group_size}",
            "algorithm.adv_estimator=grpo",
            "algorithm.norm_adv_by_std_in_grpo=true",
            "+algorithm.group_postnorm_advantage_scale_key=advantage_postnorm_scale",
            "reward_model.reward_manager=batch",
            f"custom_reward_function.path={reward_path}",
            "custom_reward_function.name=compute_eviback_reward_batch",
            f"+custom_reward_function.reward_kwargs.n_samples_per_prompt={self.group_size}",
            "+custom_reward_function.reward_kwargs.visible_top_m=5",
            f"+custom_reward_function.reward_kwargs.teacher_request.endpoint={self.teacher_endpoint}",
            f"+custom_reward_function.reward_kwargs.teacher_request.model={self.teacher_model}",
            "+custom_reward_function.reward_kwargs.teacher_request.temperature=0.0",
            "+custom_reward_function.reward_kwargs.teacher_request.top_p=1.0",
            "+custom_reward_function.reward_kwargs.teacher_request.max_tokens=512",
            "+custom_reward_function.reward_kwargs.partial_reward=0.1",
            "+custom_reward_function.reward_kwargs.gold_token_f1_bonus=0.1",
            f"+custom_reward_function.reward_kwargs.teacher_fallback_scale={self.teacher_fallback_scale}",
            "actor_rollout_ref.rollout.multi_turn.enable=true",
            f"actor_rollout_ref.rollout.multi_turn.tool_config_path={tool_config}",
            f"trainer.total_training_steps={self.total_training_steps}",
            f"trainer.default_local_dir={self.output_dir}",
        ]

    def runtime_environment(self) -> dict[str, str]:
        return {
            "EVIBACK_RETRIEVER_ENDPOINT": self.retriever_endpoint,
            "EVIBACK_TEACHER_ENDPOINT": self.teacher_endpoint,
            "EVIBACK_TEACHER_MODEL": self.teacher_model,
        }