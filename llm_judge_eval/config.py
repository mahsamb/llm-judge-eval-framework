"""Configuration for evaluation runs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = PACKAGE_ROOT / "prompts"


def resolve_api_key(api_key: str | None, env_var: str = "OPENAI_API_KEY") -> str | None:
    """Prefer explicit api_key, then environment variable."""
    return api_key or os.environ.get(env_var)


@dataclass
class LLMJudgeConfig:
    provider: str = "mock"  # mock | openai
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    max_retries: int = 2
    score_scale: tuple[int, int] = (1, 5)

    def resolved_api_key(self) -> str | None:
        return resolve_api_key(self.api_key, self.api_key_env)


@dataclass
class EvaluationConfig:
    task: str
    output_dir: Path = Path("eval_outputs")
    run_automatic: bool = True
    run_human: bool = True
    run_llm_judge: bool = True
    llm_judge: LLMJudgeConfig = field(default_factory=LLMJudgeConfig)
    human_scores_path: Path | None = None
    primary_automatic_metric: str | None = None
    primary_human_metric: str = "overall"
    primary_llm_metric: str = "overall"
