"""LLM-as-judge evaluator with pluggable providers."""

from __future__ import annotations

import json
import re
from pathlib import Path

from llm_judge_eval.config import LLMJudgeConfig, PROMPTS_DIR
from llm_judge_eval.evaluators.base import BaseEvaluator
from llm_judge_eval.schemas import EvalMethod, EvalRecord, ScoreDetail, TaskType


class LLMJudgeEvaluator(BaseEvaluator):
    PROMPT_FILES = {
        TaskType.QA: "qa_judge.txt",
        TaskType.SUMMARIZATION: "summarization_judge.txt",
        TaskType.CLASSIFICATION: "classification_judge.txt",
    }

    def __init__(self, config: LLMJudgeConfig | None = None):
        self.config = config or LLMJudgeConfig()
        self._prompts = self._load_prompts()

    def _load_prompts(self) -> dict[TaskType, str]:
        prompts: dict[TaskType, str] = {}
        for task, filename in self.PROMPT_FILES.items():
            path = PROMPTS_DIR / filename
            prompts[task] = path.read_text(encoding="utf-8")
        return prompts

    def evaluate(self, records: list[EvalRecord]) -> list[ScoreDetail]:
        scores: list[ScoreDetail] = []
        for record in records:
            prompt = self._build_prompt(record)
            response = self._call_llm(prompt, record)
            parsed = self._parse_response(response)
            scores.append(
                ScoreDetail(
                    record_id=record.id,
                    method=EvalMethod.LLM_JUDGE,
                    metric="overall",
                    score=float(parsed["score"]),
                    rationale=parsed.get("rationale"),
                    raw_response=response,
                )
            )
        return scores

    def _build_prompt(self, record: EvalRecord) -> str:
        template = self._prompts[record.task]
        low, high = self.config.score_scale
        return template.format(
            input_text=record.input_text,
            context=record.context or "(none)",
            prediction=record.prediction,
            reference=record.reference or "(none)",
            score_min=low,
            score_max=high,
        )

    def _call_llm(self, prompt: str, record: EvalRecord) -> str:
        if self.config.provider == "mock":
            return self._mock_judge(record)
        if self.config.provider == "openai":
            return self._openai_judge(prompt)
        raise ValueError(f"Unknown LLM provider: {self.config.provider}")

    def _mock_judge(self, record: EvalRecord) -> str:
        """Deterministic heuristic judge for offline demos."""
        prediction = record.prediction
        reference = record.reference or ""
        pred_norm = prediction.lower().strip()
        ref_norm = reference.lower().strip()
        low, high = self.config.score_scale

        if not ref_norm:
            score = (low + high) / 2
            rationale = "No reference available; assigned mid-scale score."
        elif pred_norm == ref_norm:
            score = high
            rationale = "Prediction matches reference exactly."
        elif pred_norm in ref_norm or ref_norm in pred_norm:
            score = high - 1
            rationale = "Partial overlap with reference."
        else:
            overlap = len(set(pred_norm.split()) & set(ref_norm.split()))
            if overlap >= 2:
                score = (low + high) / 2 + 0.5
                rationale = "Some lexical overlap with reference."
            else:
                score = low + 1
                rationale = "Low overlap with reference."

        score = max(low, min(high, score))
        return json.dumps({"score": score, "rationale": rationale})

    def _openai_judge(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc

        api_key = self.config.resolved_api_key()
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key in LLMJudgeConfig."
            )

        client = OpenAI(api_key=api_key)
        last_error: Exception | None = None
        for _ in range(self.config.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert NLP evaluator. Respond with JSON only: "
                            '{"score": <number>, "rationale": "<short explanation>"}',
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
        raise RuntimeError("OpenAI judge failed after retries") from last_error

    @staticmethod
    def _parse_response(response: str) -> dict:
        response = response.strip()
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
        raise ValueError(f"Could not parse LLM judge response: {response[:200]}")
