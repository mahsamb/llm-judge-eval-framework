# LLM Judge Evaluation Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Compare **human evaluation**, **LLM-as-judge**, and **automatic metrics** on NLP tasks (QA, summarization, classification). Built for research teams who want to validate whether LLM judges can replace or augment human annotators.

**Author:** Marzueh Babaali ([@mahsamb](https://github.com/mahsamb))

---

## Why this project?

In modern NLP evaluation, teams often use three approaches:

| Method | Strength | Weakness |
|--------|----------|----------|
| **Automatic metrics** (EM, ROUGE, accuracy) | Fast, cheap, reproducible | Miss paraphrases and nuance |
| **Human evaluation** | Gold standard quality | Slow, expensive, hard to scale |
| **LLM-as-judge** | Scalable, context-aware | Must be validated against humans |

This framework runs all three on the same dataset and reports **agreement statistics** (Pearson, Spearman, Cohen's kappa) so you can study whether LLM judges are trustworthy.

---

## Quick start

```bash
git clone https://github.com/mahsamb/llm-judge-eval-framework.git
cd llm-judge-eval-framework
pip install -e ".[dev]"
python run_eval_demo.py
```

Reports are written to `eval_outputs/`.

---

## Project structure

```
llm_judge_eval/
├── evaluators/          # automatic, human, LLM-as-judge
├── prompts/             # task-specific judge prompt templates
├── correlation/         # Pearson, Spearman, Cohen's kappa
├── reporting/           # CSV and text report generation
├── sample_data/         # demo datasets
├── runner.py            # main pipeline
└── cli.py               # command-line interface
run_eval_demo.py         # runs all 3 task demos
tests/                   # pytest suite
```

---

## Inputs

### 1. Predictions CSV (required)

| Column | Required | Description |
|--------|----------|-------------|
| `id` | Yes | Unique example ID |
| `input_text` | Yes | Question, document, or text |
| `prediction` | Yes | Model output |
| `reference` | Recommended | Gold answer / label |
| `context` | Optional | Passage (useful for QA) |

### 2. Human scores CSV (optional)

| Column | Required | Description |
|--------|----------|-------------|
| `id` | Yes | Must match predictions CSV |
| `overall` | Yes | Human score (e.g. 1–5) |
| `rationale` | Optional | Annotator explanation |

---

## Usage

### Python API

```python
from pathlib import Path
from llm_judge_eval.config import EvaluationConfig, LLMJudgeConfig
from llm_judge_eval.runner import EvaluationRunner

config = EvaluationConfig(
    task="qa",
    output_dir=Path("eval_outputs/my_run"),
    human_scores_path=Path("my_human_scores.csv"),
    llm_judge=LLMJudgeConfig(provider="mock"),  # or "openai"
)
result = EvaluationRunner(config).run("my_predictions.csv", run_name="my_run")
print(result.correlations)
```

### Command line

```bash
python -m llm_judge_eval.cli \
  --task qa \
  --dataset my_predictions.csv \
  --human-scores my_human_scores.csv \
  --output-dir eval_outputs/my_run
```

Skip human eval:

```bash
python -m llm_judge_eval.cli --task qa --dataset my_predictions.csv --skip-human
```

Real OpenAI LLM judge (use env var for the API key):

```bash
set OPENAI_API_KEY=sk-your-key-here
python -m llm_judge_eval.cli --task qa --dataset my_predictions.csv --llm-provider openai
```

---

## Outputs

Each run produces files in `eval_outputs/<run_name>/`:

| File | Description |
|------|-------------|
| `*_scores.csv` | Every score (method, metric, value) |
| `*_by_record.csv` | Side-by-side scores per example |
| `*_summary.csv` | Mean/std per method and metric |
| `*_correlations.csv` | Agreement between evaluation methods |
| `*_report.txt` | Human-readable summary |

---

## LLM judge modes

| Provider | Description |
|----------|-------------|
| `mock` | Free heuristic judge for demos and CI (default) |
| `openai` | Real GPT-based judge; requires `OPENAI_API_KEY` |

The mock judge uses simple string overlap rules. For research, always validate against `provider="openai"` (or add your own provider).

---

## Known limitations

- **Mock judge** is not a real LLM — use OpenAI mode for production research.
- **Automatic ROUGE** is a simplified n-gram overlap, not the full `rouge-score` library.
- **Classification F1** equals accuracy for single-label examples in this version.
- **Correlation across scales**: automatic scores are 0–1 while human/LLM scores are 1–5; interpret cross-method correlations cautiously.
- **Small demo datasets** (3–5 examples) are for illustration only — use 100+ examples for meaningful correlation analysis.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Citation

If you use this framework in research, please cite the repository:

```bibtex
@software{babaali2026llmjudgeeval,
  author = {Babaali, Marzueh},
  title = {LLM Judge Evaluation Framework},
  year = {2026},
  url = {https://github.com/mahsamb/llm-judge-eval-framework}
}
```
