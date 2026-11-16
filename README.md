# Hi-RAG reproduction and a self-distilling deterministic–LLM tool router

Code, generated data and per-request outputs for two papers:

1. **Reproducing Hi-RAG** (Tian & Zhou, *Expert Systems* 2026, doi:10.1111/exsy.70341) on MCPBench, with Qwen3-32B and Llama-3.3-70B-Instruct.
2. **A self-distilling deterministic–LLM router** that sends clear requests straight to a tool with no LLM call and escalates the rest to Hi-RAG + LLM.

MCPBench is **not copied** here: the authors' repository has no licence file, so `hirag/data.py` downloads it from https://github.com/TW-NLP/Hierarchical-RAG-MCP (commit d11c798) on first use.

## Layout

| Path | Contents |
|---|---|
| `hirag/` | Stage 1 (BM25 + dense + fusion), Stage 2 re-ranker, baselines, ToolBench loader, router, rule mining, risk-control calibration, self-distillation |
| `scripts/` | command-line entry points (Stage 1, Stage 2 data and training, end-to-end runs, router, distillation, ToolBench) |
| `router_extra/scripts_extra/` | extra router experiments: classifier baseline, hybrid, purity filter, worked examples, non-inferiority CIs, multi-step sub-tasks, figures |
| `notebooks/` | Colab notebooks 01–05 used for the GPU and LLM runs (outputs cleared) |
| `tests/` | unit tests (`pytest -q`) |
| `results/e2e/` | per-request end-to-end outputs for every method (Qwen3-32B, temperature 0) |
| `results/stage2_data/` | the 1,604 generated Stage 2 training requests and their LLM labels |
| `results/router_extra/` | non-inferiority results |

## Setup

```bash
pip install -r requirements.txt
pytest -q
```

LLM calls go through OpenRouter. Set the key as an environment variable (`LLM_API_KEY`), or in Colab as the secret `OPENROUTER_API_KEY`. Never put a key in code.

## Reproducing the main results

```bash
python scripts/run_stage1.py                                 # Table 2 (Stage 1)
python scripts/run_router.py --fallback-dir results/e2e \
       --calib-file results/stage2_data/train_queries.json   # router accuracy / LLM-call trade-off
python scripts/run_distill.py                                # self-distillation curve
```

The router evaluation reuses the saved per-request outputs in `results/e2e/`, so it needs no new LLM calls. See `router_extra/scripts_extra/README.md` for the additional experiments.
