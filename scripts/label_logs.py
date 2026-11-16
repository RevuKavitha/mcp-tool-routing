"""Label the self-distillation log with the LLM's REAL decisions (not the gold tool).

Runs the fallback pipeline (Hi-RAG Stage 1 + Stage 2 + LLM, same settings as run_e2e.py) on the generated
'train' queries and stores, for each query, the first tool the LLM called. This is what a deployed router
would log. The 'val' split keeps its gold labels: it plays the role of a small audited set for calibration.

Colab (GPU), same cache as notebook 03, so interrupted runs resume for free:
  python scripts/label_logs.py --train-file /content/drive/MyDrive/hirag_stage2/train_queries.json \
      --stage2 /content/drive/MyDrive/hirag_stage2/stage2.pt --model qwen/qwen3-32b \
      --extra-body '{"reasoning": {"enabled": false}}' --cache /content/drive/MyDrive/hirag_llm_cache.jsonl \
      --out /content/drive/MyDrive/hirag_stage2/train_queries_llm_labels.json
Then: python scripts/run_distill.py --calib-file .../train_queries_llm_labels.json ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hirag.data import load_mcpbench                     # noqa: E402
from hirag.dense import Embedder                         # noqa: E402
from hirag.llm import PROMPT_EN, PROMPT_ZH, DailyLimitError, LLMClient   # noqa: E402
from hirag.selectors import make_selector                # noqa: E402
from hirag.stage1 import Stage1, Stage1Config            # noqa: E402
from hirag.stage2 import Stage2Reranker                  # noqa: E402
from hirag.tools_schema import functions_for_services    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--stage2", required=True, help="trained Stage 2 weights (stage2.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen/qwen3-32b")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--extra-body", default='{"reasoning": {"enabled": false}}')
    ap.add_argument("--embed-model", default="BAAI/bge-large-en")
    ap.add_argument("--prompt", default="zh", choices=["zh", "en"])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--min-interval", type=float, default=1.0)
    ap.add_argument("--cache", default="data/llm_cache.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    bench = load_mcpbench()
    d = json.load(open(a.train_file, encoding="utf-8"))
    train = d["train"][: a.limit]
    emb = Embedder(a.embed_model, "st")
    stage1 = Stage1(bench, emb, Stage1Config())
    stage1.precompute_dense([x["query"] for x in train])
    rer = Stage2Reranker.load(a.stage2, bench, emb)
    rer.precompute([x["query"] for x in train])
    select = make_selector("hirag", bench, stage1, None, rer)
    llm = LLMClient(a.model, provider=a.provider, cache_path=a.cache, extra_body=json.loads(a.extra_body),
                    max_tokens=a.max_tokens, min_interval=a.min_interval)
    prompt = PROMPT_ZH if a.prompt == "zh" else PROMPT_EN

    out, agree, t0 = [], 0, time.time()
    for n, x in enumerate(train, 1):
        try:
            res = llm.select_tools(x["query"], functions_for_services(bench, select(x["query"], 1)), prompt, 1)
        except DailyLimitError as e:
            print(f"daily limit at {n}: {e}\nre-run later; finished queries come from the cache")
            break
        called = [c.lower() for c in res.tool_calls]
        y = dict(x)
        y["gold_tool_id"] = x["tool_id"]
        y["tool_id"] = called[0] if called else None          # the LLM's decision (None = no tool called)
        out.append(y)
        agree += y["tool_id"] == x["tool_id"]
        if n % 50 == 0 or n == len(train):
            print(f"[{n}/{len(train)}] LLM label == gold: {100*agree/n:.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    logged = [y for y in out if y["tool_id"]]                  # a request with no tool call is not logged
    json.dump({**{k: v for k, v in d.items() if k not in ("train",)}, "train": logged,
               "label_source": f"LLM decisions ({a.model}, Hi-RAG top-1 service)",
               "llm_label_accuracy": 100 * agree / max(1, len(out))},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"saved {a.out}: {len(logged)} logged queries, LLM labels correct {100*agree/max(1,len(out)):.1f}%")


if __name__ == "__main__":
    main()
