"""Phase 3: generate Stage 2 training queries with an LLM, remove test near-copies, split 80/20.

  python scripts/gen_train_data.py --model qwen/qwen3-32b --extra-body '{"reasoning": {"enabled": false}}' \
      --cache /content/drive/MyDrive/hirag_gen_cache.jsonl --out /content/drive/MyDrive/hirag_stage2/train_queries.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hirag.data import load_mcpbench          # noqa: E402
from hirag.dense import Embedder              # noqa: E402
from hirag.llm import LLMClient               # noqa: E402
from hirag.traindata import dedupe_and_split, generate   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen/qwen3-32b")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--extra-body", default='{"reasoning": {"enabled": false}}')
    ap.add_argument("--n", type=int, default=8, help="queries per tool (paper: 5-10)")
    ap.add_argument("--temperature", type=float, default=0.8, help="some variety in the generated queries")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--min-interval", type=float, default=0.5)
    ap.add_argument("--limit-tools", type=int, default=None, help="quick test on the first N tools")
    ap.add_argument("--embed-model", default="BAAI/bge-large-en")
    ap.add_argument("--embed-backend", default="st", choices=["st", "onnx"])
    ap.add_argument("--near-dup", type=float, default=0.95, help="drop if cosine to any test query >= this")
    ap.add_argument("--cache", default="data/gen_cache.jsonl")
    ap.add_argument("--out", default="data/stage2/train_queries.json")
    a = ap.parse_args()

    bench = load_mcpbench()
    llm = LLMClient(a.model, provider=a.provider, base_url=a.base_url, cache_path=a.cache,
                    temperature=a.temperature, max_tokens=a.max_tokens,
                    extra_body=json.loads(a.extra_body), min_interval=a.min_interval)
    items = generate(bench, llm, a.n, a.limit_tools)
    test_texts = [q.text for q in bench.single + bench.multi]
    split = dedupe_and_split(items, test_texts, Embedder(a.embed_model, a.embed_backend), a.near_dup)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"model": a.model, "n_per_tool": a.n, **split}, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("saved", a.out)
    for it in split["train"][:5]:
        print("  example:", it["tool_id"], "<-", it["query"])


if __name__ == "__main__":
    main()
