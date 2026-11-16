"""Phase 4: train Hi-RAG Stage 2 and evaluate re-ranking on MCPBench (Stage 1 S_cand -> Stage 2).

  python scripts/train_stage2.py --data /content/drive/MyDrive/hirag_stage2/train_queries.json \
      --out /content/drive/MyDrive/hirag_stage2/stage2.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                        # noqa: E402

from hirag.data import load_mcpbench                       # noqa: E402
from hirag.dense import Embedder                           # noqa: E402
from hirag.metrics import format_table                     # noqa: E402
from hirag.stage1 import Stage1, Stage1Config              # noqa: E402
from hirag.stage2 import (Stage2Config, Stage2Features, Stage2Reranker,   # noqa: E402
                          train_stage2)


def evaluate(bench, stage1, reranker=None):
    """Service@1 (single), Service R@3 = all gold services in top 3 (multi), candidate recall."""
    res = {}
    for split, qs, k in (("single", bench.single, 1), ("multi", bench.multi, 3)):
        hits, cand_ok, sizes = [], [], []
        for q in qs:
            cands = list(stage1.run(q.text).candidates)
            order = reranker.rerank(q.text, cands) if reranker else cands
            hits.append(all(g in order[:k] for g in q.gold_services))
            cand_ok.append(all(g in cands for g in q.gold_services))
            sizes.append(len(cands))
        res[split] = (100 * np.mean(hits), 100 * np.mean(cand_ok), np.mean(sizes))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="data/stage2/stage2.pt")
    ap.add_argument("--embed-model", default="BAAI/bge-large-en")
    ap.add_argument("--embed-backend", default="st", choices=["st", "onnx"])
    ap.add_argument("--tokenizer", default="jieba", help="Stage 1 BM25 tokenizer (jieba = authors)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    bench = load_mcpbench()
    emb = Embedder(a.embed_model, a.embed_backend)
    data = json.load(open(a.data))
    tr, va = data["train"], data["val"]
    print(f"train {len(tr)} / val {len(va)} generated queries")
    feats = Stage2Features(bench, emb)
    cfg = Stage2Config(hidden=a.hidden, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, seed=a.seed)
    trq = emb.encode([x["query"] for x in tr], is_query=True)
    vaq = emb.encode([x["query"] for x in va], is_query=True)
    model, hist = train_stage2(feats, trq, [x["service"] for x in tr], vaq, [x["service"] for x in va], cfg)
    rr = Stage2Reranker(model, feats, emb)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rr.save(a.out)
    print("saved", a.out)

    stage1 = Stage1(bench, emb, Stage1Config(tokenizer=a.tokenizer))
    allq = [q.text for q in bench.single + bench.multi]
    stage1.precompute_dense(allq)
    rr.precompute(allq)
    base, ours = evaluate(bench, stage1), evaluate(bench, stage1, rr)
    C = ["Setting", "Service@1 (single)", "Service R@3 (multi)"]
    rows = [dict(zip(C, r)) for r in (
        ["Stage 1 only (S_cand order)", base["single"][0], base["multi"][0]],
        ["Stage 1 + Stage 2 (Hi-RAG)", ours["single"][0], ours["multi"][0]],
        ["Ceiling: gold service in S_cand", base["single"][1], base["multi"][1]])]
    print(format_table(rows, C))
    out = {"stage1": base, "hirag": ours, "history": hist, "config": cfg.__dict__, "embed_model": a.embed_model,
           "n_train": len(tr), "n_val": len(va)}
    p = os.path.splitext(a.out)[0] + "_results.json"
    json.dump(out, open(p, "w"), indent=1)
    print("saved", p)


if __name__ == "__main__":
    main()
