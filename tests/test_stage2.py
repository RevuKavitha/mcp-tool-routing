"""Phase 3/4 checks. The Stage 2 test needs torch (run it on Colab); the data tests run anywhere."""
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hirag.traindata import dedupe_and_split, parse_queries   # noqa: E402


def test_parse_queries():
    txt = '<think>hm</think>```json\n["Find papers about LLMs since 2023", "Download arXiv paper 2401.12345", 7, "hi"]\n```'
    assert parse_queries(txt) == ["Find papers about LLMs since 2023", "Download arXiv paper 2401.12345"]
    assert parse_queries("no list here") == []


def test_dedupe_and_split():
    items = [{"query": f"query number {i} for tool", "tool_id": f"t{i % 2}", "service": "s", "type": "x"}
             for i in range(10)]
    items.append(dict(items[0]))                                       # exact duplicate
    items.append({"query": "THE TEST QUERY", "tool_id": "t0", "service": "s", "type": "x"})  # copy of a test query
    out = dedupe_and_split(items, ["the test query"], embedder=None)
    assert len(out["train"]) + len(out["val"]) == 10
    assert len(out["val"]) == 2                                        # 20% of 5 per tool, 2 tools


class FakeEmbedder:
    """Deterministic random vector per text; identical text -> identical vector."""
    def encode(self, texts, is_query=False):
        out = []
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=32)
            out.append(v / np.linalg.norm(v))
        return np.vstack(out).astype(np.float32)


def test_stage2_trains_and_reranks():
    pytest.importorskip("torch")
    from hirag.data import load_mcpbench
    from hirag.stage2 import Stage2Config, Stage2Features, Stage2Reranker, train_stage2

    bench = load_mcpbench()
    emb = FakeEmbedder()
    feats = Stage2Features(bench, emb)
    assert feats.tools.shape[0] == len(bench.services) and feats.tools.shape[2] == 32
    # query = the tool summary itself (+ noise copies) -> learnable
    texts = [t.summary for t in bench.tools]
    svc = [t.service for t in bench.tools]
    q = emb.encode(texts)
    tr = q + np.random.default_rng(0).normal(scale=0.05, size=q.shape).astype(np.float32)
    cfg = Stage2Config(hidden=64, gate_hidden=64, score_hidden=32, epochs=8, batch_size=32, lr=2e-3)
    model, hist = train_stage2(feats, tr, svc, q, svc, cfg, device="cpu", log=lambda *_: None)
    assert len(hist) == 8 and np.isfinite(hist[-1]["loss"])
    assert max(h["val_service_at1"] for h in hist) > 30.0          # random would be ~2.6%
    rr = Stage2Reranker(model, feats, emb)
    cands = list(bench.services)[:5]
    out = rr.rerank(bench.tools[0].summary, cands)
    assert sorted(out) == sorted(cands)
    path = os.path.join(os.path.dirname(__file__), "_tmp_stage2.pt")
    rr.save(path)
    rr2 = Stage2Reranker.load(path, bench, emb, device="cpu")
    assert rr2.rerank(bench.tools[0].summary, cands) == out
    os.remove(path)
