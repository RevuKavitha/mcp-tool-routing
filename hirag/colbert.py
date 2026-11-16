"""ColBERT-RAG baseline: token-level MaxSim with a BGE encoder (mirrors the authors' colbert_search.py).

S(q, d) = sum over query tokens of max over doc tokens of <E_q[i], E_d[j]>, with
L2-normalised last-layer token embeddings (no ColBERT-specific training, as in the authors' code).
Needs a GPU for reasonable speed.
"""
from __future__ import annotations

from typing import List

import numpy as np


class ColBERTRetriever:
    def __init__(self, docs: List[str], model_name: str = "BAAI/bge-large-en-v1.5",
                 max_query_len: int = 64, max_doc_len: int = 256, batch_size: int = 16):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_q, self.max_d = max_query_len, max_doc_len
        self.docs = []
        for i in range(0, len(docs), batch_size):
            self.docs.extend(self._encode(docs[i:i + batch_size], self.max_d))

    def _encode(self, texts: List[str], max_len: int):
        torch = self.torch
        with torch.no_grad():
            b = self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
            h = torch.nn.functional.normalize(self.model(**b).last_hidden_state, p=2, dim=-1)
            mask = b["attention_mask"].bool()
            return [h[i][mask[i]] for i in range(h.size(0))]

    def scores(self, query: str) -> np.ndarray:
        q = self._encode([query], self.max_q)[0]
        with self.torch.no_grad():
            return np.array([float((q @ d.T).max(dim=1).values.sum()) for d in self.docs])

    def rank(self, query: str, top_n: int = 20) -> List[int]:
        s = self.scores(query)
        return [int(i) for i in np.argsort(s)[::-1][:top_n]]   # authors: argsort(...)[::-1]
