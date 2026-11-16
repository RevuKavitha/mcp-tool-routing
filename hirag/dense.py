"""Dense bi-encoder retrieval with BGE embeddings.

Backends:
- "st":   sentence-transformers (use on Colab/Kaggle), e.g. model="BAAI/bge-large-en" (the paper's model)
- "onnx": a local ONNX export folder containing tokenizer.json and onnx/model.onnx (CPU, for testing)

Embeddings are L2-normalised, so cosine similarity = dot product and the ranking equals
the authors' FAISS IndexFlatL2 ranking (distance^2 = 2 - 2*cos).
Embeddings are cached on disk so the GPU work is done once.
"""
from __future__ import annotations

import hashlib
import os
from typing import List

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "emb_cache")
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model: str = "BAAI/bge-large-en", backend: str = "st",
                 query_instruction: str = "", batch_size: int = 32, cache: bool = True):
        self.model_name, self.backend = model, backend
        self.query_instruction, self.batch_size, self.cache = query_instruction, batch_size, cache
        self._model = None

    # ---------- backends ----------
    def _load(self):
        if self._model is not None:
            return
        if self.backend == "st":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        elif self.backend == "onnx":
            import onnxruntime as ort
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(os.path.join(self.model_name, "tokenizer.json"))
            tok.enable_truncation(512)
            sess = ort.InferenceSession(os.path.join(self.model_name, "onnx", "model.onnx"))
            self._model = (tok, sess, [i.name for i in sess.get_inputs()])
        else:
            raise ValueError(f"unknown backend {self.backend!r}")

    def _encode_raw(self, texts: List[str]) -> np.ndarray:
        self._load()
        if self.backend == "st":
            return np.asarray(self._model.encode(texts, batch_size=self.batch_size,
                                                 normalize_embeddings=True, show_progress_bar=False))
        tok, sess, names = self._model
        out = []
        for t in texts:  # one at a time: simple and fine for a few hundred texts
            e = tok.encode(t)
            feed = {"input_ids": np.array([e.ids]), "attention_mask": np.array([e.attention_mask])}
            if "token_type_ids" in names:
                feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
            v = sess.run(None, feed)[0][0, 0]          # CLS pooling (as BGE)
            out.append(v / np.linalg.norm(v))
        return np.vstack(out)

    # ---------- public ----------
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if is_query and self.query_instruction:
            texts = [self.query_instruction + t for t in texts]
        if not self.cache:
            return self._encode_raw(texts)
        key = hashlib.sha1(("\n".join(texts) + "|" + self.model_name).encode()).hexdigest()[:16]
        path = os.path.join(CACHE_DIR, f"{os.path.basename(self.model_name.rstrip('/'))}_{key}.npy")
        if os.path.exists(path):
            return np.load(path)
        vecs = self._encode_raw(texts)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, vecs)
        return vecs


class DenseRetriever:
    def __init__(self, docs: List[str], embedder: Embedder):
        self.emb = embedder
        self.doc_vecs = embedder.encode(docs)

    def scores_many(self, queries: List[str]) -> np.ndarray:
        """Cosine similarity matrix [n_queries, n_docs]."""
        return self.emb.encode(queries, is_query=True) @ self.doc_vecs.T

    @staticmethod
    def rank_from_scores(s: np.ndarray, top_n: int | None = None) -> List[int]:
        order = [int(i) for i in np.argsort(-s, kind="stable")]
        return order[:top_n] if top_n else order
