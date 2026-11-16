"""Phase 4: Hi-RAG Stage 2, type-aware hierarchical re-ranking (paper Eq. 6-11).

For a query q and a candidate service s (from Stage 1's S_cand):
  h_q, h_s, h_tau, h_ti = W x            shared projection of frozen BGE vectors (query, service, type, tools)
  beta   = sigmoid(MLP_gate([h_q ; h_tau ; h_q * h_tau]))                         Eq. 6  domain gate
  e_i    = LeakyReLU(a^T [h_q ; h_ti]),  alpha = softmax_i(e_i)                     Eq. 7  tool attention
  h_agg  = sum_i alpha_i W_v h_ti                                                   Eq. 8
  h'_s   = LayerNorm(h_s + gamma * h_agg)                                           Eq. 9  residual fusion
  Score  = beta * MLP(h'_s * h_q)                                                   Eq. 10
Loss (Eq. 11) = InfoNCE over 1 positive + 7 negative services (temperature T)
                + lambda_g * BCE(beta, 1[type of candidate == type of gold service]).
Paper settings: frozen bge-large-en, AdamW lr 2e-4, batch 64, 30 epochs, T = 0.5, lambda_g = 0.3, ~5.6M params.
The authors released no Stage 2 training code or weights, so this is written from the paper.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import MCPBench


@dataclass
class Stage2Config:
    hidden: int = 1024
    gate_hidden: int = 1024
    score_hidden: int = 512
    leaky_slope: float = 0.2
    temperature: float = 0.5
    lambda_gate: float = 0.3
    n_neg: int = 7
    hard_neg: int = 4           # of the 7 negatives, how many are "hard" (most similar other services)
    lr: float = 2e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    epochs: int = 30
    dropout: float = 0.1
    seed: int = 0


class Stage2Model(nn.Module):
    def __init__(self, d_in: int, cfg: Stage2Config):
        super().__init__()
        d = cfg.hidden
        self.cfg = cfg
        self.W = nn.Linear(d_in, d, bias=False)              # shared projection
        self.Wv = nn.Linear(d, d, bias=False)                # Eq. 8
        self.a = nn.Linear(2 * d, 1, bias=False)             # Eq. 7
        self.gate = nn.Sequential(nn.Linear(3 * d, cfg.gate_hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
                                  nn.Linear(cfg.gate_hidden, 1))                  # Eq. 6
        self.gamma = nn.Parameter(torch.tensor(1.0))         # Eq. 9
        self.ln = nn.LayerNorm(d)
        self.score = nn.Sequential(nn.Linear(d, cfg.score_hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
                                   nn.Linear(cfg.score_hidden, 1))                # Eq. 10

    def forward(self, q, svc, typ, tools, tool_mask):
        """q [B,D]; svc, typ [B,C,D]; tools [B,C,T,D]; tool_mask [B,C,T] (True = real tool).
        Returns score [B,C] and beta [B,C]."""
        hq = self.W(q)[:, None, :]                            # [B,1,d]
        hs, ht = self.W(svc), self.W(typ)                     # [B,C,d]
        hti = self.W(tools)                                   # [B,C,T,d]
        hqC = hq.expand_as(hs)
        beta = torch.sigmoid(self.gate(torch.cat([hqC, ht, hqC * ht], -1))).squeeze(-1)
        e = F.leaky_relu(self.a(torch.cat([hqC[:, :, None, :].expand_as(hti), hti], -1)).squeeze(-1),
                         self.cfg.leaky_slope)
        e = e.masked_fill(~tool_mask, float("-inf"))
        alpha = torch.softmax(e, -1)
        agg = (alpha[..., None] * self.Wv(hti)).sum(-2)       # [B,C,d]
        hs2 = self.ln(hs + self.gamma * agg)
        score = beta * self.score(hs2 * hqC).squeeze(-1)
        return score, beta


class Stage2Features:
    """Frozen BGE vectors for services, types and tools, plus padded tool tensors per service."""

    def __init__(self, bench: MCPBench, embedder):
        self.bench = bench
        self.services = list(bench.services)
        self.svc_index = {s: i for i, s in enumerate(self.services)}
        types = bench.types
        self.type_index = {t: i for i, t in enumerate(types)}
        tool_ids = [t.tool_id for t in bench.tools]
        tool_vec = embedder.encode([t.summary for t in bench.tools])
        svc_vec = embedder.encode([bench.services[s].title for s in self.services])
        type_vec = embedder.encode([t.replace("-", " ").replace("_", " ") for t in types])
        tid_index = {t: i for i, t in enumerate(tool_ids)}
        T = max(len(bench.services[s].tools) for s in self.services)
        D = tool_vec.shape[1]
        tools = np.zeros((len(self.services), T, D), dtype=np.float32)
        mask = np.zeros((len(self.services), T), dtype=bool)
        for i, s in enumerate(self.services):
            for j, tid in enumerate(bench.services[s].tools):
                tools[i, j] = tool_vec[tid_index[tid]]
                mask[i, j] = True
        self.D = D
        self.svc_vec = torch.tensor(svc_vec, dtype=torch.float32)
        self.svc_type = torch.tensor([self.type_index[bench.services[s].type] for s in self.services])
        self.type_vec = torch.tensor(type_vec, dtype=torch.float32)
        self.tools = torch.tensor(tools)
        self.mask = torch.tensor(mask)
        # service "centroid" (mean of its tool vectors) -> used to pick hard negatives
        cen = (self.tools * self.mask[..., None]).sum(1) / self.mask.sum(1, keepdim=True)
        self.centroid = F.normalize(cen, dim=-1)

    def batch(self, svc_idx: torch.Tensor, device):
        """svc_idx [B,C] -> tensors for Stage2Model.forward (minus the query)."""
        return (self.svc_vec[svc_idx].to(device), self.type_vec[self.svc_type[svc_idx]].to(device),
                self.tools[svc_idx].to(device), self.mask[svc_idx].to(device))


def _make_candidates(qv: torch.Tensor, gold: torch.Tensor, feats: Stage2Features, cfg: Stage2Config,
                     rng: random.Random) -> torch.Tensor:
    """[B,1+n_neg] service indices; column 0 is the gold service."""
    sims = qv @ feats.centroid.T                              # [B,S]
    S = sims.shape[1]
    rows = []
    for b in range(qv.shape[0]):
        g = int(gold[b])
        order = [int(i) for i in torch.argsort(sims[b], descending=True) if int(i) != g]
        hard = order[: cfg.hard_neg]
        rest = [i for i in range(S) if i != g and i not in hard]
        rows.append([g] + hard + rng.sample(rest, cfg.n_neg - len(hard)))
    return torch.tensor(rows)


def train_stage2(feats: Stage2Features, train_q: np.ndarray, train_svc: Sequence[str],
                 val_q: np.ndarray, val_svc: Sequence[str], cfg: Stage2Config = Stage2Config(),
                 device: str | None = None, log=print):
    """Returns (best model, history). Model selection = best validation Service@1 over all services."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    model = Stage2Model(feats.D, cfg).to(device)
    log(f"Stage 2 parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    Xq = torch.tensor(train_q, dtype=torch.float32)
    Y = torch.tensor([feats.svc_index[s] for s in train_svc])
    best, best_state, hist = -1.0, None, []
    for ep in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xq))
        tot = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            qv, gold = Xq[idx], Y[idx]
            cand = _make_candidates(qv, gold, feats, cfg, rng)
            score, beta = model(qv.to(device), *feats.batch(cand, device))
            nce = F.cross_entropy(score / cfg.temperature, torch.zeros(len(idx), dtype=torch.long, device=device))
            same_type = (feats.svc_type[cand] == feats.svc_type[gold][:, None]).float().to(device)
            gate = F.binary_cross_entropy(beta.clamp(1e-6, 1 - 1e-6), same_type)
            loss = nce + cfg.lambda_gate * gate
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        acc = evaluate_all_services(model, feats, val_q, val_svc, device)
        hist.append({"epoch": ep, "loss": tot / len(Xq), "val_service_at1": acc})
        log(f"epoch {ep:2d}  loss {tot / len(Xq):.4f}  val Service@1 (all services) {acc:.1f}%")
        if acc > best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def score_services(model: Stage2Model, feats: Stage2Features, qv: np.ndarray, svc_idx: List[List[int]],
                   device: str | None = None) -> List[np.ndarray]:
    """Scores for a variable-length candidate list per query."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for q, cands in zip(qv, svc_idx):
        if not cands:
            out.append(np.zeros(0))
            continue
        c = torch.tensor([cands])
        s, _ = model(torch.tensor(q[None], dtype=torch.float32, device=device), *feats.batch(c, device))
        out.append(s[0].cpu().numpy())
    return out


def evaluate_all_services(model, feats, qv, svc, device) -> float:
    if len(qv) == 0:
        return 0.0
    allc = [list(range(len(feats.services)))] * len(qv)
    scores = score_services(model, feats, qv, allc, device)
    hit = [int(np.argmax(s)) == feats.svc_index[g] for s, g in zip(scores, svc)]
    return 100.0 * float(np.mean(hit))


class Stage2Reranker:
    """Re-rank Stage 1's candidate services: S_cand -> services sorted by Stage 2 score."""

    def __init__(self, model: Stage2Model, feats: Stage2Features, embedder):
        self.model, self.feats, self.emb = model, feats, embedder
        self._qcache: Dict[str, np.ndarray] = {}

    def precompute(self, queries: List[str]):
        todo = [q for q in queries if q not in self._qcache]
        if todo:
            for q, v in zip(todo, self.emb.encode(todo, is_query=True)):
                self._qcache[q] = v

    def rerank(self, query: str, services: List[str]) -> List[str]:
        if len(services) <= 1:
            return list(services)
        self.precompute([query])
        idx = [self.feats.svc_index[s] for s in services]
        s = score_services(self.model, self.feats, self._qcache[query][None], [idx])[0]
        order = np.argsort(-s, kind="stable")
        return [services[i] for i in order]

    def save(self, path: str):
        torch.save({"state": self.model.state_dict(), "cfg": self.model.cfg.__dict__, "d_in": self.feats.D}, path)

    @classmethod
    def load(cls, path: str, bench: MCPBench, embedder, device: str | None = None):
        ck = torch.load(path, map_location="cpu")
        model = Stage2Model(ck["d_in"], Stage2Config(**ck["cfg"]))
        model.load_state_dict(ck["state"])
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model.to(device).eval(), Stage2Features(bench, embedder), embedder)
