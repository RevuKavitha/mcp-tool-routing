"""Paired bootstrap 95% CI of the accuracy difference (router - fallback) on the 201 single-service
requests, for the self-distilled rule router at eps = 0.15 and 0.10 (same setting as Table 1)."""
import os, sys, json, math, random, glob
ROOT = os.environ.get('ROUTER_PACK', '.')
sys.path.insert(0, os.path.join(ROOT, 'hirag-repro-f'))
import numpy as np
from hirag.data import load_mcpbench
from hirag.registry import build_registry
from hirag.distill import Distiller, with_learned
from hirag.router import HybridRouter
from hirag.sparse import BM25Retriever
from hirag.calibrate import tau_by_shape
bench = load_mcpbench(); reg0 = build_registry(bench); ids = [t.tool_id for t in bench.tools]
bm = BM25Retriever([t.summary for t in bench.tools], "clean")
retr = lambda text: ids[bm.rank(text, 1)[0]]
d = json.load(open(os.path.join(ROOT, 'hirag_stage2', 'train_queries_llm_labels.json')))
train = [(x['query'], x['tool_id']) for x in d['train']]; val = [(x['query'], x['tool_id']) for x in d['val']]
router = HybridRouter(with_learned(reg0, Distiller(2, 0.6).fit(train)), tau=math.inf, retriever=retr)
rnd = random.Random(7); svc = {t.tool_id: t.service for t in bench.tools}; joins = [", then ", " and then ", ". After that, ", ", and after that "]
composed = []
while len(composed) < 400:
    (q1, t1), (q2, t2) = rnd.sample(val, 2)
    if svc[t1] == svc[t2]: continue
    j = rnd.choice(joins); q2 = q2[0].lower() + q2[1:] if not j.endswith(". After that, ") else q2
    composed.append((q1.rstrip(" .?!") + j + q2, [t1, t2]))
cal = []
for q, tool in val:
    subs, conf = router.tier1(q); cal.append({"conf": conf, "shape": router.shape(subs), "correct": bool(subs) and subs[0].tool == tool})
for q, tools in composed:
    subs, conf = router.tier1(q)
    if router.shape(subs) == 2: cal.append({"conf": conf, "shape": 2, "correct": all(t in [x.tool for x in subs] for t in tools)})
T = [(q, *router.tier1(q.text)) for q in bench.single]
out = {}
for eps in [0.15, 0.10]:
    taus = tau_by_shape(cal, eps)
    det = np.array([c >= taus[router.shape(s)] for q, s, c in T])
    rc = np.array([bool(s) and s[0].tool == q.gold_tools[0] for q, s, c in T])
    for m in ['hirag', 'full', 'flat', 'colbert', 'stage1']:
        f = [f for f in glob.glob(f'{ROOT}/hirag_results_or/e2e_{m}_single_k1_*.json') if 'PARTIAL' not in f][0]
        rows = {r['qid']: r for r in json.load(open(f))['rows']}
        base = np.array([bool(rows[q.qid]['correct']) for q, _, _ in T])
        rout = np.where(det, rc, base)
        diff = rout.astype(float) - base.astype(float)
        rng = np.random.default_rng(0); n = len(diff)
        boots = [diff[rng.integers(0, n, n)].mean() for _ in range(10000)]
        lo, hi = np.percentile(boots, [2.5, 97.5]) * 100
        out[f'{eps}_{m}'] = dict(router=round(100 * rout.mean(), 1), fallback=round(100 * base.mean(), 1), diff=round(100 * diff.mean(), 1), lo=round(lo, 1), hi=round(hi, 1))
        print(f'eps={eps} {m:8} router {100*rout.mean():5.1f}  fallback {100*base.mean():5.1f}  diff {100*diff.mean():+5.1f}  95% CI [{lo:+.1f}, {hi:+.1f}]  coverage {100*det.mean():.1f}%')
json.dump(out, open('noninferiority.json', 'w'), indent=1)
