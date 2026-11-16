import os
ROOT = os.environ.get('ROUTER_PACK', '.')  # folder that contains hirag-repro-f/, hirag_stage2/, hirag_results_or/
import sys, json, math, random, glob
sys.path.insert(0, os.path.join(ROOT, 'hirag-repro-f'))
import numpy as np
from hirag.data import load_mcpbench
from hirag.registry import build_registry, norm_tokens
from hirag.distill import Distiller, with_learned
from hirag.router import HybridRouter
from hirag.sparse import BM25Retriever
from hirag.calibrate import mcnemar_exact
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
bench = load_mcpbench(); reg0 = build_registry(bench); ids = [t.tool_id for t in bench.tools]
bm = BM25Retriever([t.summary for t in bench.tools], "clean"); cache = {}
def bmtop(q):
    if q not in cache: cache[q] = ids[bm.rank(q, 1)[0]]
    return cache[q]
d = json.load(open(os.path.join(ROOT, 'hirag_stage2', 'train_queries_llm_labels.json')))
train = [(x['query'], x['tool_id']) for x in d['train']]
val = [(x['query'], x['tool_id']) for x in d['val']]; test = bench.single
an = lambda t: ' '.join(norm_tokens(t))
fb = {}
for m in ['hirag', 'full', 'flat', 'colbert', 'stage1']:
    f = [f for f in glob.glob(f'{ROOT}/hirag_results_or/e2e_{m}_single_k1_*.json') if 'PARTIAL' not in f][0]
    fb[m] = {r['qid']: r for r in json.load(open(f))['rows']}
def run(size, seed, eps=0.15):
    logs = train[:]; random.Random(seed).shuffle(logs); logs = logs[:size]
    reg = with_learned(reg0, Distiller(2, 0.6).fit(logs)) if size else reg0
    router = HybridRouter(reg, tau=math.inf, retriever=bmtop)
    def rule(q):
        subs, c = router.tier1(q); return subs[0].tool, c, router.shape(subs)
    if size >= 50:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, preprocessor=an)
        clf = LogisticRegression(C=10, max_iter=3000).fit(vec.fit_transform([q for q, _ in logs]), [t for _, t in logs]); cls = clf.classes_
        def lr(texts):
            P = clf.predict_proba(vec.transform(texts)); o = np.argsort(-P, 1); tool = cls[o[:, 0]]
            mg = P[np.arange(len(P)), o[:, 0]] - P[np.arange(len(P)), o[:, 1]]
            return list(tool), list(np.where([a == bmtop(q) for a, q in zip(tool, texts)], mg, 0.0))
    else:
        lr = lambda texts: ([None] * len(texts), [0.0] * len(texts))
    def feats(items):
        R = [rule(q) for q in items]; L = lr(items); return R, L
    vq = [q for q, _ in val]; vy = [t for _, t in val]
    Rv, (Lt, Lm) = feats(vq)
    # candidate thresholds
    rc = sorted(set([c for _, c, sh in Rv if sh == 1 and c > 0])) + [math.inf]
    lc = sorted(set([m for m in Lm if m > 0])) + [math.inf]
    rc = [x for i, x in enumerate(rc) if i % max(1, len(rc) // 60) == 0] + [math.inf]
    lc = [x for i, x in enumerate(lc) if i % max(1, len(lc) // 60) == 0] + [math.inf]
    def decide(R, L, tr, tl):
        out = []
        for (rt, c, sh), lt, m in zip(R, *L):
            pr = sh == 1 and c > 0 and c >= tr; pl = lt is not None and m > 0 and m >= tl
            if pr and pl: out.append(rt if rt == lt else None)
            elif pr: out.append(rt)
            elif pl: out.append(lt)
            else: out.append(None)
        return out
    best = (-1, math.inf, math.inf)
    for tr in rc:
        for tl in lc:
            dv = decide(Rv, (Lt, Lm), tr, tl); cov = sum(x is not None for x in dv); err = sum(x is not None and x != y for x, y in zip(dv, vy))
            if cov and (err + 1) / (cov + 1) <= eps and cov > best[0]: best = (cov, tr, tl)
    tq = [q.text for q in test]; Rt, Ltt = feats(tq)
    dt = decide(Rt, Ltt, best[1], best[2])
    det = [x is not None for x in dt]; corr = [x == q.gold_tools[0] for x, q in zip(dt, test)]
    res = {'cov': 100 * np.mean(det), 'prec': 100 * sum(corr) / max(1, sum(det))}
    for m in fb:
        r = fb[m]; ok = [(c if dd else bool(r[q.qid]['correct'])) for c, dd, q in zip(corr, det, test)]
        base = [bool(r[q.qid]['correct']) for q in test]
        res[m] = (100 * np.mean(ok), np.mean([0 if dd else r[q.qid].get('prompt_tokens', 0) for dd, q in zip(det, test)]), mcnemar_exact(ok, base))
    return res
for size in [0, 100, 200, 400, 800, 1151]:
    rs = [run(size, s) for s in range(1 if size in (0, 1151) else 3)]
    cov = [r['cov'] for r in rs]; prec = [r['prec'] for r in rs]
    h = rs[0]['hirag']; f = rs[0]['full']
    print(f"{size:5} logs | hybrid coverage {np.mean(cov):5.1f} ± {np.std(cov):4.1f}%  precision {np.mean(prec):5.1f}%" + (f" | Hi-RAG {h[0]:.1f}% {h[1]:.0f} tok (+{h[2][0]}/-{h[2][1]}, p={h[2][2]:.3f}) | Full {f[0]:.1f}% {f[1]:.0f} tok (+{f[2][0]}/-{f[2][1]}, p={f[2][2]:.3f})" if size == 1151 else ''))
    if size == 1151:
        for m in ['flat', 'colbert', 'stage1']:
            x = rs[0][m]; print(f"      {m}: {x[0]:.1f}% {x[1]:.0f} tok (+{x[2][0]}/-{x[2][1]}, p={x[2][2]:.3f})")
