import os
ROOT = os.environ.get('ROUTER_PACK', '.')  # folder that contains hirag-repro-f/, hirag_stage2/, hirag_results_or/
import sys, json, random
sys.path.insert(0, os.path.join(ROOT, 'hirag-repro-f'))
import numpy as np
from hirag.data import load_mcpbench
from hirag.sparse import BM25Retriever
from hirag.calibrate import risk_control_tau
from hirag.registry import norm_tokens
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
bench = load_mcpbench(); ids = [t.tool_id for t in bench.tools]
bm = BM25Retriever([t.summary for t in bench.tools], "clean"); cache = {}
def bmtop(q):
    if q not in cache: cache[q] = ids[bm.rank(q, 1)[0]]
    return cache[q]
d = json.load(open(os.path.join(ROOT, 'hirag_stage2', 'train_queries_llm_labels.json')))
train = [(x['query'], x['tool_id']) for x in d['train']]
Xva = [x['query'] for x in d['val']]; yva = [x['tool_id'] for x in d['val']]
test = bench.single; an = lambda t: ' '.join(norm_tokens(t))
for size in [100, 200, 400, 800, 1151]:
    covs, precs = [], []
    for seed in range(3 if size < 1151 else 1):
        logs = train[:]; random.Random(seed).shuffle(logs); logs = logs[:size]
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, preprocessor=an)
        A = vec.fit_transform([q for q, _ in logs]); clf = LogisticRegression(C=10, max_iter=3000).fit(A, [t for _, t in logs]); cls = clf.classes_
        def pred(texts):
            P = clf.predict_proba(vec.transform(texts)); o = np.argsort(-P, 1)
            tool = cls[o[:, 0]]; mg = P[np.arange(len(P)), o[:, 0]] - (P[np.arange(len(P)), o[:, 1]] if P.shape[1] > 1 else 0)
            return tool, np.where([a == bmtop(q) for a, q in zip(tool, texts)], mg, 0.0)
        tv, mv = pred(Xva); tau = risk_control_tau(list(mv), [a == b for a, b in zip(tv, yva)], 0.15)
        tt, mt = pred([q.text for q in test]); det = mt >= tau; corr = np.array([a == q.gold_tools[0] for a, q in zip(tt, test)])
        covs.append(100 * det.mean()); precs.append(100 * corr[det].mean() if det.any() else float('nan'))
    print(f'{size:5} logs: LR+consensus coverage {np.mean(covs):5.1f} ± {np.std(covs):4.1f}%   precision {np.nanmean(precs):5.1f}%   classes seen {len(cls)}')
