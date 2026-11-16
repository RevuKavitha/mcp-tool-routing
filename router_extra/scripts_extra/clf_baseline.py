import os
ROOT = os.environ.get('ROUTER_PACK', '.')  # folder that contains hirag-repro-f/, hirag_stage2/, hirag_results_or/
import sys, json, glob
sys.path.insert(0, os.path.join(ROOT, 'hirag-repro-f'))
import numpy as np
from hirag.data import load_mcpbench
from hirag.sparse import BM25Retriever
from hirag.calibrate import risk_control_tau, mcnemar_exact
from hirag.registry import norm_tokens
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
bench = load_mcpbench(); ids = [t.tool_id for t in bench.tools]
bm = BM25Retriever([t.summary for t in bench.tools], "clean"); bmtop = lambda q: ids[bm.rank(q, 1)[0]]
d = json.load(open(os.path.join(ROOT, 'hirag_stage2', 'train_queries_llm_labels.json')))
Xtr = [x['query'] for x in d['train']]; ytr = [x['tool_id'] for x in d['train']]
Xva = [x['query'] for x in d['val']]; yva = [x['tool_id'] for x in d['val']]
test = bench.single
def rows(m):
    f = [f for f in glob.glob(f'{ROOT}/hirag_results_or/e2e_{m}_single_k1_*.json') if 'PARTIAL' not in f][0]
    return {r['qid']: r for r in json.load(open(f))['rows']}
fb = {m: rows(m) for m in ['hirag', 'full', 'flat', 'colbert', 'stage1']}
an = lambda t: ' '.join(norm_tokens(t))   # same stemming/stop-words as the router
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, preprocessor=an)
A = vec.fit_transform(Xtr); V = vec.transform(Xva); T = vec.transform([q.text for q in test])
models = {'Logistic regression (TF-IDF)': LogisticRegression(C=10, max_iter=3000),
          'k-nearest neighbours (TF-IDF, k=5)': KNeighborsClassifier(n_neighbors=5, metric='cosine', weights='distance')}
out = {}
for name, clf in models.items():
    clf.fit(A, ytr); cls = clf.classes_
    def pred(M, texts):
        P = clf.predict_proba(M); top2 = np.argsort(-P, 1)[:, :2]
        tool = cls[top2[:, 0]]; margin = P[np.arange(len(P)), top2[:, 0]] - P[np.arange(len(P)), top2[:, 1]]
        return tool, margin
    for gate in ['margin only', 'margin + BM25 consensus']:
        tv, mv = pred(V, Xva); tt, mt = pred(T, [q.text for q in test])
        if gate.endswith('consensus'):
            mv = np.where([a == bmtop(q) for a, q in zip(tv, Xva)], mv, 0.0)
            mt = np.where([a == bmtop(q.text) for a, q in zip(tt, test)], mt, 0.0)
        for eps in [0.10, 0.15]:
            tau = risk_control_tau(list(mv), [a == b for a, b in zip(tv, yva)], eps)
            det = [m >= tau for m in mt]; corr = [a == q.gold_tools[0] for a, q in zip(tt, test)]
            n = len(test); cov = 100 * sum(det) / n
            prec = 100 * sum(c for c, dd in zip(corr, det) if dd) / max(1, sum(det))
            res = {'cov': round(cov, 1), 'prec': round(prec, 1), 'tau': round(float(tau), 3) if tau != float('inf') else None}
            for m in ['hirag', 'full']:
                r = fb[m]
                ok = [(c if dd else bool(r[q.qid]['correct'])) for c, dd, q in zip(corr, det, test)]
                base = [bool(r[q.qid]['correct']) for q in test]
                tok = np.mean([0 if dd else r[q.qid].get('prompt_tokens', 0) for dd, q in zip(det, test)])
                b = sum(1 for o, bb in zip(ok, base) if o and not bb); c_ = sum(1 for o, bb in zip(ok, base) if bb and not o)
                res[m] = {'acc': round(100 * sum(ok) / n, 1), 'base': round(100 * sum(base) / n, 1), 'tok': round(float(tok)), 'plus': b, 'minus': c_, 'p': round(mcnemar_exact(ok, base)[2], 3)}
            out[f'{name} | {gate} | eps={eps}'] = res
            print(f'{name:36} {gate:24} eps={eps:.2f}  cov {cov:5.1f}%  prec {prec:5.1f}%  HiRAG {res["hirag"]["acc"]}% ({res["hirag"]["tok"]} tok, +{b if False else res["hirag"]["plus"]}/-{res["hirag"]["minus"]})  Full {res["full"]["acc"]}% ({res["full"]["tok"]} tok)')
json.dump(out, open('clf_baseline.json', 'w'), indent=1)
