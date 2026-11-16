import os
ROOT = os.environ.get('ROUTER_PACK', '.')  # folder that contains hirag-repro-f/, hirag_stage2/, hirag_results_or/
import sys, json, math, random
sys.path.insert(0, os.path.join(ROOT, 'hirag-repro-f'))
from collections import Counter
from hirag.data import load_mcpbench
from hirag.registry import build_registry
from hirag.distill import Distiller, with_learned, query_feature_set
from hirag.router import HybridRouter
from hirag.sparse import BM25Retriever
from hirag.calibrate import tau_by_shape
bench = load_mcpbench(); reg0 = build_registry(bench); ids = [t.tool_id for t in bench.tools]
bm = BM25Retriever([t.summary for t in bench.tools], "clean")
retr = lambda text: ids[bm.rank(text, 1)[0]]
d = json.load(open(os.path.join(ROOT, 'hirag_stage2', 'train_queries_llm_labels.json')))
train = [(x['query'], x['tool_id']) for x in d['train']]; val = [(x['query'], x['tool_id']) for x in d['val']]
learned = Distiller(2, 0.6).fit(train); reg = with_learned(reg0, learned)
router = HybridRouter(reg, tau=math.inf, retriever=retr)
# calibration exactly as run_distill (single-shape + composed multi)
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
taus = tau_by_shape(cal, 0.15); router.tau = taus
print('tau', taus)
out = []
for q in bench.single:
    dcs = router.route(q.text); s = dcs.subs[0]
    reason = 'answered by rules' if dcs.tier == 'deterministic' else ('retriever disagrees' if not s.agrees else 'margin below tau')
    out.append(dict(q=q.text, gold=q.gold_tools[0], tier=dcs.tier, tool=s.tool, conf=round(dcs.confidence, 3), ru=s.runner_up, bm=s.retriever_tool, reason=reason, parts={k: round(v, 2) for k, v in s.parts.items()}))
c = Counter(o['reason'] for o in out); print(c)
json.dump(out, open('examples.json', 'w'), indent=1)
for r in ['answered by rules', 'margin below tau', 'retriever disagrees']:
    print('\n#####', r)
    for o in [o for o in out if o['reason'] == r][:7]:
        print(f"- {o['q'][:110]}\n    rule→{o['tool']} (runner-up {o['ru']}), BM25→{o['bm']}, c={o['conf']}, gold={o['gold']}, parts={o['parts']}")
# purity worked example
pair = Counter(); tot = Counter()
for q, t in train:
    for f in query_feature_set(q): pair[(f, t)] += 1; tot[f] += 1
for f in ['b:merg_request', 'w:pull', 'b:pull_request', 'w:branch', 'w:note', 'w:video', 'w:bazi', 'w:templat']:
    tops = sorted([(t, c) for (ff, t), c in pair.items() if ff == f], key=lambda x: -x[1])[:3]
    print(f, 'total', tot[f], tops, [round(c / (tot[f] + 1), 2) for _, c in tops])
