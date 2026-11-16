"""How many sub-tasks inside the 60 multi-service requests could the rules answer on their own
(each sub-task gated with the single-step threshold tau_1 at eps = 0.15)?"""
import os, sys, json, math, random
ROOT = os.environ.get('ROUTER_PACK', '.')
sys.path.insert(0, os.path.join(ROOT, 'hirag-repro-f'))
from hirag.data import load_mcpbench
from hirag.registry import build_registry
from hirag.distill import Distiller, with_learned
from hirag.router import HybridRouter
from hirag.sparse import BM25Retriever
from hirag.calibrate import tau_by_shape
bench = load_mcpbench(); reg0 = build_registry(bench); ids = [t.tool_id for t in bench.tools]
bm = BM25Retriever([t.summary for t in bench.tools], "clean"); retr = lambda t: ids[bm.rank(t, 1)[0]]
d = json.load(open(os.path.join(ROOT, 'hirag_stage2', 'train_queries_llm_labels.json')))
train = [(x['query'], x['tool_id']) for x in d['train']]; val = [(x['query'], x['tool_id']) for x in d['val']]
router = HybridRouter(with_learned(reg0, Distiller(2, 0.6).fit(train)), tau=math.inf, retriever=retr)
cal = []
for q, tool in val:
    subs, conf = router.tier1(q); cal.append({"conf": conf, "shape": router.shape(subs), "correct": bool(subs) and subs[0].tool == tool})
tau1 = tau_by_shape(cal, 0.15)[1]
n_sub = n_det = n_ok = req_any = req_gold_subs = 0
split_counts = []
for q in bench.multi:
    subs, conf = router.tier1(q.text); split_counts.append(len(subs))
    dets = [s for s in subs if s.agrees and s.confidence >= tau1]
    n_sub += len(subs); n_det += len(dets); n_ok += sum(s.tool in q.gold_tools for s in dets)
    req_any += bool(dets)
print(f'tau_1 = {tau1:.3f}')
print(f'60 multi-service requests -> {n_sub} sub-tasks after splitting (requests split into >=2: {sum(c>=2 for c in split_counts)})')
print(f'sub-tasks the rules could answer alone: {n_det} ({100*n_det/n_sub:.1f}%), correct: {n_ok} ({100*n_ok/max(1,n_det):.1f}%)')
print(f'requests with at least one such sub-task: {req_any} ({100*req_any/60:.1f}%)')
