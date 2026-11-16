import os
ROOT = os.environ.get('ROUTER_PACK', '.')  # folder that contains hirag-repro-f/, hirag_stage2/, hirag_results_or/
import sys, json, random
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(ROOT, 'hirag-repro-f'))
from hirag.data import load_mcpbench
from hirag.sparse import BM25Retriever
from hirag.distill import query_feature_set
bench = load_mcpbench(); ids = [t.tool_id for t in bench.tools]
bm = BM25Retriever([t.summary for t in bench.tools], "clean")
d = json.load(open(os.path.join(ROOT, 'hirag_stage2', 'train_queries_llm_labels.json')))
llm = [(x['query'], x['tool_id'], x['gold_tool_id']) for x in d['train']]
g = json.load(open(os.path.join(ROOT, 'hirag_stage2', 'train_queries.json')))
gold = [(x['query'], x['tool_id']) for x in g['train']]
# noisy gold logs exactly as run_distill (all logs, seed 0)
rnd = random.Random(0); logs = gold[:]; rnd.shuffle(logs)
noisy = []
for q, t in logs:
    if rnd.random() < 0.15:
        wrong = next((x for x in [ids[i] for i in bm.rank(q, 5)][:5] if x != t), t)
        noisy.append((q, wrong, t))
    else: noisy.append((q, t, t))
# gold association of features: majority tool under gold labels
gpair = Counter(); gtot = Counter()
for q, t in gold:
    for f in query_feature_set(q): gpair[(f, t)] += 1; gtot[f] += 1
gbest = {}
for (f, t), c in gpair.items():
    if c > gpair.get((f, gbest.get(f)), 0): gbest[f] = t
def analyse(name, L):
    pair = Counter(); tot = Counter(); wrongc = Counter()
    for q, t, gt in L:
        for f in query_feature_set(q):
            pair[(f, t)] += 1; tot[f] += 1
            if t != gt: wrongc[(f, t)] += 1
    nwrong = sum(1 for _, t, gt in L if t != gt)
    res = {}
    for label, pur in [('count only (>=2)', 0.0), ('count + purity >= 0.6', 0.6)]:
        rules = {(f, t): c / (tot[f] + 1) for (f, t), c in pair.items() if c >= 2 and c / (tot[f] + 1) >= pur}
        bad = {k: v for k, v in rules.items() if wrongc[k] > 0 and gbest.get(k[0]) != k[1]}
        res[label] = (len(rules), len(bad))
    # examples: rules created by wrong labels that count-only keeps but purity drops / keeps
    cand = [((f, t), pair[(f, t)], pair[(f, t)] / (tot[f] + 1)) for (f, t) in pair
            if pair[(f, t)] >= 2 and wrongc[(f, t)] > 0 and gbest.get(f) != t]
    dropped = sorted([c for c in cand if c[2] < 0.6], key=lambda c: -c[1])[:6]
    kept = sorted([c for c in cand if c[2] >= 0.6], key=lambda c: -c[1])[:6]
    good = sorted([((f, t), c, c / (tot[f] + 1)) for (f, t), c in pair.items() if c >= 2 and c / (tot[f] + 1) >= 0.6 and gbest.get(f) == t], key=lambda c: -c[2])[:6]
    print(f'\n=== {name}: {len(L)} logs, {nwrong} wrong labels ({100*nwrong/len(L):.1f}%)')
    for k, (n, b) in res.items(): print(f'  {k:24}: {n:5} rules, {b:4} from wrong labels ({100*b/max(n,1):.1f}%)')
    show = lambda c: f'{c[0][0][2:]} -> {c[0][1]} (count {c[1]}, purity {c[2]:.2f}, gold tool: {gbest.get(c[0][0])})'
    print('  wrong-label rules REMOVED by purity filter:'); [print('    ', show(c)) for c in dropped]
    print('  wrong-label rules that still PASS:'); [print('    ', show(c)) for c in kept]
    print('  correct rules that pass:'); [print('    ', show(c)) for c in good]
    return res
a = analyse("LLM's own decisions", llm)
b = analyse('Gold + 15% wrong labels (BM25 runner-up)', noisy)
# where do LLM mistakes happen? feature ambiguity of mislabeled queries
def amb(L):
    vals = []
    for q, t, gt in L:
        fs = query_feature_set(q)
        if not fs: continue
        # mean gold purity of the query's features for its gold tool
        vals.append((t != gt, sum(gpair[(f, gt)] / (gtot[f] + 1) for f in fs) / len(fs)))
    w = [v for x, v in vals if x]; r = [v for x, v in vals if not x]
    return sum(w)/len(w), sum(r)/len(r)
print('\nmean gold-purity of query words (wrong-labelled vs correctly-labelled queries):')
print("  LLM logs   :", ['%.3f' % v for v in amb(llm)])
print("  noisy logs :", ['%.3f' % v for v in amb(noisy)])
