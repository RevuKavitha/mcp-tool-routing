# Extra experiments for the router paper (ESWA submission)

Run from a folder that contains `hirag-repro-f/`, `hirag_stage2/` and `hirag_results_or/`
(or set `ROUTER_PACK=/path/to/that/folder`). CPU only, no LLM calls: everything reuses the
saved per-request outputs and the 1,151 logged LLM decisions.

| Script | Paper |
|---|---|
| `python hirag-repro-f/scripts/run_distill.py --calib-file hirag_stage2/train_queries_llm_labels.json --fallback-dir hirag_results_or --fallback-method hirag --compose-multi 400 --sizes all --eps 0.02,0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.25,0.30` | Figure 5 (ε sweep); repeat with `--fallback-method full` |
| `purity.py` | Table 3 (rules from wrong labels, with and without the purity filter) |
| `examples.py` | Section 4.2.5 (routed / escalated examples) and the purity worked example in Section 3.7 |
| `clf_baseline.py` | Classifier baselines (logistic regression, kNN) behind the gate |
| `clf_curve.py`, `hybrid.py` | Table 4 (Section 4.3.1): classifier and rules + classifier learning curves |
| `noninferiority.py` | Section 4.2.1 (95% CI of the accuracy difference router − fallback) |
| `multistep_subtasks.py` | Section 4.6 (sub-tasks of multi-service requests the rules could answer) |
| `make_paper2_figures.py` | Figures 3, 4 and 5 (values copied from the runs above) |

Requires `scikit-learn` in addition to `requirements.txt`.
