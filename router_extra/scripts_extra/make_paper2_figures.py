import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib.lines import Line2D
blue = '#2a6fdb'; orange = '#e8642c'; grey = '#7d828c'; dark = '#333333'; green = '#2e8b57'
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5})
W = 6.8   # inches; printed at 6 in -> ~0.88 scale


def lab(a, pt, at, txt, col):
    a.annotate(txt, pt, xytext=at, textcoords='data', fontsize=7.5, color=col, ha='left', va='center',
               bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=col, lw=0.7),
               arrowprops=dict(arrowstyle='-', color=col, lw=0.7, shrinkA=0, shrinkB=5))


# ---------------- Figure 3 ----------------
fig, ax = plt.subplots(2, 1, figsize=(W, 8.0))
a = ax[0]
for n, x, y, dy in [('Hi-RAG', 971, 85.1, 0), ('ColBERT-RAG', 995, 87.1, 0), ('Stage 1 only', 994, 83.6, 0.12), ('Flat-RAG', 1008, 83.1, -0.12)]:
    a.scatter(x, y, marker='s', s=40, color=grey, zorder=3)
    a.text(x + 15, y + dy, f'{n} alone: {y}%', fontsize=7.5, color=grey, va='center')
P = [(526, 88.6, True, 'self-distilled, ε = 0.15\n88.6% · 526 tokens · 46% skip LLM', (405, 90.55)),
     (680, 89.1, True, 'self-distilled, ε = 0.10\n89.1% · 680 tokens · 33% skip LLM', (790, 90.55)),
     (834, 87.1, False, 'static rules, ε = 0.10\n87.1% · 834 tokens · 17% skip LLM', (850, 88.6)),
     (735, 86.6, False, 'static rules, ε = 0.15\n86.6% · 735 tokens · 26% skip LLM', (405, 84.2))]
for x, y, f, t, at in P:
    a.scatter(x, y, s=60, facecolor=blue if f else 'white', edgecolor=blue, lw=1.8, zorder=4)
    lab(a, (x, y), at, t, blue)
a.annotate('', xy=(540, 88.5), xytext=(960, 85.2), arrowprops=dict(arrowstyle='->', color=blue, lw=1.0, ls='--'))
a.set_xlim(400, 1250); a.set_ylim(82.5, 91.3)
a.set_xlabel('Average prompt tokens per request (lower = cheaper)'); a.set_ylabel('Top-1 accuracy (%)')
a.set_title('(a) Router in front of Hi-RAG (other retrieval fallbacks shown for reference)'); a.grid(alpha=.3)
b = ax[1]
b.scatter(17718, 92.5, marker='s', s=40, color=grey, zorder=3)
lab(b, (17718, 92.5), (15300, 91.15), 'Full context alone\n92.5% · 17,718 tokens', grey)
F = [(9609, 93.5, True, 'self-distilled, ε = 0.15\n93.5% · 9,609 tokens · 46% skip LLM', (7150, 95.05)),
     (11900, 94.0, True, 'self-distilled, ε = 0.10\n94.0% · 11,900 tokens · 33% skip LLM', (13700, 95.05)),
     (14721, 93.0, False, 'static rules, ε = 0.10\n93.0% · 14,721 tokens · 17% skip LLM', (15300, 93.85)),
     (13135, 92.5, False, 'static rules, ε = 0.15\n92.5% · 13,135 tokens · 26% skip LLM', (7400, 91.15))]
for x, y, f, t, at in F:
    b.scatter(x, y, s=60, facecolor=orange if f else 'white', edgecolor=orange, lw=1.8, zorder=4)
    lab(b, (x, y), at, t, orange)
b.annotate('', xy=(9800, 93.47), xytext=(17550, 92.52), arrowprops=dict(arrowstyle='->', color=orange, lw=1.0, ls='--'))
b.set_xlim(7000, 21000); b.set_ylim(90.6, 95.6)
b.set_xlabel('Average prompt tokens per request (lower = cheaper)'); b.set_ylabel('Top-1 accuracy (%)')
b.set_title('(b) Router in front of full-context injection'); b.grid(alpha=.3)
h = [Line2D([], [], marker='s', ls='', color=grey, ms=6, label='fallback alone (LLM called for every request)'),
     Line2D([], [], marker='o', ls='', mfc='white', mec=dark, mew=1.6, ms=7, label='router, static rules only (from tool metadata)'),
     Line2D([], [], marker='o', ls='', mfc=dark, mec=dark, ms=7, label='router after self-distillation (+ rules from logs)'),
     Line2D([], [], ls='--', color=dark, label='fallback alone → self-distilled router, ε = 0.15')]
fig.legend(handles=h, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle('201 MCPBench single-service requests, Qwen3-32B.  "skip LLM" = share answered by the rules alone', fontsize=8.5)
plt.tight_layout(rect=(0, 0.065, 1, 0.97)); plt.savefig('fig3_new.png', dpi=300)

# ---------------- Figure 4 ----------------
x = [0, 100, 200, 400, 800, 1151]; xg = [0, 100, 200, 400, 800, 1200]
L = ([25.9, 16.7, 28.4, 37.0, 43.0, 45.8], [0, 12.9, 5.4, 0.6, 3.1, 0])
G = ([25.9, 14.3, 13.3, 29.4, 42.1, 45.3], [0, 10.2, 9.4, 5.9, 3.4, 0])
N = ([25.9, 22.2, 15.4, 26.9, 37.0, 38.6], [0, 2.8, 5.8, 5.5, 5.1, 0.2])
fig, a = plt.subplots(figsize=(W, 4.0))
for (m, sd), xs, col, lab_, mk, dy in [(L, x, blue, "LLM's own decisions (1,151 logs, no gold labels)", 'o', 1.8),
                                     (G, xg, green, 'gold labels (upper reference, 1,200 logs)', '^', -1.8),
                                     (N, xg, grey, 'gold labels, 15% replaced by a wrong tool', 's', 0)]:
    m = np.array(m); sd = np.array(sd)
    a.plot(xs, m, marker=mk, color=col, lw=1.8, ms=5, label=lab_, zorder=3)
    a.fill_between(xs, m - sd, m + sd, color=col, alpha=.08)
    a.text(xs[-1] + 22, m[-1] + dy, f'{m[-1]}%', color=col, fontsize=8, fontweight='bold', va='center')
a.axhline(25.9, color=dark, ls=':', lw=1.0)
a.text(1000, 24.2, 'static rules only (no logs): 25.9%', fontsize=7.5, color=dark, ha='center', va='top')
a.annotate('few logs: learned rules are still sparse,\nso rule confidence dips before it rises', (150, 15), xytext=(250, 5),
           fontsize=7.5, color=dark, arrowprops=dict(arrowstyle='->', color=dark, lw=0.7))
a.set_xlabel('Number of logged decisions used to learn new rules')
a.set_ylabel('Requests answered without the LLM (%)')
a.set_xlim(-30, 1340); a.set_ylim(0, 55); a.grid(alpha=.3)
a.legend(loc='upper left', title='Where the learning labels come from', title_fontsize=7.5)
a.set_title('ε = 0.15. Lines: mean of 3 random log orders; bands: ±1 s.d. Blue-line precision stays 94–98%.', fontsize=8.5)
plt.tight_layout(); plt.savefig('fig4_new.png', dpi=300)

# ---------------- Figure 5 ----------------
eps = [0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30]
cov = [0, 0, 17.4, 32.8, 36.8, 45.8, 48.8, 54.7, 54.7, 54.7]
prec = [None, None, 97.1, 97.0, 94.6, 95.7, 92.9, 90.9, 90.9, 90.9]
acc_h = [85.1, 85.1, 87.1, 89.1, 88.1, 88.6, 88.1, 88.1, 88.1, 88.1]
acc_f = [92.5, 92.5, 93.5, 94.0, 93.5, 93.5, 92.5, 91.5, 91.5, 91.5]
fig, ax = plt.subplots(2, 1, figsize=(W, 6.6))
a = ax[0]
a.plot(eps, cov, 'o-', color=blue, lw=2, ms=5, label='requests answered without the LLM (%)')
pe = [(e, p) for e, p in zip(eps, prec) if p is not None]
a.plot([e for e, _ in pe], [p for _, p in pe], 's--', color=dark, lw=1.5, ms=4.5, label='precision of these rule decisions (%)')
a.plot(eps, [100 * (1 - e) for e in eps], ':', color=grey, lw=1.3, label='target precision 1 − ε')
a.axvline(0.15, color=blue, alpha=.2, lw=6)
a.text(0.155, 66, 'ε = 0.15 (used in the paper)', color=blue, fontsize=7.5)
a.text(0.035, 5, 'no rule decision\ncan be certified', fontsize=7.5, color=dark, ha='center')
a.set_xlabel('Error target ε'); a.set_ylabel('%'); a.set_ylim(0, 103)
a.set_title('(a) Coverage and precision of the rules'); a.grid(alpha=.3)
a.legend(loc='lower right', bbox_to_anchor=(1.0, 0.05))
b = ax[1]
b.plot(eps, acc_h, 'o-', color=blue, lw=2, ms=5, label='router + Hi-RAG fallback')
b.axhline(85.1, color=blue, ls='--', lw=1.0, alpha=.7, label='Hi-RAG alone (85.1%)')
b.plot(eps, acc_f, '^-', color=dark, lw=1.8, ms=5, label='router + full-context fallback')
b.axhline(92.5, color=dark, ls='--', lw=1.0, alpha=.7, label='full context alone (92.5%)')
b.axvline(0.15, color=blue, alpha=.2, lw=6)
b.annotate('above 0.15: below\nfull context alone', (0.2, 91.5), xytext=(0.215, 89.6), fontsize=7.5, color=dark,
           arrowprops=dict(arrowstyle='->', color=dark, lw=0.7))
b.set_xlabel('Error target ε'); b.set_ylabel('Top-1 accuracy (%)'); b.set_ylim(80.5, 95.5)
b.set_title('(b) End-to-end accuracy'); b.grid(alpha=.3)
b.legend(loc='lower right', ncol=2)
fig.suptitle('201 MCPBench single-service requests, Qwen3-32B, self-distilled router', fontsize=8.5)
plt.tight_layout(rect=(0, 0, 1, 0.975)); plt.savefig('fig_eps.png', dpi=300)
