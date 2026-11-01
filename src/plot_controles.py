# grafico de barras dos controles (64 tokens, protocolo leve, acuracia de teste)
# barra = media das seeds; pontos brancos = cada seed
import json, os, sys
import numpy as np
from figstyle import *

src = sys.argv[1]; out = sys.argv[2]
d = json.load(open(src))
acc = lambda k: d[k]["test_acc"]
rows = [  # (rotulo, chaves das seeds, cor)
    ("frontend\nonly", ["NO_SSM"], NONE),
    ("conv\nmixer", ["CONVMIX", "CONVMIX_s43", "CONVMIX_s44", "CONVMIX_s45", "CONVMIX_s46"], CONV),
    ("SSM", ["BASE", "BASE_s43", "BASE_s44"], SSM),
    ("Transformer", ["TRANSFORMER", "TRANSFORMER_s43", "TRANSFORMER_s44", "TRANSFORMER_s45", "TRANSFORMER_s46"], TRF),
]
f, ax = fig(height=2.5)
labels = []
for i, (lab, keys, col) in enumerate(rows):
    keys = [k for k in keys if k in d]
    v = np.array([acc(k) for k in keys]); m = v.mean()
    ax.bar(i, m, width=0.66, color=col, zorder=2)
    if len(v) > 1:  # each seed as a small white dot inside the bar top
        xs = i + np.linspace(-0.18, 0.18, len(v))
        ax.scatter(xs, v, s=11, facecolor="white", edgecolor=INK, lw=0.6, zorder=3)
    ax.text(i, max(v.max(), m) + 1.2, f"{m:.1f}%", ha="center", va="bottom", color=INK)
    labels.append(f"{lab}\n({len(v)} seed{'s' if len(v) > 1 else ''})")
ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels)
ax.set_ylabel("test accuracy (%)"); ax.set_ylim(0, 60)
ax.grid(axis="y")
save(f, out)
