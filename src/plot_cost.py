# figura de custo medido (substitui a curva teorica normalizada da Fig. complexidade)
import sys, json

import os
from figstyle import *
HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "cost_ssm_vs_transformer.json")))
out = os.path.join(HERE, "..", "paper", "figs", "complexidade_medida.png")
SSM_C, TRF_C = SSM, TRF

def series(name, key):
    r = [x for x in rows if x["model"] == name]
    return [x["tokens"] for x in r], [x[key] for x in r]

f, axs = fig(width=PAGE_W, height=2.3, ncols=3)

def lines(ax, key, ylab, title, log=False):
    for name, c, mk in (("SSM", SSM_C, "o"), ("Transformer", TRF_C, "s")):
        x, y = series(name, key)
        ax.plot(x, y, color=c, marker=mk, mec="white", mew=0.5, label=name)
        ax.annotate(name, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK, fontsize=7, va="center")
    ax.set_xscale("log", base=2); ax.set_xticks([32, 64, 128, 256, 512])
    ax.set_xticklabels(["32", "64", "128", "256", "512"])
    if log: ax.set_yscale("log")
    ax.set_xlabel("number of tokens $N$"); ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid()
    ax.set_xlim(26, 1100)

lines(axs[0], "total_GFLOPs", "GFLOPs (forward, batch 1)", "(a) Measured compute")
axs[0].axvline(256, color=MUTED, lw=0.8, ls=":")
axs[0].annotate("crossover", (256, 0.85), xytext=(-5, 0), textcoords="offset points",
                fontsize=7, color=MUTED, ha="right")
lines(axs[1], "latency_cpu_ms", "CPU latency (ms, log scale)", "(b) Measured latency", log=True)

# (c) acuracia x custo real, configuracao base (Tabela II)
base = [(32, 0.0510, 36.34), (64, 0.0985, 48.87), (256, 0.3876, 56.61)]   # acuracia de TESTE (seed 42)
ax = axs[2]
ax.plot([b[1] for b in base], [b[2] for b in base], color=SSM_C, marker="o", mec="white", mew=0.5)
for t, g, a in base:
    ax.annotate(f"{t} tokens", (g, a), xytext=(6, -10), textcoords="offset points",
                fontsize=7, color=INK)
ax.set_xlabel("GFLOPs (forward, batch 1)"); ax.set_ylabel("test accuracy (%)")
ax.set_title("(c) Accuracy vs. compute (SSM)")
ax.grid(); ax.set_xlim(0, 0.47); ax.set_ylim(30, 62)

save(f, out)
