# figura de robustez a taxa para o artigo: SSM x Transformer x conv mixer, taxa fixa x multi-taxa.
# le invariance_radioml_arch.json (gerado por INV_SET=arch python invariance_radioml.py)
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "invariance_radioml_arch.json"
out = sys.argv[2] if len(sys.argv) > 2 else "robustez_arquiteturas.png"
d = json.load(open(src)); F = d["factors"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = [("SSM", "#2a78d6", "o", "BASE", "MULTIRATE"),
         ("Transformer", "#eb6834", "s", "TRANSFORMER", "TRANSFORMER_MULTIRATE"),
         ("conv mixer", "#1baf7a", "^", "CONVMIX", "CONVMIX_MULTIRATE")]

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.6, 2.9))
for name, col, mk, fixed, multi in STYLE:
    for key, ls, lab in ((fixed, "--", "fixed rate"), (multi, "-", "multi-rate")):
        a = np.array(d["curves_per_seed"][key]); mean = a.mean(0)
        n = f", {len(a)} seeds" if len(a) > 1 else ""
        ax.plot(F, mean, ls=ls, color=col, lw=1.8, marker=mk, ms=4.5, mec="white", mew=0.8,
                label=f"{name}, {lab} ({mean.mean():.1f}%{n})")
        if len(a) > 1:
            ax.fill_between(F, a.min(0), a.max(0), color=col, alpha=0.12, lw=0)
ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_xticks([0.6, 0.8, 1.0, 1.25, 1.5]); ax.set_xticklabels(["0.6", "0.8", "1.0", "1.25", "1.5"]); ax.minorticks_off()
ax.set_xlabel("resampling factor at test time"); ax.set_ylabel("accuracy (%)")
ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True); ax.set_ylim(0, 56)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.6, frameon=False,
          handlelength=2.6, columnspacing=1.0)
plt.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
print("saved", out)
