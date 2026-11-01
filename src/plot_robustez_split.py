"""Rate robustness as two single-column figures: fixed-rate models and multi-rate models.
Line = mean over seeds, band = min to max over seeds. Reads invariance_radioml_arch_polyphase.json."""
import sys, json
import numpy as np
from figstyle import *

src = sys.argv[1]; outdir = sys.argv[2]
d = json.load(open(src)); F = d["factors"]
MIX = [("SSM", SSM, "o", ""), ("Transformer", TRF, "s", "TRANSFORMER_"), ("conv mixer", CONV, "^", "CONVMIX_")]
for mode, name in (("fixed", "robustez_taxa_fixa.png"), ("multi", "robustez_multitaxa.png")):
    f, ax = fig(height=2.5)
    for lab, col, mk, pre in MIX:
        key = (pre + "MULTIRATE") if mode == "multi" else (pre[:-1] if pre else "BASE")
        a = np.array(d["curves_per_seed"][key])
        ax.fill_between(F, a.min(0), a.max(0), color=col, alpha=0.18, lw=0)
        ax.plot(F, a.mean(0), color=col, marker=mk, mec="white", mew=0.5, label=f"{lab} (mean {a.mean():.1f}%)")
    ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xticks([0.6, 0.8, 1.0, 1.25, 1.5])
    ax.set_xticklabels(["0.6", "0.8", "1.0", "1.25", "1.5"]); ax.minorticks_off()
    ax.set_xlabel("resampling factor at test time"); ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 55); ax.grid()
    legend_above(ax)
    save(f, f"{outdir}/{name}")
