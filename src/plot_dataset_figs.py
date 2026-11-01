"""Figures computed directly from the data, so they match the tables exactly:
  snr_distribution.png  - SNR histograms of RadioML and ModForge, same 2 dB bins
  acc_vs_snr.png        - test accuracy vs SNR of the best model (MULTISCALE), from test_accuracy.json
  confusion_matrix.png  - its high-SNR (>= 10 dB) confusion matrix, from test_accuracy.json
"""
import os, sys, json
import numpy as np, h5py
from figstyle import *

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
classes = json.load(open(os.path.join(HERE, "classes-fixed.json")))

# --- SNR distribution: fraction of examples per 2 dB bin, both datasets on the same axes ---
r = np.round(h5py.File(os.path.join(DATA, "GOLD_XYZ_OSC.0001_1024.hdf5"), "r")["Z"][:].squeeze())
m = h5py.File(os.path.join(DATA, "modforge.h5"), "r")["snr_db"][:]
bins = np.arange(-31, 33, 2)                      # centred on the even RadioML levels
f, ax = fig(width=PAGE_W, height=2.4)
for s, name, col, off in ((r, "RadioML 2018.01A (synthetic)", RADIOML, -0.45), (m, "ModForge (recorded, CorteXlab)", MODFORGE, 0.45)):
    h, e = np.histogram(s, bins=bins)
    ax.bar((e[:-1] + e[1:]) / 2 + off, h / len(s) * 100, width=0.9, color=col,
           label=f"{name}: mean {s.mean():.1f} dB, {(s < -10).mean()*100:.0f}% below $-10$ dB, "
                 f"{(s >= 20).mean()*100:.1f}% at 20 dB or more")
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("examples (%)")
ax.grid(axis="y"); ax.set_xlim(-31, 33)
legend_above(ax, ncol=1)
save(f, os.path.join(OUT, "snr_distribution.png"))

# --- accuracy vs SNR of the best model, from the full test set ---
d = json.load(open(os.path.join(HERE, "test_accuracy.json")))["MULTISCALE"]
snrs = sorted(int(k) for k in d["per_snr"]); acc = [d["per_snr"][str(k)] for k in snrs]
f, ax = fig(height=2.5)
ax.plot(snrs, acc, color=SSM, marker="o", mec="white", mew=0.5, label="SSM, multi-scale frontend")
ax.axhline(100 / 24, color=MUTED, lw=0.8, ls=":", label="chance (1/24)")
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("test accuracy (%)"); ax.set_ylim(0, 100); ax.grid()
legend_above(ax, ncol=2)
save(f, os.path.join(OUT, "acc_vs_snr.png"))

# --- high-SNR confusion matrix of the same model, rows normalized ---
cm = np.array(d["confusion_snr_ge10"], dtype=float); cm /= cm.sum(1, keepdims=True)
f, ax = fig(height=3.4)
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(24)); ax.set_yticks(range(24))
ax.set_xticklabels(classes, rotation=90, fontsize=6); ax.set_yticklabels(classes, fontsize=6)
ax.tick_params(length=2, pad=1)
ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
for spine in ax.spines.values(): spine.set_visible(False)
cb = f.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label("fraction of the true class")
cb.outline.set_visible(False)
save(f, os.path.join(OUT, "confusion_matrix.png"))
print("ok:", OUT)
