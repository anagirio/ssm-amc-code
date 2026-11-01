"""Accuracy vs input length at test time (from varlen_length.json, written by varlen_length.py)."""
import os, sys, json
from figstyle import *

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
d = json.load(open(os.path.join(HERE, "varlen_length.json")))
L = d["lengths"]
f, ax = fig(height=2.5)
ax.axvspan(256, 1024, color="#f1efe8", lw=0)                       # lengths seen in random-length training
ax.plot(L, d["curves"]["BASE"], color=SSM, ls=(0, (2.5, 1.5)), marker="o", mec="white", mew=0.5, label="SSM trained at 1024 samples")
ax.plot(L, d["curves"]["VARLEN"], color=SSM, marker="s", mec="white", mew=0.5, label="SSM trained with random length (shaded range)")
ax.set_xlabel("input length at test time (samples)"); ax.set_ylabel("test accuracy (%)")
ax.set_xticks(L); ax.set_xlim(100, 1050); ax.set_ylim(10, 55); ax.grid()
legend_above(ax)
save(f, os.path.join(OUT, "varlen_length.png"))
