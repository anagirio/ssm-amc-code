"""Block diagram of the model (Fig. 2), drawn at single-column width so every label is legible."""
import os, sys
from figstyle import *
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "block_diagram.png"
BOXES = [  # (title, detail, face colour, text colour)
    ("I/Q input", "1024 complex samples (2 channels)", "#e4e3df", INK),
    ("CNN frontend", "4 strided conv stages (kernel 7, stride 2)\nGroupNorm + GELU  →  64 tokens × 128", NONE, INK),
    ("Sequence mixer: 4 SSM blocks", "bidirectional selective scan\n(controls: none · conv · attention)", SSM, "white"),
    ("Mean pooling + linear", "average over tokens  →  24 classes", "#e4e3df", INK),
]
f, ax = plt.subplots(figsize=(COL_W, 3.2))
f.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
h, gap, top = 0.205, 0.045, 0.98
for i, (t, d, fc, tc) in enumerate(BOXES):
    y = top - h - i * (h + gap)
    ax.add_patch(FancyBboxPatch((0.04, y), 0.92, h, boxstyle="round,pad=0,rounding_size=0.025",
                                fc=fc, ec="none"))
    ax.text(0.5, y + h * 0.74, t, ha="center", va="center", color=tc, fontsize=8, weight="bold")
    ax.text(0.5, y + h * 0.33, d, ha="center", va="center", color=tc, fontsize=7, linespacing=1.25)
    if i < len(BOXES) - 1:
        ax.annotate("", xy=(0.5, y - gap + 0.006), xytext=(0.5, y - 0.006),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
save(f, OUT)
