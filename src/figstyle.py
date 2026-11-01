"""One style for every paper figure: same widths, fonts, colours, lines and legend placement.
Figures are saved at their final printed width, so 8 pt in the file is 8 pt in the PDF."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_W, PAGE_W = 3.5, 7.16                      # IEEE column and page width (in)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SSM, TRF, CONV, NONE = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29b"   # one colour per mixer, everywhere
RADIOML, MODFORGE = "#173a63", "#9a7fd1"                             # one colour per dataset
LW, MS = 1.5, 3.5

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False, "axes.titlesize": 8, "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": LW, "lines.markersize": MS, "savefig.dpi": 300, "savefig.facecolor": "white",
})

def fig(width=COL_W, height=2.3, **kw):
    return plt.subplots(figsize=(width, height), layout="constrained", **kw)

def legend_above(ax, ncol=1):
    """Every legend sits above the plot area, outside the data."""
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=ncol, borderaxespad=0,
              handlelength=2.4, columnspacing=1.2)

def save(f, path):
    f.savefig(path); plt.close(f); print("saved", path)
