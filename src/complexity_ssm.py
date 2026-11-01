# complexity vs accuracy for the SSM.
# the scan is O(N) in the sequence length, attention is O(N^2). this shows the trade-off:
# fewer tokens = cheaper model, and how much accuracy that costs.
# runs locally: just builds the model and measures params/FLOPs, no dataset needed.
# accuracies come from the token experiments (32->36.46, 64->48.95, 256->56.62).
import os, json, importlib.util
import numpy as np, torch
from thop import profile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SIG_LEN = 1024
# variant -> (name, n_tokens, measured accuracy)
CFG = [("TOKENS32", 32, 36.46), ("BASE", 64, 48.95), ("TOKENS256", 256, 56.62)]

def load_variant(v):
    os.environ["MV_VARIANT"] = v; os.environ["MV_DATASET"] = "radioml"
    sp = importlib.util.spec_from_file_location(f"mv_{v}", os.path.join(HERE, "mamba_variants.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

rows = []
for v, ntok, acc in CFG:
    m = load_variant(v)
    model = m.MambaClassifier(input_mode="IQ").eval()
    dummy = torch.randn(1, SIG_LEN, 2)
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    rows.append({"variant": v, "tokens": ntok, "accuracy": acc,
                 "params_M": params/1e6, "GFLOPs": flops/1e9})
    print(f"{v:10s} tokens={ntok:4d} acc={acc:5.2f}%  params={params/1e6:.2f}M  FLOPs={flops/1e9:.3f}G")

json.dump(rows, open(os.path.join(HERE, "complexity_ssm.json"), "w"), indent=2)

# theoretical scaling: SSM O(N) vs attention O(N^2), normalized at N=64
Ns = np.array([32, 64, 128, 256, 512])
ssm_cost   = Ns / 64.0            # linear
trans_cost = (Ns / 64.0) ** 2     # quadratic

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

# left: cost vs number of tokens (SSM linear vs Transformer quadratic)
ax1.plot(Ns, trans_cost, "o-", color="#c0392b", lw=2.4, label="Attention (Transformer)  O(N$^2$)")
ax1.plot(Ns, ssm_cost, "o-", color="#12395E", lw=2.4, label="SSM (Mamba)  O(N)")
ax1.set_xlabel("number of tokens (N)"); ax1.set_ylabel("relative cost (norm. at N=64)")
ax1.set_title("Why the SSM scales better: O(N) vs O(N$^2$)", fontweight="bold")
ax1.grid(alpha=.3); ax1.legend()
for x in Ns:
    ax1.annotate(f"{(x/64)**2:.0f}x", (x, (x/64)**2), textcoords="offset points",
                 xytext=(0, 8), fontsize=8, color="#c0392b", ha="center")

# right: complexity (real FLOPs of our model) vs accuracy
gf = [r["GFLOPs"] for r in rows]; ac = [r["accuracy"] for r in rows]; tk = [r["tokens"] for r in rows]
ax2.plot(gf, ac, "o-", color="#0E7C7B", lw=2.4, ms=9)
for g, a, t in zip(gf, ac, tk):
    ax2.annotate(f"{t} tokens", (g, a), textcoords="offset points", xytext=(8, -4), fontsize=9)
ax2.set_xlabel("complexity — GFLOPs (real model)"); ax2.set_ylabel("RadioML accuracy (%)")
ax2.set_title("Trade-off: complexity vs accuracy (our SSM)", fontweight="bold")
ax2.grid(alpha=.3)

fig.suptitle("Efficiency through complexity", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "complexity_ssm.png"), dpi=150, bbox_inches="tight")
print("done")
