"""Robustness to the sampling rate (RadioML, in-domain).
Models are trained on RadioML at a fixed rate, then tested on RadioML resampled to
several rates (factors 0.6..1.5). A model that is robust to the rate (adaptive/
physical delta, varlen, multirate) should stay flat across factors; the fixed-delta
baseline should drop off. Train at one rate, test at many.
"""
import os, json, importlib.util
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.environ.get("MAMBA_CKPT_DIR", HERE)
L = 1024; BS = 256
FACTORS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]   # test-time sampling rates
if os.environ.get("INV_FACTORS"):                     # e.g. a finer grid around the native rate
    FACTORS = [float(f) for f in os.environ["INV_FACTORS"].split(",")]
# INV_RESAMPLER=polyphase uses scipy's polyphase resampler with an anti-aliasing filter (the standard
# signal-processing method); 'linear' is the plain interpolation used in the first version
RESAMPLER = os.environ.get("INV_RESAMPLER", "linear")
N_TEST = int(os.environ.get("INV_N", "60000"))
# models to compare (each needs its .pt trained on radioml).
# INV_SET=arch compares the three sequence mixers with and without multi-rate training.
INV_SET = os.environ.get("INV_SET", "delta").lower()
if INV_SET == "arch":
    MODELS = ["BASE", "MULTIRATE", "TRANSFORMER", "TRANSFORMER_MULTIRATE",
              "CONVMIX", "CONVMIX_MULTIRATE"]
else:
    MODELS = ["BASE", "DELTA_ADAPT", "PHYS_DELTA", "VARLEN", "MULTIRATE"]
if os.environ.get("INV_MODELS"):                      # explicit list, e.g. TRANSFORMER,TRANSFORMER_NOPOS
    MODELS = [v.strip().upper() for v in os.environ["INV_MODELS"].split(",")]
OUT_NAME = "invariance_radioml" + ("_arch" if INV_SET == "arch" else "") + os.environ.get("INV_TAG", "")
# INV_ONLY=A,B re-evaluates only those models and keeps the others from the existing JSON
# (avoids re-running the slow SSM evaluations when only new seeds of the controls were added)
_ONLY = [v.strip().upper() for v in os.environ.get("INV_ONLY", "").split(",") if v.strip()]
_PREV = {}
if _ONLY and os.path.exists(os.path.join(HERE, OUT_NAME + ".json")):
    _PREV = json.load(open(os.path.join(HERE, OUT_NAME + ".json"))).get("curves_per_seed", {})

def load_variant(v):
    os.environ["MV_VARIANT"] = v; os.environ["MV_DATASET"] = "radioml"
    sp = importlib.util.spec_from_file_location(f"mv_{v}", os.path.join(HERE, "mamba_variants.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

base = load_variant("BASE"); DEV = base.DEVICE
import h5py
_TEST = {}


def use_test_set(seed):
    """point X, Y at 60 000 test examples of the split made with this seed. the split depends
    on the seed, so a seed-43 model must be tested on the seed-43 test set, never on 42's."""
    global X, Y
    if seed not in _TEST:
        print(f"[1] building the RadioML test set (seed {seed})...", flush=True)
        _, _, idx = base.build_balanced_datasets(base.DATASET_PATH, seed)
        rng = np.random.default_rng(0); rng.shuffle(idx); idx = np.sort(idx[:N_TEST])
        with h5py.File(base.DATASET_PATH, "r") as f:
            _TEST[seed] = (f["X"][idx][:, :L, :].astype(np.float32),
                           torch.from_numpy(f["Y"][idx].argmax(1)))
    X, Y = _TEST[seed]


use_test_set(base.SEED)

def resample(Xb, factor, out=L):
    # resample the whole signal by 'factor' with linear interp, then crop or pad back to 'out'.
    # symmetric, so it matches the multirate training augmentation both ways.
    n = Xb.shape[1]
    if factor == 1.0:
        return Xb[:, :out, :]
    if RESAMPLER == "polyphase":
        from fractions import Fraction
        from scipy.signal import resample_poly
        fr = Fraction(factor).limit_denominator(100)           # e.g. 0.99 -> 99/100
        y = resample_poly(Xb, fr.numerator, fr.denominator, axis=1).astype(np.float32)
        tgt = y.shape[1]
    else:
        tgt = max(8, int(round(n * factor)))
        idx = np.clip(np.arange(tgt) * (n - 1) / max(tgt - 1, 1), 0, n - 1)
        i0 = np.floor(idx).astype(int); i1 = np.minimum(i0 + 1, n - 1)
        w = (idx - i0).astype(np.float32)[None, :, None]
        y = Xb[:, i0, :] * (1 - w) + Xb[:, i1, :] * w      # linear interpolation
    if tgt >= out:
        return y[:, :out, :]
    z = np.zeros((Xb.shape[0], out, Xb.shape[2]), np.float32)
    z[:, :tgt, :] = y
    return z

def acc_at(model, factor):
    ok = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), BS):
            xb = resample(X[i:i+BS], factor, L)
            p = model(torch.from_numpy(xb).float().to(DEV)).argmax(-1).cpu()
            ok += (p == Y[i:i+BS]).sum().item()
    return 100.0 * ok / len(X)

# average all available seeds per model
# (mv_<v>_radioml_IQ.pt = seed 42; mv_<v>_radioml_s43_IQ.pt = 43; etc.)
SEEDS = ["", "_s43", "_s44"]   # "" = seed 42 (default)
results = {}       # mean curve per model
results_seeds = {} # every seed's curve (for mean/std)
for v in MODELS:
    curves = []
    if _ONLY and v not in _ONLY and v in _PREV:      # reuse the curves already computed
        results[v] = np.array(_PREV[v]).mean(0).tolist(); results_seeds[v] = _PREV[v]
        print(f"  {v:12s} (reused, {len(_PREV[v])} seeds)"); continue
    for sfx in SEEDS:
        ck = os.path.join(REPO, f"mv_{v.lower()}_radioml{sfx}_IQ.pt")
        if not os.path.exists(ck):
            continue
        use_test_set(int(sfx[2:]) if sfx else 42)   # "" = seed 42, "_s43" = 43, ...
        m = load_variant(v); model = m.MambaClassifier(input_mode="IQ").to(DEV)
        model.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True))
        curves.append([acc_at(model, fac) for fac in FACTORS])
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if not curves:
        print(f"  [skip] {v}: no trained model"); continue
    arr = np.array(curves)                       # (n_seeds, n_factors)
    results[v] = arr.mean(0).tolist()
    results_seeds[v] = curves
    sd = arr.std(0).mean() if len(curves) > 1 else 0.0
    print(f"  {v:12s} ({len(curves)} seeds): " + " ".join(f"{a:.1f}" for a in arr.mean(0)) +
          f"   | mean={arr.mean():.1f} std_across_seeds={sd:.2f}")

if results:
    plt.figure(figsize=(10, 6.5))
    nice = {"BASE": ("fixed delta (baseline)", "#c0392b", "--"),
            "DELTA_ADAPT": ("adaptive delta (learned)", "#27ae60", "-"),
            "PHYS_DELTA": ("physical delta (bandwidth)", "#1a3a8a", "-"),
            "VARLEN": ("variable length", "#e67e22", "-"),
            "MULTIRATE": ("random-rate augmentation", "#8e44ad", "-")}
    if INV_SET == "arch":   # one hue per mixer; dashed = fixed rate, solid = multi-rate
        nice = {"BASE": ("SSM, fixed rate", "#2a78d6", "--"),
                "MULTIRATE": ("SSM, multi-rate", "#2a78d6", "-"),
                "TRANSFORMER": ("Transformer, fixed rate", "#eb6834", "--"),
                "TRANSFORMER_MULTIRATE": ("Transformer, multi-rate", "#eb6834", "-"),
                "CONVMIX": ("conv mixer, fixed rate", "#1baf7a", "--"),
                "CONVMIX_MULTIRATE": ("conv mixer, multi-rate", "#1baf7a", "-")}
    for v, curve in results.items():
        lab, col, ls = nice.get(v, (v, "gray", "-"))
        arr = np.array(results_seeds[v]); ns = len(arr)
        mean = arr.mean(0)
        plt.plot(FACTORS, mean, marker="o", lw=2.4, color=col, ls=ls,
                 label=f"{lab}  (mean={mean.mean():.1f}" + (f", {ns} seeds)" if ns > 1 else ")"))
        if ns > 1:   # +-std band across seeds
            sd = arr.std(0)
            plt.fill_between(FACTORS, mean - sd, mean + sd, color=col, alpha=.15)
    plt.axvline(1.0, color="k", lw=.7, ls=":"); plt.text(1.0, plt.ylim()[0], " training rate", fontsize=8)
    plt.xlabel("resampling factor (test-time sampling rate)", fontsize=11)
    plt.ylabel("accuracy (%)", fontsize=11)
    plt.title("Robustness to the sampling rate (symmetric resampling, averaged over seeds)\n"
              "train at one rate, test at many — a higher, flatter curve = more robust")
    plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(HERE, OUT_NAME + ".png"), dpi=150)
    json.dump({"factors": FACTORS, "curves": results, "curves_per_seed": results_seeds},
              open(os.path.join(HERE, OUT_NAME + ".json"), "w"), indent=2)
    print(f"PNG: {OUT_NAME}.png | JSON: {OUT_NAME}.json")
print("done")
