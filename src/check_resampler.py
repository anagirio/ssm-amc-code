"""Separate the effect of the sampling rate from the effect of the resampling method.

The robustness curves used linear interpolation, which also low-pass filters the signal. A 1%
rate change is physically negligible, so any large drop at 0.99 or 1.01 comes from the resampler,
not from the rate. Here the same fixed-rate and multi-rate models are tested with linear
interpolation and with a polyphase resampler with an anti-aliasing filter (scipy resample_poly),
the standard method in signal processing. Seed 42 test split.
"""
import os, sys, json, importlib.util
import numpy as np, torch, h5py
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("MAMBA_CKPT_DIR", HERE)
FACTORS = {0.8: (4, 5), 0.9: (9, 10), 0.99: (99, 100), 1.0: (1, 1), 1.01: (101, 100), 1.1: (11, 10), 1.25: (5, 4)}
N = int(os.environ.get("RS_N", "10000"))
MODELS = os.environ.get("RS_MODELS", "TRANSFORMER,CONVMIX,BASE,TRANSFORMER_MULTIRATE,MULTIRATE").split(",")


def load_variant(v):
    os.environ["MV_VARIANT"] = v; os.environ["MV_DATASET"] = "radioml"; os.environ["MV_SEED"] = "42"
    sp = importlib.util.spec_from_file_location(f"mv_{v}", os.path.join(HERE, "mamba_variants.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def fit(y, L=1024):
    """crop or zero-pad to L samples (same as the robustness test)"""
    if y.shape[1] >= L:
        return y[:, :L]
    out = np.zeros((y.shape[0], L, 2), np.float32); out[:, :y.shape[1]] = y; return out


def linear(X, f):
    n = X.shape[1]; tgt = max(8, int(round(n * f)))
    idx = np.linspace(0, n - 1, tgt); i0 = np.floor(idx).astype(int); i1 = np.minimum(i0 + 1, n - 1)
    w = (idx - i0).astype(np.float32)[None, :, None]
    return fit(X[:, i0] * (1 - w) + X[:, i1] * w)


def polyphase(X, f):
    up, down = FACTORS[f]
    return fit(resample_poly(X, up, down, axis=1).astype(np.float32))


base = load_variant("BASE"); dev = base.DEVICE
_, _, idx = base.build_balanced_datasets(base.DATASET_PATH, 42)
rng = np.random.default_rng(0); rng.shuffle(idx); idx = np.sort(idx[:N])
with h5py.File(base.DATASET_PATH, "r") as f:
    X = f["X"][idx].astype(np.float32); Y = f["Y"][idx].argmax(1)

res = {}
for v in MODELS:
    m = load_variant(v)
    model = m.MambaClassifier(input_mode="IQ").to(dev).eval()
    model.load_state_dict(torch.load(os.path.join(CKPT, f"mv_{v.lower()}_radioml_IQ.pt"), map_location=dev, weights_only=True))
    for name, fn in (("linear", linear), ("polyphase", polyphase)):
        accs = []
        with torch.no_grad():
            for fac in FACTORS:
                Xf = X if fac == 1.0 else fn(X, fac)
                p = np.concatenate([model(torch.from_numpy(Xf[i:i + 500]).to(dev)).argmax(-1).cpu().numpy()
                                    for i in range(0, N, 500)])
                accs.append(float((p == Y).mean() * 100))
        res[f"{v}|{name}"] = accs
        print(f"{v:22s} {name:9s} " + "  ".join(f"x{f}={a:.1f}" for f, a in zip(FACTORS, accs)), flush=True)
json.dump({"factors": list(FACTORS), "curves": res}, open(os.path.join(CKPT, "resampler_check.json"), "w"), indent=2)
