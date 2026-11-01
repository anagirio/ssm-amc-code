"""Is the dip of the multi-rate curves at factor 1.0 an artefact of interpolation?

Multi-rate training draws the factor uniformly in [0.5, 2.0], so a model almost never sees a
signal that was not interpolated. At factor 1.0 the test signal is passed through untouched.
If that matters, accuracy at 0.99 and 1.01 (practically the same rate, but interpolated) will be
clearly higher than at 1.0 for multi-rate models, and about the same for fixed-rate models.
Each model is tested on the test split of its own seed.
"""
import os, json, importlib.util
import numpy as np, torch, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("MAMBA_CKPT_DIR", HERE)
FACTORS = [0.98, 0.99, 1.0, 1.01, 1.02]
N = 20000


def load_variant(v, seed=42):
    os.environ["MV_VARIANT"] = v; os.environ["MV_DATASET"] = "radioml"; os.environ["MV_SEED"] = str(seed)
    sp = importlib.util.spec_from_file_location(f"mv_{v}_{seed}", os.path.join(HERE, "mamba_variants.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def resample(X, f, L=1024):
    # same transform as invariance_radioml.py: linear interpolation, then crop or zero-pad to L
    n = X.shape[1]; tgt = max(8, int(round(n * f)))
    idx = np.linspace(0, n - 1, tgt); i0 = np.floor(idx).astype(int); i1 = np.minimum(i0 + 1, n - 1)
    w = (idx - i0).astype(np.float32)[None, :, None]
    y = X[:, i0] * (1 - w) + X[:, i1] * w
    if tgt >= L:
        return y[:, :L]
    out = np.zeros((X.shape[0], L, 2), np.float32); out[:, :tgt] = y; return out


res = {}
for seed in (42, 43, 44):
    base = load_variant("BASE", seed); dev = base.DEVICE
    _, _, idx = base.build_balanced_datasets(base.DATASET_PATH, seed)
    rng = np.random.default_rng(0); rng.shuffle(idx); idx = np.sort(idx[:N])
    with h5py.File(base.DATASET_PATH, "r") as f:
        X = f["X"][idx].astype(np.float32); Y = f["Y"][idx].argmax(1)
    sfx = "" if seed == 42 else f"_s{seed}"
    for v in ["TRANSFORMER", "TRANSFORMER_MULTIRATE", "CONVMIX", "CONVMIX_MULTIRATE"]:
        ck = os.path.join(CKPT, f"mv_{v.lower()}_radioml{sfx}_IQ.pt")
        if not os.path.exists(ck):
            continue
        m = load_variant(v, seed)
        model = m.MambaClassifier(input_mode="IQ").to(dev).eval()
        model.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
        accs = []
        with torch.no_grad():
            for fac in FACTORS:
                Xf = X if fac == 1.0 else resample(X, fac)
                p = np.concatenate([model(torch.from_numpy(Xf[i:i + 500]).to(dev)).argmax(-1).cpu().numpy()
                                    for i in range(0, N, 500)])
                accs.append(float((p == Y).mean() * 100))
        res.setdefault(v, []).append(accs)
        print(f"seed {seed} {v:22s} " + "  ".join(f"x{f}={a:.1f}" for f, a in zip(FACTORS, accs)), flush=True)

print("\nmedia das 3 seeds:")
for v, a in res.items():
    a = np.array(a).mean(0)
    print(f"  {v:22s} " + "  ".join(f"x{f}={x:.1f}" for f, x in zip(FACTORS, a)))
json.dump({"factors": FACTORS, "curves": res}, open(os.path.join(CKPT, "interp_artifact.json"), "w"), indent=2)
