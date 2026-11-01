"""Is the fixed-rate Transformer tuned to absolute token positions? Shift the test signal in time by
k samples (no change of rate: drop the first k samples and zero-pad the end) and compare mixers.
A model that depends on absolute position loses accuracy with the shift; one that does not, keeps it.
Seed 42 test split, same N as check_resampler."""
import os, json, importlib.util
import numpy as np, torch, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("MAMBA_CKPT_DIR", HERE)
SHIFTS = [0, 1, 2, 4, 8, 16, 32]
N = int(os.environ.get("SH_N", "10000"))
MODELS = os.environ.get("SH_MODELS", "BASE,TRANSFORMER,CONVMIX").split(",")


def load_variant(v):
    os.environ["MV_VARIANT"] = v; os.environ["MV_DATASET"] = "radioml"; os.environ["MV_SEED"] = "42"
    sp = importlib.util.spec_from_file_location(f"mv_{v}", os.path.join(HERE, "mamba_variants.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def shift(X, k):
    if k == 0:
        return X
    out = np.zeros_like(X); out[:, :-k] = X[:, k:]; return out


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
    accs = []
    with torch.no_grad():
        for k in SHIFTS:
            Xs = shift(X, k)
            p = np.concatenate([model(torch.from_numpy(Xs[i:i + 500]).to(dev)).argmax(-1).cpu().numpy()
                                for i in range(0, N, 500)])
            accs.append(float((p == Y).mean() * 100))
    res[v] = accs
    print(f"{v:20s} " + "  ".join(f"+{k}={a:.1f}" for k, a in zip(SHIFTS, accs)), flush=True)
json.dump({"shifts": SHIFTS, "curves": res}, open(os.path.join(CKPT, os.environ.get("SH_OUT", "shift_check.json")), "w"), indent=2)
