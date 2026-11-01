# acuracia de TESTE para checkpoints treinados no RadioML.
# cada modelo e' avaliado no conjunto de teste da SUA seed: o split balanceado depende da seed
# (MV_SEED), entao avaliar um modelo de seed 43 no teste da seed 42 misturaria exemplos que ele
# viu no treino. a seed vem do sufixo do checkpoint (_s43, _s44; sem sufixo = 42).
# uso: MAMBA_CKPT_DIR=<pasta com os .pt> python eval_test.py VARIANTE[:sufixo] ...
#      ex.: python eval_test.py BASE BASE:_s43 NO_SSM TRANSFORMER
import os, re, sys, json, time, importlib.util
import numpy as np, torch, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("MAMBA_CKPT_DIR", HERE)
OUT  = os.path.join(CKPT, "test_accuracy.json")


def load_variant(v, seed):
    os.environ["MV_VARIANT"] = v; os.environ["MV_DATASET"] = "radioml"; os.environ["MV_SEED"] = str(seed)
    sp = importlib.util.spec_from_file_location(f"mv_{v}_{seed}", os.path.join(HERE, "mamba_variants.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def seed_of(sfx):
    s = re.search(r"_s(\d+)", sfx)
    return int(s.group(1)) if s else 42


def read_test(seed):
    base = load_variant("BASE", seed)
    _, _, idx = base.build_balanced_datasets(base.DATASET_PATH, seed)
    idx = np.sort(idx); t0 = time.time()
    with h5py.File(base.DATASET_PATH, "r") as f:
        X = np.empty((len(idx), 1024, 2), dtype=np.float32)
        for i in range(0, len(idx), 20000):
            X[i:i + 20000] = f["X"][idx[i:i + 20000]]
        Y = f["Y"][idx].argmax(1)
        Z = np.round(f["Z"][idx].squeeze()).astype(int)
    print(f"[teste seed {seed}] {len(idx):,} exemplos lidos em {time.time() - t0:.0f}s", flush=True)
    return base.DEVICE, X, Y, Z


results = json.load(open(OUT)) if os.path.exists(OUT) else {}
specs = [s.partition(":")[::2] for s in sys.argv[1:]]           # [(variante, sufixo)]
for seed in sorted({seed_of(sfx) for _, sfx in specs}):         # um conjunto de teste por seed
    DEV, X, Y, Z = read_test(seed)
    for v, sfx in [s for s in specs if seed_of(s[1]) == seed]:
        key = f"{v}{sfx}"
        ck = os.path.join(CKPT, f"mv_{v.lower()}_radioml{sfx}_IQ.pt")
        if not os.path.exists(ck):
            print(f"  [skip] {key}: {os.path.basename(ck)} nao existe", flush=True); continue
        m = load_variant(v, seed)
        model = m.MambaClassifier(input_mode="IQ").to(DEV).eval()
        model.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True))
        preds = np.empty(len(Y), dtype=np.int64); t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(Y), 512):
                preds[i:i + 512] = model(torch.from_numpy(X[i:i + 512]).to(DEV)).argmax(-1).cpu().numpy()
        ok = preds == Y
        per_snr = {int(s): float(ok[Z == s].mean() * 100) for s in np.unique(Z)}
        hi = Z >= 10                                    # confusion matrix at high SNR (rows = true)
        cm = np.zeros((24, 24), dtype=np.int64)
        np.add.at(cm, (Y[hi], preds[hi]), 1)
        results[key] = {"test_acc": float(ok.mean() * 100), "test_acc_snr_ge10": float(ok[hi].mean() * 100),
                        "test_seed": seed, "per_snr": per_snr, "confusion_snr_ge10": cm.tolist()}
        print(f"  {key:26s} teste={ok.mean()*100:6.2f}%  SNR>=10dB={ok[hi].mean()*100:6.2f}%  "
              f"[{time.time() - t0:.0f}s]", flush=True)
        json.dump(results, open(OUT, "w"), indent=2)
        del model
    del X, Y, Z
print("salvo", OUT)
