"""Andrea #3 (demonstracao que faltava): o modelo classifica com input de
COMPRIMENTO variavel? Testamos os modelos treinados no RadioML com sinais truncados
em varios tamanhos (128..1024 amostras). Previsao: o VARLEN (treinado com comprimento
aleatorio) fica ESTAVEL; o modelo normal (BASE) DESABA com input curto.
Roda LOCAL (GPU MPS do Mac), usando o RadioML em 6_DADOS_BRUTOS/.
"""
import os, json, importlib.util
import numpy as np, torch, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "GOLD_XYZ_OSC.0001_1024.hdf5")          # RadioML next to the script
MODELS_DIR = os.environ.get("MAMBA_CKPT_DIR", HERE)                # trained .pt files
LENGTHS = [128, 256, 384, 512, 768, 1024]
BS = 256
MODELS = ["BASE", "VARLEN", "MULTIRATE"]

def load_variant(v):
    os.environ["MV_VARIANT"] = v; os.environ["MV_DATASET"] = "radioml"
    sp = importlib.util.spec_from_file_location(f"mv_{v}", os.path.join(HERE, "mamba_variants.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

base = load_variant("BASE"); DEV = base.DEVICE
print("device:", DEV)
_, _, test_idx = base.build_balanced_datasets(DATA, base.SEED)
rng = np.random.default_rng(0); rng.shuffle(test_idx); test_idx = np.sort(test_idx[:40000])
with h5py.File(DATA, "r") as f:
    X = f["X"][test_idx][:, :1024, :].astype(np.float32)
    Y = torch.from_numpy(f["Y"][test_idx].argmax(1))
print("teste:", X.shape)

def acc_at(model, L):
    ok = 0; model.eval()
    with torch.no_grad():
        for i in range(0, len(X), BS):
            xb = torch.from_numpy(X[i:i+BS, :L, :]).float().to(DEV)
            ok += (model(xb).argmax(-1).cpu() == Y[i:i+BS]).sum().item()
    return 100.0 * ok / len(X)

results = {}
for v in MODELS:
    ck = os.path.join(MODELS_DIR, f"mv_{v.lower()}_radioml_IQ.pt")
    if not os.path.exists(ck):
        print("pulando (sem modelo):", v); continue
    m = load_variant(v); model = m.MambaClassifier(input_mode="IQ").to(DEV)
    model.load_state_dict(torch.load(ck, map_location=DEV))
    curve = []
    for L in LENGTHS:
        try:
            curve.append(acc_at(model, L))
        except Exception as e:
            print(f"  {v} L={L} erro: {str(e)[:60]}"); curve.append(None)
    results[v] = curve
    print(f"{v:10s}: " + " ".join(f"{a:.1f}" if a is not None else "  x " for a in curve))
    del model

if results:
    plt.figure(figsize=(9.5, 6))
    nice = {"BASE": ("Normal (comprimento fixo 1024)", "#c0392b", "--"),
            "VARLEN": ("VARLEN (comprimento aleatorio no treino)", "#e67e22", "-"),
            "MULTIRATE": ("MULTIRATE", "#8e44ad", "-")}
    for v, curve in results.items():
        lab, col, ls = nice.get(v, (v, "gray", "-"))
        xs = [L for L, a in zip(LENGTHS, curve) if a is not None]
        ys = [a for a in curve if a is not None]
        plt.plot(xs, ys, "o-", lw=2.4, ms=7, color=col, ls=ls, label=lab)
    plt.xlabel("comprimento do sinal de entrada (nº de amostras)", fontsize=11)
    plt.ylabel("acuracia no RadioML (%)", fontsize=11)
    plt.title("Comprimento de entrada VARIAVEL (Andrea #3): o VARLEN aguenta input curto?\n(curva mais ALTA e PLANA = classifica sem depender do tamanho)")
    plt.grid(alpha=.3); plt.legend()
    plt.tight_layout()
    out = os.path.join(HERE, "varlen_length.png")
    plt.savefig(out, dpi=150)
    json.dump({"lengths": LENGTHS, "curves": results},
              open(os.path.join(HERE, "varlen_length.json"), "w"), indent=2)
    print("PNG:", out)
print("PRONTO")
