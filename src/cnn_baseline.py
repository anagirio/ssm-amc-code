import json, math, os, time, copy
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "GOLD_XYZ_OSC.0001_1024.hdf5")
CLASSES_PATH = os.path.join(BASE_DIR, "classes-fixed.json")
OUT_DIR      = BASE_DIR
# Checkpoint/modelos salvos em pasta PERSISTENTE (home) — sobrevive a preempcao
# e guarda o CNN treinado no RadioML p/ depois testar no modforge (teste cruzado).
CKPT_DIR     = os.environ.get("MAMBA_CKPT_DIR", OUT_DIR)

NUM_CLASSES = 24
BATCH_SIZE  = 256
NUM_EPOCHS  = 12
LR          = 5e-4
GRAD_CLIP   = 1.0
SEED        = 42
SNR_MIN     = -4          # filtro de SNR APENAS no treino
PATIENCE    = 4           # early stopping: para se nao melhorar por 4 epocas
TRAIN_RATIO = 0.70
EVAL_RATIO  = 0.20
TEST_RATIO  = 0.10

# rodar os dois tipos de entrada e comparar
INPUT_MODES = ["IQ", "AP", "FFT"]   # IQ cru ; AP amplitude/fase ; FFT frequencia

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")


# ─── Carregador de batches ────────────────────────────────────────────────────
class RadioMLBatchLoader:
    # Cache compartilhado: dataset inteiro na RAM (X em float16, ~10.5 GB),
    # carregado UMA vez. Batches viram fatias na RAM (rapido na GPU).
    _CACHE = {}

    def __init__(self, hdf5_path, indices, batch_size, shuffle=True, seed=0):
        self.path       = hdf5_path
        self.indices    = np.sort(indices)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.seed       = seed
        self.epoch      = 0

    @classmethod
    def _load_ram(cls, path):
        if path not in cls._CACHE:
            print("  [RAM] carregando dataset inteiro na memoria (float16, uma vez)...")
            t0 = time.time()
            with h5py.File(path, "r") as f:
                Xd = f["X"]
                X  = np.empty(Xd.shape, dtype=np.float16)
                CH = 200000
                for i in range(0, Xd.shape[0], CH):
                    X[i:i + CH] = Xd[i:i + CH]
                Y = f["Y"][:].argmax(axis=1).astype(np.int64)
                Z = f["Z"][:].squeeze().astype(np.float32)
            cls._CACHE[path] = (X, Y, Z)
            print(f"  [RAM] pronto: X={X.shape} float16 ({X.nbytes/1e9:.1f} GB) "
                  f"em {time.time()-t0:.0f}s")
        return cls._CACHE[path]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __iter__(self):
        X, Y, Z = self._load_ram(self.path)
        order = self.indices.copy()
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(order)                       # embaralha por AMOSTRA
        n = len(order)
        for i in range(0, n, self.batch_size):
            idx = order[i:i + self.batch_size]
            yield (torch.from_numpy(X[idx]).float(),  # float16 -> float32
                   torch.from_numpy(Y[idx]),
                   torch.from_numpy(Z[idx]))


# ─── Split BALANCEADO (todas as classes e todos os SNRs) ──────────────────────
def build_balanced_datasets(path, seed):
    """
    Garante que eval e teste tenham a MESMA quantidade de cada
    (modulacao x SNR). Assim a acuracia nao fica enviesada por uma classe
    ou SNR que aparece mais. O treino mantem o filtro SNR >= SNR_MIN.
    """
    print("Lendo Y e Z para montar split balanceado...")
    with h5py.File(path, "r") as f:
        labels = f["Y"][:].argmax(axis=1)
        snrs   = np.round(f["Z"][:].squeeze()).astype(int)

    rng = np.random.default_rng(seed)
    train_parts, eval_parts, test_parts = [], [], []

    for cls in range(NUM_CLASSES):
        for snr in np.unique(snrs):
            cell = np.where((labels == cls) & (snrs == snr))[0]
            rng.shuffle(cell)
            n   = len(cell)
            ntr = int(n * TRAIN_RATIO)
            nev = int(n * EVAL_RATIO)
            train_parts.append(cell[:ntr])
            eval_parts.append(cell[ntr:ntr + nev])
            test_parts.append(cell[ntr + nev:])

    train_idx = np.concatenate(train_parts)
    eval_idx  = np.concatenate(eval_parts)
    test_idx  = np.concatenate(test_parts)

    # treino: descarta SNR muito baixo (sinal = ruido puro)
    train_idx = train_idx[snrs[train_idx] >= SNR_MIN]

    print(f"Total samples         : {len(labels):,}")
    print(f"Train (SNR >= {SNR_MIN}dB) : {len(train_idx):,}")
    print(f"Eval  (balanceado)    : {len(eval_idx):,}")
    print(f"Teste (balanceado)    : {len(test_idx):,}")
    return train_idx, eval_idx, test_idx


# ─── Transformacao IQ -> Amplitude/Fase ───────────────────────────────────────
def iq_to_amp_phase(x):
    """
    x: (B, L, 2) com canais [I, Q]
    retorna (B, L, 2) com canais [amplitude, fase]
        amplitude = sqrt(I^2 + Q^2)   -> "forca" do sinal
        fase      = atan2(Q, I)       -> "angulo" do sinal
    """
    I = x[..., 0]
    Q = x[..., 1]
    amp = torch.sqrt(I**2 + Q**2 + 1e-12)
    pha = torch.atan2(Q, I)
    return torch.stack([amp, pha], dim=-1)


def iq_to_fft(x):
    """
    x: (B, L, 2) com canais [I, Q]  (sinal complexo x = I + jQ)
    retorna (B, L, 2) com canais [parte_real, parte_imag] do ESPECTRO (FFT).
        Aplica FFT no sinal complexo e centraliza a freq. 0 (fftshift).
        A FFT leva o sinal pro dominio da FREQUENCIA — captura o "conteudo
        espectral" da modulacao, que e complementar ao dominio do tempo (IQ/AP).
    """
    comp = torch.complex(x[..., 0], x[..., 1])          # (B, L) complexo
    spec = torch.fft.fft(comp, dim=-1)                  # FFT (B, L) complexo
    spec = torch.fft.fftshift(spec, dim=-1)             # freq 0 no centro
    return torch.stack([spec.real, spec.imag], dim=-1)  # (B, L, 2)


# ─── Modelo (ResNet 1D + GroupNorm) ───────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, groups=8):
        super().__init__()
        pad = kernel // 2
        self.conv1    = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad)
        self.gn1      = nn.GroupNorm(groups, out_ch)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel, padding=pad)
        self.gn2      = nn.GroupNorm(groups, out_ch)
        self.drop     = nn.Dropout(0.1)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.gelu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        x = self.drop(x)
        return F.gelu(x + residual)


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_mode="IQ"):
        super().__init__()
        self.input_mode = input_mode
        self.blocks = nn.Sequential(
            ResBlock(2,   64),  nn.MaxPool1d(2),
            ResBlock(64,  128), nn.MaxPool1d(2),
            ResBlock(128, 256), nn.MaxPool1d(2),
            ResBlock(256, 512), nn.MaxPool1d(2),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, L, 2) IQ cru
        if self.input_mode == "AP":
            x = iq_to_amp_phase(x)                 # converte para amplitude/fase
        elif self.input_mode == "FFT":
            x = iq_to_fft(x)                        # converte para dominio da frequencia
        mean = x.mean(dim=(1, 2), keepdim=True)
        std  = x.std(dim=(1, 2),  keepdim=True) + 1e-8
        x    = (x - mean) / std
        x = x.transpose(1, 2)                      # (B, 2, L)
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


# ─── Treino ───────────────────────────────────────────────────────────────────
def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def run_epoch(model, loader, optimizer, criterion, device, train):
    model.train(train)
    total_loss, total_acc, n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, z in loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            bs          = x.size(0)
            total_loss += loss.item() * bs
            total_acc  += accuracy(logits, y) * bs
            n          += bs
    return total_loss / n, total_acc / n


def train_model(input_mode, train_loader, eval_loader):
    print(f"\n========== TREINANDO MODELO [{input_mode}] ==========")
    model = CNNClassifier(input_mode=input_mode).to(DEVICE)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros treinaveis: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_eval_acc     = 0.0
    best_state        = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history           = {"train_acc": [], "eval_acc": [],
                         "train_loss": [], "eval_loss": []}
    start_epoch       = 1
    ckpt_path         = os.path.join(CKPT_DIR, f"cnn_v4_ckpt_{input_mode}.pt")
    model_path        = os.path.join(CKPT_DIR, f"cnn_v4_radioml_{input_mode}.pt")

    # RETOMADA: se ha checkpoint (preemptado antes), continua de onde parou
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optim"])
        scheduler.load_state_dict(ck["sched"])
        best_eval_acc = ck["best_acc"]; best_state = ck["best_state"]
        epochs_no_improve = ck["no_improve"]; history = ck["history"]
        start_epoch = ck["epoch"] + 1
        print(f"[{input_mode}] RETOMANDO do checkpoint -> epoca {start_epoch}, "
              f"melhor eval {best_eval_acc*100:.2f}%")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_loader.set_epoch(epoch)
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, DEVICE, train=True)
        ev_loss, ev_acc = run_epoch(model, eval_loader,  optimizer, criterion, DEVICE, train=False)
        scheduler.step()
        history["train_acc"].append(tr_acc)
        history["eval_acc"].append(ev_acc)
        history["train_loss"].append(tr_loss)
        history["eval_loss"].append(ev_loss)
        elapsed = time.time() - t0

        improved = ev_acc > best_eval_acc
        if improved:
            best_eval_acc     = ev_acc
            best_state        = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        flag = " <-best" if improved else ""
        print(f"[{input_mode}] Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"Train acc {tr_acc*100:.2f}% | Eval acc {ev_acc*100:.2f}%  "
              f"[{elapsed:.0f}s]{flag}")

        # checkpoint (retomada) + modelo do melhor eval (p/ teste cruzado)
        torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(),
                    "sched": scheduler.state_dict(), "best_acc": best_eval_acc,
                    "best_state": best_state, "no_improve": epochs_no_improve,
                    "epoch": epoch, "history": history}, ckpt_path)
        torch.save(best_state, model_path)   # CNN treinado no RadioML (melhor mode)

        # EARLY STOPPING
        if epochs_no_improve >= PATIENCE:
            print(f"[{input_mode}] Early stopping na epoca {epoch} "
                  f"(sem melhora por {PATIENCE} epocas). Melhor eval: {best_eval_acc*100:.2f}%")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, model_path)       # garante modelo salvo ao terminar o mode
    print(f"[{input_mode}] modelo salvo: {os.path.basename(model_path)}")
    return model, history, best_eval_acc


# ─── Avaliacoes ───────────────────────────────────────────────────────────────
def eval_overall(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, z in loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(dim=-1).cpu()
            correct += int((preds == y).sum())
            total   += len(y)
    return correct / total


def eval_per_snr(model, loader):
    """retorna (lista_snrs, lista_acuracias) ordenado por SNR"""
    model.eval()
    stats = {}
    with torch.no_grad():
        for x, y, z in loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(dim=-1).cpu()
            correct = (preds == y)
            for i in range(len(y)):
                snr = int(round(float(z[i])))
                if snr not in stats:
                    stats[snr] = [0, 0]
                stats[snr][0] += int(correct[i])
                stats[snr][1] += 1
    snrs = sorted(stats.keys())
    accs = [100.0 * stats[s][0] / stats[s][1] for s in snrs]
    return snrs, accs


def confusion_matrix(model, loader):
    n = NUM_CLASSES
    mat = np.zeros((n, n), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for x, y, z in loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(dim=-1).cpu().numpy()
            for t, p in zip(y.numpy(), preds):
                mat[t][p] += 1
    return mat


# ─── Plots (imagens, nada de grafico no terminal) ─────────────────────────────
def plot_confusion_matrix(mat, classes, title, save_path):
    n = len(classes)
    norm = mat.astype(float)
    row_sums = norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = norm / row_sums * 100

    fig, ax = plt.subplots(figsize=(18, 15))
    cmap = LinearSegmentedColormap.from_list("cm", ["#ffffff", "#1a4a8a"])
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Classe Predita", fontsize=13, labelpad=10)
    ax.set_ylabel("Classe Verdadeira", fontsize=13, labelpad=10)
    ax.set_title(title, fontsize=15, pad=15)
    for i in range(n):
        for j in range(n):
            v = norm[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.5, color="white" if v > 55 else "black")
    plt.colorbar(im, ax=ax, label="% das amostras verdadeiras")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {save_path}")


def plot_snr_comparison(results, save_path):
    """results = {modo: (snrs, accs)} — curva acuracia x SNR comparando IQ vs AP"""
    fig, ax = plt.subplots(figsize=(11, 6))
    cores = {"IQ": "#1a4a8a", "AP": "#c0392b", "FFT": "#27ae60"}
    for modo, (snrs, accs) in results.items():
        ax.plot(snrs, accs, marker="o", linewidth=2,
                color=cores.get(modo, None),
                label=f"{modo} (media {np.mean(accs):.1f}%)")
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Acuracia (%)", fontsize=12)
    ax.set_title("Acuracia x SNR  —  IQ vs Amplitude/Fase", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {save_path}")


def plot_history(histories, save_path):
    """curva de treino/eval ao longo das epocas para cada modo"""
    fig, ax = plt.subplots(figsize=(11, 6))
    cores = {"IQ": "#1a4a8a", "AP": "#c0392b", "FFT": "#27ae60"}
    for modo, h in histories.items():
        ep = range(1, len(h["eval_acc"]) + 1)
        ax.plot(ep, [a*100 for a in h["train_acc"]], "--",
                color=cores.get(modo), alpha=0.5, label=f"{modo} treino")
        ax.plot(ep, [a*100 for a in h["eval_acc"]], "-", marker="o",
                color=cores.get(modo), label=f"{modo} eval")
    ax.set_xlabel("Epoca", fontsize=12)
    ax.set_ylabel("Acuracia (%)", fontsize=12)
    ax.set_title("Curva de aprendizado (treino vs eval)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {save_path}")


def plot_loss_history(histories, save_path):
    """curva epoca x LOSS (treino e eval) para cada modo IQ e AP"""
    fig, ax = plt.subplots(figsize=(11, 6))
    cores = {"IQ": "#1a4a8a", "AP": "#c0392b", "FFT": "#27ae60"}
    for modo, h in histories.items():
        ep = range(1, len(h["eval_loss"]) + 1)
        ax.plot(ep, h["train_loss"], "--",
                color=cores.get(modo), alpha=0.5, label=f"{modo} treino")
        ax.plot(ep, h["eval_loss"], "-", marker="o",
                color=cores.get(modo), label=f"{modo} eval")
    ax.set_xlabel("Epoca", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Curva de loss (treino vs eval) — IQ vs Amplitude/Fase", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {save_path}")


def confusion_matrix_per_snr(model, loader):
    """
    Retorna {snr: matriz_confusao (NxN)} — uma matriz para cada SNR.
    Faz uma unica passada no teste, separando as amostras por SNR.
    """
    n = NUM_CLASSES
    mats = {}
    model.eval()
    with torch.no_grad():
        for x, y, z in loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(dim=-1).cpu().numpy()
            y_np  = y.numpy()
            for i in range(len(y_np)):
                snr = int(round(float(z[i])))
                if snr not in mats:
                    mats[snr] = np.zeros((n, n), dtype=np.int64)
                mats[snr][y_np[i]][preds[i]] += 1
    return mats


def plot_confusion_per_snr(mats, classes, best_mode, out_dir, step=2):
    """
    Salva UMA matriz de confusao por SNR (de `step` em `step` dB) numa subpasta.
    Usa o modelo vencedor (best_mode, ex.: AP).
    """
    sub_dir = os.path.join(out_dir, f"v4_confusion_per_snr_{best_mode}")
    os.makedirs(sub_dir, exist_ok=True)
    snrs = sorted(mats.keys())
    salvos = 0
    for snr in snrs:
        if snr % step != 0:               # de `step` em `step` dB
            continue
        save_path = os.path.join(sub_dir, f"cm_{best_mode}_snr_{snr:+03d}dB.png")
        plot_confusion_matrix(
            mats[snr], classes,
            f"Matriz de Confusao — {best_mode} @ SNR = {snr:+d} dB",
            save_path)
        salvos += 1
    print(f"{salvos} matrizes de confusao por SNR salvas em: {sub_dir}")


def plot_ap_example(loader, classes, save_path):
    """pega 1 amostra e mostra I, Q, amplitude e fase lado a lado"""
    sample, label = None, None
    for x, y, z in loader:
        sample = x[0].numpy()      # (L, 2)
        label  = classes[int(y[0])]
        break
    I = sample[:, 0]
    Q = sample[:, 1]
    amp = np.sqrt(I**2 + Q**2)
    pha = np.arctan2(Q, I)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(I, color="#1a4a8a");  axes[0, 0].set_title("I (em fase)")
    axes[0, 1].plot(Q, color="#1a4a8a");  axes[0, 1].set_title("Q (quadratura)")
    axes[1, 0].plot(amp, color="#c0392b"); axes[1, 0].set_title("Amplitude = sqrt(I^2+Q^2)")
    axes[1, 1].plot(pha, color="#27ae60"); axes[1, 1].set_title("Fase = atan2(Q, I)")
    for a in axes.ravel():
        a.grid(True, alpha=0.3); a.set_xlabel("tempo (amostras)")
    fig.suptitle(f"Representacoes do sinal — exemplo: {label}", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with open(CLASSES_PATH) as f:
        classes = json.load(f)
    assert len(classes) == NUM_CLASSES

    print("\n[1] Montando splits balanceados...")
    train_idx, eval_idx, test_idx = build_balanced_datasets(DATASET_PATH, SEED)
    train_loader = RadioMLBatchLoader(DATASET_PATH, train_idx, BATCH_SIZE, shuffle=True,  seed=SEED)
    eval_loader  = RadioMLBatchLoader(DATASET_PATH, eval_idx,  BATCH_SIZE, shuffle=False, seed=SEED)
    test_loader  = RadioMLBatchLoader(DATASET_PATH, test_idx,  BATCH_SIZE, shuffle=False, seed=SEED)

    histories   = {}
    snr_results = {}
    test_accs   = {}
    models      = {}

    # treina um modelo para cada tipo de entrada (IQ e AP) e compara
    for mode in INPUT_MODES:
        model, hist, _ = train_model(mode, train_loader, eval_loader)
        models[mode]    = model
        histories[mode] = hist

        test_accs[mode]   = eval_overall(model, test_loader) * 100
        snr_results[mode] = eval_per_snr(model, test_loader)
        print(f"[{mode}] Acuracia no TESTE: {test_accs[mode]:.2f}%")

    # ── escolhe o melhor para a matriz de confusao ──
    best_mode = max(test_accs, key=test_accs.get)
    print(f"\nMelhor entrada: {best_mode}  ({test_accs[best_mode]:.2f}%)")

    # ── PLOTS (tudo em imagem) ──
    print("\n[2] Gerando graficos...")
    mat = confusion_matrix(models[best_mode], test_loader)
    plot_confusion_matrix(
        mat, classes,
        f"Matriz de Confusao — {best_mode} (teste, todos os SNR)",
        os.path.join(OUT_DIR, "v4_confusion_matrix.png"))

    plot_snr_comparison(snr_results, os.path.join(OUT_DIR, "v4_snr_comparison.png"))
    plot_history(histories,         os.path.join(OUT_DIR, "v4_learning_curves.png"))
    plot_loss_history(histories,    os.path.join(OUT_DIR, "v4_loss_curves.png"))
    plot_ap_example(test_loader, classes, os.path.join(OUT_DIR, "v4_amplitude_fase.png"))

    # ── matriz de confusao por SNR (de 2 em 2 dB) do melhor modelo ──
    mats_snr = confusion_matrix_per_snr(models[best_mode], test_loader)
    plot_confusion_per_snr(mats_snr, classes, best_mode, OUT_DIR, step=2)

    # ── resumo final ──
    print("\n" + "=" * 50)
    print("RESUMO FINAL")
    print("=" * 50)
    for mode in INPUT_MODES:
        print(f"  Entrada {mode:3s} -> acuracia teste: {test_accs[mode]:.2f}%")
    print(f"  Vencedor: {best_mode}")
    print("=" * 50)


if __name__ == "__main__":
    main()
