"""
RadioML 2018 modulation classifier with a Mamba backbone (fixed A, plain PyTorch).

Same pipeline as the CNN baseline, but the backbone is a stack of selective SSM
(Mamba) blocks in the original formulation: A is a fixed parameter, only B, C and
delta depend on the input.

Outputs (saved as images):
  - trains and compares two inputs: IQ (raw I/Q) and AP (amplitude/phase)
  - accuracy vs SNR, and accuracy/loss vs epoch
  - confusion matrix of the best model (all SNRs, and per SNR)

Run: put this file next to GOLD_XYZ_OSC.0001_1024.hdf5 and classes-fixed.json,
install torch/h5py/numpy/matplotlib, and run it. Paths are resolved relative to
this file; edit DATASET_PATH / CLASSES_PATH if the dataset lives elsewhere.

Dataset: GOLD_XYZ_OSC.0001_1024.hdf5
  2,555,904 samples, 24 classes, 1024 time steps, 2 channels (I/Q).
Balanced split: 70% train / 20% eval / 10% test (per class x SNR).
"""

import json, math, os, time, copy
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ─── config ───────────────────────────────────────────────────────────────────
# paths resolved relative to this file so it's portable across machines
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = BASE_DIR

# dataset: 'modforge' (2048, iq/modulation/snr_db) or 'radioml' (1024, X/Y/Z)
DATASET = os.environ.get("MV_DATASET", "radioml").lower()
if DATASET == "radioml":
    DATASET_PATH = os.path.join(BASE_DIR, "GOLD_XYZ_OSC.0001_1024.hdf5")
    CLASSES_PATH = os.path.join(BASE_DIR, "classes-fixed.json")
    SIG_LEN      = 1024
else:
    DATASET_PATH = os.path.join(BASE_DIR, "modforge.h5")
    CLASSES_PATH = None
    SIG_LEN      = 2048

# variant: which change to test
VARIANT = os.environ.get("MV_VARIANT", "BASE").upper()
_VALID = ["BASE", "FUSAO4CH", "TOKENS128", "DSTATE64", "WIDER",
          "AUGMENT", "ATTNPOOL", "MULTISCALE", "ADIN_LAST", "LOSSPESADA", "COMBO", "MULTI_AUG", "FUSAO_AUG", "COMBO3", "MULTI_FUSAO", "COMBO_TOP3",
          "NO_SSM", "TOKENS32", "TOKENS256", "DELTA_ADAPT", "PHYS_DELTA", "VARLEN", "MULTIRATE",
          # controls asked by the reviewer: same frontend/protocol, only the sequence mixer changes
          "TRANSFORMER", "TRANSFORMER_MULTIRATE", "TRANSFORMER256", "CONVMIX", "CONVMIX_MULTIRATE",
          "TOKENS512", "TRANSFORMER512", "TRANSFORMER_RMS", "TRANSFORMER_MULTISCALE",
          "TRANSFORMER_PHASEAUG", "PHASEAUG", "TRANSFORMER_NOPOS"]
assert VARIANT in _VALID, f"VARIANT invalida: {VARIANT}. Use uma de {_VALID}"

_CLS2IDX = {}          # modulation string -> int index; filled in main (modforge)


def _decode_mods(arr):
    """bytes/obj -> list of modulation strings."""
    return np.array([m.decode() if isinstance(m, (bytes, bytearray)) else str(m)
                     for m in arr])


def get_modforge_classes(path):
    """read the 'modulation' column and return the sorted class list."""
    with h5py.File(path, "r") as f:
        mods = _decode_mods(f["modulation"][:])
    return sorted(np.unique(mods).tolist())
# checkpoint in a persistent dir so a preempted job can resume
CKPT_DIR     = os.environ.get("MAMBA_CKPT_DIR", OUT_DIR)

NUM_CLASSES = 24
BATCH_SIZE  = 256
NUM_EPOCHS  = int(os.environ.get("MV_EPOCHS", "12"))   # default 12
LR          = 5e-4
GRAD_CLIP   = 1.0
SEED        = int(os.environ.get('MV_SEED','42'))
SNR_MIN     = -4          # SNR filter, training only
PATIENCE    = 4           # early stopping patience
TRAIN_RATIO = 0.70
EVAL_RATIO  = 0.20
TEST_RATIO  = 0.10

# input modes via MV_MODES (default IQ,AP). we run IQ and AP as parallel jobs
INPUT_MODES = os.environ.get("MV_MODES", "IQ,AP").split(",")

# ─── Mamba backbone hyperparameters (BASE, tweaked per variant below) ──
D_MODEL    = 128     # model dimension
D_STATE    = 32      # SSM state size
D_CONV     = 4       # depthwise conv width inside the block
EXPAND     = 2       # inner-dimension multiplier (d_inner = 256)
N_LAYERS   = 4       # number of Mamba blocks
# stages so that SIG_LEN/2^stages = 64 tokens (1024->4, 2048->5)
CNN_STAGES = 4 if SIG_LEN == 1024 else 5

IN_CHANNELS = 2      # [I,Q] or [amp,phase]; FUSAO4CH uses 4
USE_AUGMENT = False  # RF augmentation during training
PHASE_AUG   = False  # random carrier phase only (used for the ModForge -> RadioML test)
POOL_MODE   = "mean" # 'mean' or 'attn'
FRONTEND    = "std"  # 'std' or 'multiscale'
A_DYNAMIC_LAST = False   # dynamic A only on the last layer
USE_FOCAL   = False  # focal loss (focus on the hard, confusable pairs)
NO_SSM      = False   # ablation: skip the SSM blocks (isolate their contribution)
NORM_MODE   = "meanstd"   # per-sample input normalization: 'meanstd' or 'rms' (power only)
# selective scan implementation: 0 = original sequential loop; >0 = parallel chunks of that
# many tokens (same recurrence, same result up to float rounding; needed for 512 tokens)
SCAN_CHUNK  = int(os.environ.get("MV_SCAN_CHUNK", "0"))
MIXER       = "ssm"   # what sits after the frontend: 'ssm' (Mamba), 'attn' (Transformer)
                      # or 'conv' (local convs with about the same parameter count as the SSM)
DELTA_ADAPT = False   # learned adaptive delta (timescale)
PHYS_DELTA  = False   # scale delta by the spectral bandwidth
VARLEN      = False   # train with random input length
MULTIRATE   = False   # train resampling at a random rate
NO_POS      = False   # attention mixer without the learned positional embedding

if   VARIANT == "FUSAO4CH":   IN_CHANNELS = 4          # [I,Q,amp,phase] together
elif VARIANT == "TOKENS128":  CNN_STAGES -= 1          # 128 tokens
elif VARIANT == "DSTATE64":   D_STATE = 64             # bigger SSM state
elif VARIANT == "WIDER":      D_MODEL, N_LAYERS = 192, 6
elif VARIANT == "AUGMENT":    USE_AUGMENT = True
elif VARIANT == "ATTNPOOL":   POOL_MODE = "attn"
elif VARIANT == "MULTISCALE": FRONTEND = "multiscale"
elif VARIANT == "ADIN_LAST":  A_DYNAMIC_LAST = True
elif VARIANT == "LOSSPESADA": USE_FOCAL = True
elif VARIANT == "COMBO":      # the 3 best together
    IN_CHANNELS, USE_AUGMENT, FRONTEND = 4, True, "multiscale"
elif VARIANT == "MULTI_AUG":  # multiscale + augment
    USE_AUGMENT, FRONTEND = True, "multiscale"
elif VARIANT == "FUSAO_AUG":  # fusao4ch + augment
    IN_CHANNELS, USE_AUGMENT = 4, True
elif VARIANT == "COMBO3":     # fusao4ch + attnpool + multiscale
    IN_CHANNELS, POOL_MODE, FRONTEND = 4, "attn", "multiscale"
elif VARIANT == "MULTI_FUSAO":  # multiscale + fusao4ch
    IN_CHANNELS, FRONTEND = 4, "multiscale"
elif VARIANT == "COMBO_TOP3":   # multiscale + attnpool + dstate64
    FRONTEND, POOL_MODE, D_STATE = "multiscale", "attn", 64
elif VARIANT == "NO_SSM":       NO_SSM = True                # CNN frontend only
elif VARIANT == "TOKENS32":     CNN_STAGES += 1             # 32 tokens
elif VARIANT == "TOKENS256":    CNN_STAGES -= 2; BATCH_SIZE = 64  # 256 tokens (smaller batch to avoid OOM)
elif VARIANT == "DELTA_ADAPT":  DELTA_ADAPT = True          # adaptive delta
elif VARIANT == "PHYS_DELTA":   PHYS_DELTA = True           # delta ~ 1/bandwidth
elif VARIANT == "VARLEN":       VARLEN = True               # random input length
elif VARIANT == "MULTIRATE":    MULTIRATE = True            # random-rate augmentation
elif VARIANT == "TRANSFORMER":  MIXER = "attn"              # SSM blocks -> self-attention
elif VARIANT == "TRANSFORMER_MULTIRATE": MIXER, MULTIRATE = "attn", True
elif VARIANT == "TRANSFORMER256":   # same token budget and batch as TOKENS256
    MIXER = "attn"; CNN_STAGES -= 2; BATCH_SIZE = 64
elif VARIANT == "TOKENS512":    # 512 tokens: the regime where the SSM should need fewer operations
    CNN_STAGES -= 3; BATCH_SIZE = int(os.environ.get("MV_BATCH512", "64"))
elif VARIANT == "TRANSFORMER512":
    MIXER = "attn"; CNN_STAGES -= 3; BATCH_SIZE = int(os.environ.get("MV_BATCH512", "64"))
elif VARIANT == "TRANSFORMER_NOPOS": MIXER, NO_POS = "attn", True   # attention without positional embedding
elif VARIANT == "TRANSFORMER_RMS": MIXER, NORM_MODE = "attn", "rms"   # keep the DC/carrier cue
elif VARIANT == "PHASEAUG":     PHASE_AUG = True                # SSM trained with random carrier phase
elif VARIANT == "TRANSFORMER_PHASEAUG": MIXER, PHASE_AUG = "attn", True
elif VARIANT == "TRANSFORMER_MULTISCALE":   # the Transformer compared with MULTISCALE on full data
    MIXER, FRONTEND = "attn", "multiscale"
elif VARIANT == "CONVMIX":      MIXER = "conv"              # size-matched control without global mixing
elif VARIANT == "CONVMIX_MULTIRATE": MIXER, MULTIRATE = "conv", True

# optional suffix (e.g. MV_SUFFIX=full) so the 100%-data model doesn't overwrite the 30% one
_SUF = os.environ.get("MV_SUFFIX", "").strip()
TAG = f"{VARIANT.lower()}_{DATASET}" + (f"_{_SUF}" if _SUF else "") + (f"_s{SEED}" if SEED != 42 else "")
print(f"[variant] {VARIANT} | dataset={DATASET} (len={SIG_LEN}) | "
      f"d_model={D_MODEL} d_state={D_STATE} layers={N_LAYERS} stages={CNN_STAGES} "
      f"in_ch={IN_CHANNELS} pool={POOL_MODE} frontend={FRONTEND} mixer={MIXER} "
      f"aug={USE_AUGMENT} Adin_last={A_DYNAMIC_LAST}")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")


# ─── Carregador de batches ────────────────────────────────────────────────────
class RadioMLBatchLoader:
    # Cache compartilhado entre TODOS os loaders (train/eval/test): o dataset
    # inteiro fica na RAM, carregado UMA vez. Elimina a leitura por batch do
    # HDF5 (que era o gargalo na GPU) — batches viram fatias de arrays na RAM.
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
        """
        Load the full X, Y, Z into RAM once (shared across loaders).
        X is stored as float16 (half the RAM: ~10.5 GB instead of 21 GB) and read
        in chunks to avoid a memory spike. It's cast to float32 per batch (the net
        normalizes anyway, so float16 storage doesn't noticeably change the result).
        """
        if path not in cls._CACHE:
            print("  [RAM] loading the whole dataset into memory (float16, once)...")
            t0 = time.time()
            with h5py.File(path, "r") as f:
                key = "X" if DATASET == "radioml" else "iq"
                Xd = f[key]                                       # (N, L, 2)
                X  = np.empty(Xd.shape, dtype=np.float16)        # float16
                CH = 200000                                       # le em blocos
                for i in range(0, Xd.shape[0], CH):
                    X[i:i + CH] = Xd[i:i + CH]                    # le f32, guarda f16
                if DATASET == "radioml":
                    Y = f["Y"][:].argmax(axis=1).astype(np.int64)
                    Z = f["Z"][:].squeeze().astype(np.float32)
                else:
                    mods = _decode_mods(f["modulation"][:])      # strings
                    Y = np.array([_CLS2IDX[m] for m in mods], dtype=np.int64)
                    Z = f["snr_db"][:].squeeze().astype(np.float32)  # continuous SNR
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
            idx = order[i:i + self.batch_size]       # fatia direto da RAM
            yield (torch.from_numpy(X[idx]).float(),  # float16 -> float32
                   torch.from_numpy(Y[idx]),
                   torch.from_numpy(Z[idx]))


# ─── balanced split (all classes and all SNRs) ──────────────────────
def build_balanced_datasets(path, seed):
    """
    Balanced split: eval and test get the same amount of each (modulation x SNR),
    so accuracy isn't biased by a class or SNR that shows up more often. Training
    keeps the SNR >= SNR_MIN filter.
    """
    print("reading Y and Z to build the balanced split...")
    with h5py.File(path, "r") as f:
        if DATASET == "radioml":
            labels = f["Y"][:].argmax(axis=1)
            snrs   = np.round(f["Z"][:].squeeze()).astype(int)
        else:
            mods   = _decode_mods(f["modulation"][:])
            labels = np.array([_CLS2IDX[m] for m in mods], dtype=np.int64)
            snrs   = np.round(f["snr_db"][:].squeeze()).astype(int)

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

    # training: drop very low SNR (signal is basically pure noise)
    train_idx = train_idx[snrs[train_idx] >= SNR_MIN]

    # optional training subsample (MV_SUBSAMPLE); test stays at 100%
    _sub = float(os.environ.get("MV_SUBSAMPLE", "1.0"))
    if _sub < 1.0:
        rng.shuffle(train_idx)
        train_idx = train_idx[: int(len(train_idx) * _sub)]
        print(f"  [subsample] training reduced to {_sub:.0%}")

    print(f"Total samples         : {len(labels):,}")
    print(f"Train (SNR >= {SNR_MIN}dB) : {len(train_idx):,}")
    print(f"Eval  (balanced)      : {len(eval_idx):,}")
    print(f"Test  (balanced)      : {len(test_idx):,}")
    return train_idx, eval_idx, test_idx


# ─── IQ -> amplitude/phase ───────────────────────────────────────
def iq_to_amp_phase(x):
    """
    x: (B, L, 2) with channels [I, Q]
    returns (B, L, 2) with channels [amplitude, phase]
        amplitude = sqrt(I^2 + Q^2)
        phase     = atan2(Q, I)
    """
    I = x[..., 0]
    Q = x[..., 1]
    amp = torch.sqrt(I**2 + Q**2 + 1e-12)
    pha = torch.atan2(Q, I)
    return torch.stack([amp, pha], dim=-1)


def _bandwidth(x):
    """spectral bandwidth per sample.
    x: (B, L, 2) [I,Q]. returns (B,) = power-weighted std of the frequency."""
    comp = torch.complex(x[..., 0], x[..., 1])                     # (B, L)
    S = torch.abs(torch.fft.fft(comp, dim=1)) ** 2                 # power spectrum
    S = S / (S.sum(1, keepdim=True) + 1e-9)
    L = x.shape[1]
    f = torch.fft.fftfreq(L, device=x.device).unsqueeze(0)        # (1, L)
    centroid = (f * S).sum(1, keepdim=True)                        # (B,1)
    spread = torch.sqrt(((f - centroid) ** 2 * S).sum(1) + 1e-9)  # (B,) = bandwidth
    return spread


def _norm_per_sample(x):
    """normalize each sample to mean 0, std 1 (removes the amplitude effect).
    NORM_MODE='rms' only divides by the RMS power and keeps the mean: in baseband the carrier
    of AM-xx-WC shows up as a (slowly rotating) DC offset, and mean removal attenuates it."""
    if NORM_MODE == "rms":
        return x / (x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt() + 1e-8)
    mean = x.mean(dim=(1, 2), keepdim=True)
    std  = x.std(dim=(1, 2),  keepdim=True) + 1e-8
    return (x - mean) / std


def rf_augment(x):
    """physical RF invariances, applied during training only.
      1) random phase rotation: multiply the complex signal by e^{j*phi};
         the receiver never knows the absolute phase, so the class doesn't change.
      2) circular time shift: the frame start is arbitrary.
    x: (B, L, 2) [I,Q]
    """
    B, L, _ = x.shape
    phi = torch.rand(B, 1, device=x.device) * (2 * math.pi)
    cos, sin = torch.cos(phi), torch.sin(phi)
    I, Q = x[..., 0], x[..., 1]
    x = torch.stack([I * cos - Q * sin, I * sin + Q * cos], dim=-1)   # rotation
    shift = int(torch.randint(0, L, (1,)).item())
    return torch.roll(x, shifts=shift, dims=1)                        # circular shift


class FocalLoss(nn.Module):
    """focal loss (Lin et al.). the (1-p)^gamma weight makes training focus on the
    hard samples — in practice the pairs the model confuses (QAM16/64/256,
    AM-DSB/SSB). no class is hardcoded."""
    def __init__(self, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        focal = -((1 - pt) ** self.gamma) * logpt
        smooth = -logp.mean(dim=-1)                     # label smoothing
        return ((1 - self.smoothing) * focal + self.smoothing * smooth).mean()


# ─── Mamba Block (puro PyTorch, sem kernels CUDA/Triton) — VERSAO ORIGINAL ────
class MambaBlock(nn.Module):
    # mamba block in plain pytorch (no fused kernels so it also runs on mps/cpu).
    # A is fixed, only B/C/delta depend on the input.

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, a_dynamic=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.a_dynamic = a_dynamic
        dt_rank      = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d  = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.x_proj  = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # keep A in log space so it stays positive; forward uses -exp(A_log)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log   = nn.Parameter(torch.log(A))
        self.D       = nn.Parameter(torch.ones(self.d_inner))
        if a_dynamic:   # let A depend on the input instead of being fixed
            self.A_net = nn.Linear(self.d_inner, d_state)
        if DELTA_ADAPT:
            # tiny net that predicts a per-sample scale for delta, so the timescale adapts to the rate
            self.rate_net = nn.Sequential(
                nn.Linear(self.d_inner, self.d_inner // 4), nn.GELU(),
                nn.Linear(self.d_inner // 4, 1))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    @staticmethod
    def _scan_chunk(h0, delta, A, B, u, C):
        # one chunk of the same recurrence, computed in parallel (Hillis-Steele scan).
        # each step t is the pair (a_t, x_t) with h_t = a_t*h_{t-1} + x_t; pairs compose as
        # (a1,x1) then (a2,x2) = (a1*a2, a2*x1 + x2), so log2(T) rounds give every h_t.
        # only products of a_t <= 1 appear (no division), so it is numerically stable.
        # h0:(B,d_in,N)  delta,u:(B,T,d_in)  A:(d_in,N) or (B,T,N)  B,C:(B,T,N)
        dl = delta.unsqueeze(-1)
        a = torch.exp(dl * (A.unsqueeze(2) if A.dim() == 3 else A))       # (B,T,d_in,N)
        x = dl * B.unsqueeze(2) * u.unsqueeze(-1)
        T, d = a.shape[1], 1
        while d < T:
            a_prev = torch.cat([torch.ones_like(a[:, :d]), a[:, :-d]], dim=1)
            x_prev = torch.cat([torch.zeros_like(x[:, :d]), x[:, :-d]], dim=1)
            x = x + a * x_prev
            a = a * a_prev
            d *= 2
        h = x + a * h0.unsqueeze(1)                    # add the state carried in from the last chunk
        return (h * C.unsqueeze(2)).sum(-1), h[:, -1]  # y:(B,T,d_in), state at the chunk end

    @staticmethod
    def _scan_dir(u, delta, A, B, C):
        # u,delta:(B,L,d_in)  A:(d_in,N)  B,C:(B,L,N)  ->  y:(B,L,d_in)
        if SCAN_CHUNK > 0:
            # same recurrence in parallel chunks; with grad, each chunk is recomputed in the
            # backward instead of storing every (B,T,d_in,N) state (what the fused Mamba kernel
            # does too), otherwise long sequences run out of memory.
            h = u.new_zeros(u.shape[0], u.shape[2], A.shape[-1])
            dyn, ys = (A.dim() == 3), []
            for s in range(0, u.shape[1], SCAN_CHUNK):
                sl = slice(s, s + SCAN_CHUNK)
                args = (h, delta[:, sl], A[:, sl] if dyn else A, B[:, sl], u[:, sl], C[:, sl])
                if torch.is_grad_enabled():
                    y_c, h = torch.utils.checkpoint.checkpoint(
                        MambaBlock._scan_chunk, *args, use_reentrant=False)
                else:
                    y_c, h = MambaBlock._scan_chunk(*args)
                ys.append(y_c)
            return torch.cat(ys, dim=1)
        # sequential scan (one direction), the original implementation. I compute y_t inside
        # the loop instead of materializing the full (B,L,d_in,N) tensor.
        B_sz, L, d_in = u.shape
        h  = u.new_zeros(B_sz, d_in, A.shape[-1])
        dyn = (A.dim() == 3)          # dynamic A is (B,L,N), fixed A is (d_in,N)
        ys = []
        for t in range(L):
            A_t   = A[:, t].unsqueeze(1) if dyn else A
            dA_t  = torch.exp(delta[:, t].unsqueeze(-1) * A_t)     # discretized A
            dBu_t = delta[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            h = dA_t * h + dBu_t                                   # state update: h_t = A_bar*h + B_bar*x
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))         # output: y_t = C*h_t
        return torch.stack(ys, dim=1)

    def _ssm_scan(self, u, delta, A, B, C, D):
        # run the scan forward and backward and add them. we classify the whole signal,
        # so both directions help (a vanilla SSM only sees the past).
        A_bwd = A.flip(1) if A.dim() == 3 else A
        y_fwd = self._scan_dir(u, delta, A, B, C)
        y_bwd = self._scan_dir(u.flip(1), delta.flip(1), A_bwd,
                               B.flip(1), C.flip(1)).flip(1)
        y = y_fwd + y_bwd
        return y + u * D.unsqueeze(0).unsqueeze(0)   # skip connection

    def forward(self, x, dt_scale=None):
        residual = x
        x        = self.norm(x)
        xz       = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv1d(x_ssm)[..., :x_ssm.shape[-1]]
        x_ssm = F.silu(x_ssm).transpose(1, 2)

        x_dbc   = self.x_proj(x_ssm)
        dt_rank = self.dt_proj.in_features
        dt, B, C = x_dbc.split([dt_rank, self.d_state, self.d_state], dim=-1)
        dt      = F.softplus(self.dt_proj(dt))       # delta is predicted from the input
        if DELTA_ADAPT:
            scale = F.softplus(self.rate_net(x_ssm.mean(dim=1)))   # per-sample scale (rate proxy)
            dt = dt * scale.unsqueeze(1)
        if dt_scale is not None:
            dt = dt * dt_scale.unsqueeze(1)          # optional external scale for delta

        if self.a_dynamic:
            A = -F.softplus(self.A_net(x_ssm))       # A depends on the input
        else:
            A = -torch.exp(self.A_log)               # fixed A, negative
        y = self._ssm_scan(x_ssm, dt, A, B, C, self.D)

        y = y * F.silu(z)
        y = self.out_proj(y)
        return y + residual


# ─── full model: CNN frontend + Mamba blocks ─────────────────────────────
def _make_cnn_frontend(in_ch, out_ch, n_stages):
    """
    Reduce (B, in_ch, 1024) -> (B, out_ch, 1024/2^n_stages) with strided convs.

    Uses GroupNorm, not BatchNorm, on purpose: training filters SNR >= -4 but
    eval/test go down to -20 dB, so BatchNorm's running stats wouldn't match
    between train and eval and eval would collapse. GroupNorm normalizes within
    each sample, so there's no such mismatch.
    """
    layers = []
    ch = in_ch
    for i in range(n_stages):
        out = out_ch if i == n_stages - 1 else max(8, in_ch * (2 ** (i + 1)))
        groups = 8 if out % 8 == 0 else 1
        layers += [
            nn.Conv1d(ch, out, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(groups, out),
            nn.GELU(),
        ]
        ch = out
    return nn.Sequential(*layers)


class _MultiScaleStage(nn.Module):
    """one stage with parallel convs (kernels 3/7/15) concatenated,
    to catch symbol patterns at several durations at once."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        per = out_ch // 3
        chs = [per, per, out_ch - 2 * per]           # sums exactly to out_ch
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, c, kernel_size=k, stride=2, padding=k // 2, bias=False)
            for c, k in zip(chs, (3, 7, 15))
        ])
        groups = 8 if out_ch % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, out_ch)
        self.act  = nn.GELU()

    def forward(self, x):
        L = min(c(x).shape[-1] for c in self.convs)   # align lengths
        y = torch.cat([c(x)[..., :L] for c in self.convs], dim=1)
        return self.act(self.norm(y))


def _make_multiscale_frontend(in_ch, out_ch, n_stages):
    layers, ch = [], in_ch
    for i in range(n_stages):
        out = out_ch if i == n_stages - 1 else max(24, in_ch * (2 ** (i + 1)))
        layers.append(_MultiScaleStage(ch, out))
        ch = out
    return nn.Sequential(*layers)


class _AttnPool(nn.Module):
    """attention pooling: learns a weight per position instead of a plain mean,
    so the model focuses on the parts that actually decide the modulation."""
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):                 # x: (B, L, d_model)
        w = torch.softmax(self.score(x), dim=1)     # (B, L, 1)
        return (x * w).sum(dim=1)                   # (B, d_model)


class _ConvMixBlock(nn.Module):
    """residual conv block over the tokens: same width as the SSM and about the same
    parameter count, but it only mixes neighbouring tokens (receptive field of 7 per block).
    replacing the SSM by this separates 'global recurrence' from 'just more parameters'."""
    def __init__(self, d_model, kernel=7):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2, bias=False)
        self.norm = nn.GroupNorm(8, d_model)
        self.act  = nn.GELU()

    def forward(self, x, dt_scale=None):             # x: (B, L, d_model)
        y = self.act(self.norm(self.conv(x.transpose(1, 2)))).transpose(1, 2)
        return x + y


class MambaClassifier(nn.Module):
    """frontend + sequence mixer + classifier. the mixer is the SSM by default;
    MIXER='attn' or 'conv' swaps only that part, for the controls."""
    def __init__(self, num_classes=NUM_CLASSES, input_mode="IQ",
                 d_model=D_MODEL, d_state=D_STATE, d_conv=D_CONV,
                 expand=EXPAND, n_layers=N_LAYERS, cnn_stages=CNN_STAGES):
        super().__init__()
        self.input_mode = input_mode
        mk = _make_multiscale_frontend if FRONTEND == "multiscale" else _make_cnn_frontend
        self.cnn    = mk(IN_CHANNELS, d_model, cnn_stages)
        self.pos    = None
        if MIXER == "attn" and not NO_POS:
            # same Transformer as transformer_vs_ssm.py: 4 heads, FFN 2*d, learned positions
            self.pos = nn.Parameter(torch.zeros(1, SIG_LEN // 2 ** cnn_stages, d_model))
            nn.init.trunc_normal_(self.pos, std=0.02)
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model, 4, dim_feedforward=2 * d_model, dropout=0.1,
                                           batch_first=True, activation="gelu")
                for _ in range(n_layers)
            ])
        elif MIXER == "conv":
            self.blocks = nn.ModuleList([_ConvMixBlock(d_model) for _ in range(n_layers)])
        else:
            # dynamic A only on the last block
            self.blocks = nn.ModuleList([
                MambaBlock(d_model, d_state, d_conv, expand,
                           a_dynamic=(A_DYNAMIC_LAST and i == n_layers - 1))
                for i in range(n_layers)
            ])
        self.norm_out   = nn.LayerNorm(d_model)
        self.pool       = _AttnPool(d_model) if POOL_MODE == "attn" else None
        self.classifier = nn.Linear(d_model, num_classes)
        if PHYS_DELTA:   # learned proportionality constant (delta = c/bandwidth)
            self.phys_c = nn.Parameter(torch.tensor(-2.0))   # init: softplus~0.13

    def forward(self, x):
        # x: (B, L, 2) raw I/Q
        dt_scale = None
        if PHYS_DELTA:
            bw = _bandwidth(x)                     # scale delta by 1/bandwidth (kept absolute)
            dt_scale = (F.softplus(self.phys_c) / (bw + 1e-6)).clamp(0.05, 20.0).unsqueeze(-1)
        if IN_CHANNELS == 4:
            ap = iq_to_amp_phase(x)                # feed [I,Q,amp,phase], each pair normalized apart
            x  = torch.cat([_norm_per_sample(x), _norm_per_sample(ap)], dim=-1)
        else:
            if self.input_mode == "AP":
                x = iq_to_amp_phase(x)
            x = _norm_per_sample(x)                # per-sample normalization

        x = x.transpose(1, 2)          # (B, C, L)
        x = self.cnn(x)                # frontend: 1024 samples -> tokens
        x = x.transpose(1, 2)          # (B, tokens, d_model)
        if self.pos is not None:
            x = x + self.pos[:, :x.shape[1], :]
        if not NO_SSM:                 # ablation: skip the SSM blocks to measure how much they help
            for block in self.blocks:
                x = block(x) if MIXER == "attn" else block(x, dt_scale=dt_scale)
        x = self.norm_out(x)
        x = self.pool(x) if self.pool is not None else x.mean(dim=1)   # pool over tokens
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
            if train and USE_AUGMENT:
                x = rf_augment(x)          # augmentation, training only
            if train and PHASE_AUG:
                # random carrier phase only (one angle per example): the single thing that
                # differs between RadioML (random phase) and ModForge (almost fixed phase)
                phi = torch.rand(x.shape[0], 1, device=x.device) * (2 * math.pi)
                c, s = torch.cos(phi), torch.sin(phi)
                x = torch.stack([x[..., 0] * c - x[..., 1] * s, x[..., 0] * s + x[..., 1] * c], dim=-1)
            if train and VARLEN:
                # each batch gets a random length, so the model learns to classify
                # without knowing the input size
                import random
                Lr = random.randint(SIG_LEN // 4, SIG_LEN)
                x = x[:, :Lr]
            if train and MULTIRATE:
                # each batch is resampled at a random rate (both slower and faster).
                # resample the whole signal to L*f and crop (f>1) or pad the tail (f<1)
                # back to L -- same transform as the robustness test, so the model learns
                # to classify without depending on the rate.
                import random
                f = random.uniform(0.5, 2.0)
                Lc = x.size(1); tgt = max(8, int(round(Lc * f)))
                # on cpu: a new length every batch makes the mps backend cache a kernel per
                # shape, and memory grows until the machine swaps (same result either way)
                xt = x.transpose(1, 2).cpu()                            # (B,C,Lc)
                xt = F.interpolate(xt, size=tgt, mode="linear", align_corners=False).to(x.device)
                if tgt >= Lc:
                    xt = xt[:, :, :Lc]                                  # up: crop window
                else:
                    xt = F.pad(xt, (0, Lc - tgt))                      # down: pad tail
                x  = xt.transpose(1, 2).contiguous()                   # (B,L,C)
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
    print(f"\n========== TREINANDO MAMBA [{input_mode}] ==========")
    model = MambaClassifier(input_mode=input_mode).to(DEVICE)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros treinaveis: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    criterion = FocalLoss(gamma=2.0) if USE_FOCAL else nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_eval_acc     = 0.0
    best_state        = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history           = {"train_acc": [], "eval_acc": [],
                         "train_loss": [], "eval_loss": []}
    start_epoch       = 1
    ckpt_path         = os.path.join(CKPT_DIR, f"mv_{TAG}_ckpt_{input_mode}.pt")
    model_path        = os.path.join(CKPT_DIR, f"mv_{TAG}_{input_mode}.pt")

    # RETOMADA: se ha checkpoint (besteffort foi preemptado antes), continua
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optim"])
        scheduler.load_state_dict(ck["sched"])
        best_eval_acc = ck["best_acc"]; best_state = ck["best_state"]
        epochs_no_improve = ck["no_improve"]; history = ck["history"]
        start_epoch = ck["epoch"] + 1
        print(f"[{input_mode}] resuming from checkpoint -> epoch {start_epoch}, "
              f"best eval so far {best_eval_acc*100:.2f}%")

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
              f"Train acc {tr_acc*100:.2f}% loss {tr_loss:.3f} | "
              f"Eval acc {ev_acc*100:.2f}% loss {ev_loss:.3f}  "
              f"[{elapsed:.0f}s]{flag}")

        # save checkpoint (for resume) + best-eval model (for cross-dataset tests)
        torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(),
                    "sched": scheduler.state_dict(), "best_acc": best_eval_acc,
                    "best_state": best_state, "no_improve": epochs_no_improve,
                    "epoch": epoch, "history": history}, ckpt_path)
        torch.save(best_state, model_path)   # best model so far

        # early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"[{input_mode}] early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs). Best eval: {best_eval_acc*100:.2f}%")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, model_path)       # make sure the model is saved at the end
    print(f"[{input_mode}] model saved: {os.path.basename(model_path)}")
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
    """returns (snr_list, accuracy_list) sorted by SNR"""
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


def confusion_matrix_per_snr(model, loader):
    """returns {snr: confusion_matrix (NxN)} — one matrix per SNR."""
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


# ─── plots (saved as images) ─────────────────────────────
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
    plt.colorbar(im, ax=ax, label="% of true samples")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {save_path}")


def plot_confusion_per_snr(mats, classes, best_mode, out_dir, step=2):
    """saves one confusion matrix per SNR (every `step` dB) into a subfolder."""
    sub_dir = os.path.join(out_dir, f"mv_{TAG}_confusion_per_snr_{best_mode}")
    os.makedirs(sub_dir, exist_ok=True)
    snrs = sorted(mats.keys())
    n_saved = 0
    for snr in snrs:
        if snr % step != 0:
            continue
        save_path = os.path.join(sub_dir, f"cm_{best_mode}_snr_{snr:+03d}dB.png")
        plot_confusion_matrix(
            mats[snr], classes,
            f"Confusion matrix — {best_mode} @ SNR = {snr:+d} dB",
            save_path)
        n_saved += 1
    print(f"{n_saved} per-SNR confusion matrices saved in: {sub_dir}")


def plot_snr_comparison(results, save_path):
    """results = {mode: (snrs, accs)} — accuracy vs SNR curve, IQ vs AP"""
    fig, ax = plt.subplots(figsize=(11, 6))
    cores = {"IQ": "#1a4a8a", "AP": "#c0392b"}
    for modo, (snrs, accs) in results.items():
        ax.plot(snrs, accs, marker="o", linewidth=2,
                color=cores.get(modo, None),
                label=f"{modo} (media {np.mean(accs):.1f}%)")
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Acuracia (%)", fontsize=12)
    ax.set_title("Acuracia x SNR  —  IQ vs Amplitude/Fase (Mamba)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {save_path}")


def plot_history(histories, save_path):
    """accuracy vs epoch (train/eval) for each mode"""
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"IQ": "#1a4a8a", "AP": "#c0392b"}
    for mode, h in histories.items():
        ep = range(1, len(h["eval_acc"]) + 1)
        ax.plot(ep, [a*100 for a in h["train_acc"]], "--",
                color=colors.get(mode), alpha=0.5, label=f"{mode} train")
        ax.plot(ep, [a*100 for a in h["eval_acc"]], "-", marker="o",
                color=colors.get(mode), label=f"{mode} eval")
    ax.set_xlabel("epoch", fontsize=12)
    ax.set_ylabel("accuracy (%)", fontsize=12)
    ax.set_title("Learning curve (train vs eval) — Mamba", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {save_path}")


def plot_loss_history(histories, save_path):
    """loss vs epoch (train and eval) for each mode, IQ and AP"""
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"IQ": "#1a4a8a", "AP": "#c0392b"}
    for mode, h in histories.items():
        ep = range(1, len(h["eval_loss"]) + 1)
        ax.plot(ep, h["train_loss"], "--",
                color=colors.get(mode), alpha=0.5, label=f"{mode} train")
        ax.plot(ep, h["eval_loss"], "-", marker="o",
                color=colors.get(mode), label=f"{mode} eval")
    ax.set_xlabel("epoch", fontsize=12)
    ax.set_ylabel("loss", fontsize=12)
    ax.set_title("Loss curve (train vs eval) — IQ vs amplitude/phase (Mamba)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {save_path}")


def plot_ap_example(loader, classes, save_path):
    """take one sample and show I, Q, amplitude and phase side by side"""
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
    axes[1, 1].plot(pha, color="#27ae60"); axes[1, 1].set_title("Phase = atan2(Q, I)")
    for a in axes.ravel():
        a.grid(True, alpha=0.3); a.set_xlabel("time (samples)")
    fig.suptitle(f"Signal representations — example: {label}", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if DATASET == "radioml":
        with open(CLASSES_PATH) as f:
            classes = json.load(f)
    else:
        classes = get_modforge_classes(DATASET_PATH)
        _CLS2IDX.clear()
        _CLS2IDX.update({m: i for i, m in enumerate(classes)})
    assert len(classes) == NUM_CLASSES, f"esperava {NUM_CLASSES}, achei {len(classes)}"
    print(f"[classes] {NUM_CLASSES} modulacoes ({DATASET})")

    print("\n[1] Montando splits balanceados...")
    train_idx, eval_idx, test_idx = build_balanced_datasets(DATASET_PATH, SEED)
    train_loader = RadioMLBatchLoader(DATASET_PATH, train_idx, BATCH_SIZE, shuffle=True,  seed=SEED)
    eval_loader  = RadioMLBatchLoader(DATASET_PATH, eval_idx,  BATCH_SIZE, shuffle=False, seed=SEED)
    test_loader  = RadioMLBatchLoader(DATASET_PATH, test_idx,  BATCH_SIZE, shuffle=False, seed=SEED)

    histories   = {}
    snr_results = {}
    test_accs   = {}
    models      = {}

    # train one model per input type (IQ and AP) and compare
    for mode in INPUT_MODES:
        model, hist, _ = train_model(mode, train_loader, eval_loader)
        models[mode]    = model
        histories[mode] = hist

        test_accs[mode]   = eval_overall(model, test_loader) * 100
        snr_results[mode] = eval_per_snr(model, test_loader)
        print(f"[{mode}] test accuracy: {test_accs[mode]:.2f}%")

    # ── pick the best one for the confusion matrix ──
    best_mode = max(test_accs, key=test_accs.get)
    print(f"\nbest input: {best_mode}  ({test_accs[best_mode]:.2f}%)")

    # ── plots ──
    print("\n[2] generating plots...")
    mat = confusion_matrix(models[best_mode], test_loader)
    plot_confusion_matrix(
        mat, classes,
        f"Confusion matrix — {best_mode} (test, all SNRs) — Mamba",
        os.path.join(OUT_DIR, f"mv_{TAG}_confusion_matrix.png"))

    plot_snr_comparison(snr_results, os.path.join(OUT_DIR, f"mv_{TAG}_snr_comparison.png"))
    plot_history(histories,          os.path.join(OUT_DIR, f"mv_{TAG}_learning_curves.png"))
    plot_loss_history(histories,     os.path.join(OUT_DIR, f"mv_{TAG}_loss_curves.png"))
    plot_ap_example(test_loader, classes, os.path.join(OUT_DIR, f"mv_{TAG}_amplitude_fase.png"))

    # ── per-SNR confusion matrix (every 2 dB) of the best model ──
    mats_snr = confusion_matrix_per_snr(models[best_mode], test_loader)
    plot_confusion_per_snr(mats_snr, classes, best_mode, OUT_DIR, step=2)

    # ── final summary ──
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    for mode in INPUT_MODES:
        print(f"  input {mode:3s} -> test accuracy: {test_accs[mode]:.2f}%")
    print(f"  winner: {best_mode}")
    print("=" * 50)


if __name__ == "__main__":
    main()
