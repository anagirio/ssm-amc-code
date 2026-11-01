# Itens 1 e 3 do parecer de especialista:
#  (1) contar FLOPs de verdade, incluindo o scan (que o thop nao enxerga);
#  (3) FLOPs + latencia do SSM vs Transformer (mesmo frontend) de 32 a 512 tokens.
# Nao precisa do dataset: so constroi os modelos e roda entradas aleatorias.
import os, sys, json, time, copy, statistics, importlib.util
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry
from thop import profile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = HERE
OUT = os.path.join(HERE, "cost_ssm_vs_transformer.json")
torch.manual_seed(0)

os.environ["MV_VARIANT"] = "MULTISCALE"; os.environ["MV_DATASET"] = "radioml"
sp = importlib.util.spec_from_file_location("mv", os.path.join(SRC, "mamba_variants.py"))
m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)


class TransformerClassifier(nn.Module):
    """copia do transformer_vs_ssm.py: mesmo frontend, blocos SSM trocados por auto-atencao."""
    def __init__(self, tokens, cnn_stages, d_model=m.D_MODEL, n_layers=m.N_LAYERS, nhead=4):
        super().__init__()
        self.cnn = m._make_multiscale_frontend(m.IN_CHANNELS, d_model, cnn_stages)
        self.pos = nn.Parameter(torch.zeros(1, tokens, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model,
                                         dropout=0.1, batch_first=True, activation="gelu")
        # enable_nested_tensor=False: mantem o caminho de calculo padrao (contavel)
        self.encoder = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
        self.norm_out = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, m.NUM_CLASSES)

    def forward(self, x):
        x = m._norm_per_sample(x)
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.pos[:, :x.shape[1], :]
        x = self.encoder(x)
        return self.classifier(self.norm_out(x).mean(dim=1))


# ops que so movem/rearranjam memoria (custo aritmetico zero)
NO_COMPUTE = {
    "view", "_unsafe_view", "reshape", "transpose", "permute", "expand", "clone", "copy_",
    "slice", "select", "unsqueeze", "squeeze", "t", "cat", "stack", "detach", "alias",
    "split", "split_with_sizes", "as_strided", "zeros", "zeros_like", "new_zeros", "empty",
    "empty_like", "fill_", "contiguous", "index", "unbind", "_to_copy", "lift_fresh",
    "constant_pad_nd", "chunk", "narrow", "full", "ones", "ones_like", "scalar_tensor",
    "_reshape_alias", "unsafe_split", "unflatten", "flatten", "masked_fill", "_local_scalar_dense",
}


class FullFlopCounter(TorchDispatchMode):
    """matmul/conv: formula padrao do PyTorch (2 FLOPs por MAC).
    demais ops com saida tensor: 1 FLOP por elemento de saida (exp, mul, add, gelu...);
    reducoes (sum/mean/var/std/softmax): 1 FLOP por elemento de entrada."""
    def __init__(self):
        super().__init__(); self.matmul = 0; self.elementwise = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        pkt = func.overloadpacket
        name = pkt.__name__.rstrip("_") if pkt.__name__ != "copy_" else "copy_"
        if pkt in flop_registry:
            self.matmul += flop_registry[pkt](*args, **kwargs, out_val=out)
        elif pkt.__name__ in NO_COMPUTE or name in NO_COMPUTE:
            pass
        elif name in ("sum", "mean", "var", "std", "var_mean", "_softmax", "native_layer_norm",
                      "native_group_norm", "amax", "logsumexp"):
            first = args[0] if args and isinstance(args[0], torch.Tensor) else None
            self.elementwise += (first.numel() if first is not None else 0) * (
                5 if name in ("native_layer_norm", "native_group_norm", "_softmax") else 1)
        else:
            outs = out if isinstance(out, (tuple, list)) else (out,)
            self.elementwise += sum(o.numel() for o in outs if isinstance(o, torch.Tensor))
        return out


def count(model):
    x = torch.randn(1, 1024, 2)
    # o "fastpath" do Transformer usa um kernel fundido que o contador nao enxerga;
    # desligado so aqui, para contar as operacoes uma a uma
    torch.backends.mha.set_fastpath_enabled(False)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), FullFlopCounter() as fc:
        model(x)
    torch.backends.mha.set_fastpath_enabled(True)
    macs, _ = profile(model, inputs=(x,), verbose=False)
    return {"thop_GMACs": macs / 1e9, "matmul_GFLOPs": fc.matmul / 1e9,
            "elementwise_GFLOPs": fc.elementwise / 1e9,
            "total_GFLOPs": (fc.matmul + fc.elementwise) / 1e9}


def latency_ms(model, device, budget_s=20.0, max_runs=30):
    model = model.to(device).eval()
    x = torch.randn(1, 1024, 2, device=device)
    sync = (lambda: torch.mps.synchronize()) if device == "mps" else (lambda: None)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        sync()
        times, t_start = [], time.perf_counter()
        while len(times) < max_runs and (time.perf_counter() - t_start) < budget_s:
            t0 = time.perf_counter(); model(x); sync()
            times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times), len(times)


rows = []
for stages in [5, 4, 3, 2, 1]:
    tokens = 1024 // (2 ** stages)
    ssm = m.MambaClassifier(input_mode="IQ", cnn_stages=stages).eval()
    trf = TransformerClassifier(tokens=tokens, cnn_stages=stages).eval()
    for name, model in (("SSM", ssm), ("Transformer", trf)):
        params = sum(p.numel() for p in model.parameters()) / 1e6
        c = count(copy.deepcopy(model))   # thop deixa buffers float64 (MPS nao aceita)
        lat_cpu, n_cpu = latency_ms(model, "cpu")
        lat_mps, n_mps = latency_ms(model, "mps") if torch.backends.mps.is_available() else (None, 0)
        row = {"model": name, "tokens": tokens, "params_M": params, **c,
               "latency_cpu_ms": lat_cpu, "latency_mps_ms": lat_mps}
        rows.append(row)
        print(f"{name:11s} N={tokens:4d} params={params:.3f}M thop={c['thop_GMACs']:.4f}GMAC "
              f"matmul={c['matmul_GFLOPs']:.4f} elem={c['elementwise_GFLOPs']:.4f} "
              f"total={c['total_GFLOPs']:.4f}GFLOP  cpu={lat_cpu:.1f}ms({n_cpu})  "
              f"mps={lat_mps if lat_mps is None else round(lat_mps, 1)}ms({n_mps})", flush=True)

json.dump(rows, open(OUT, "w"), indent=2)
print("saved", OUT)
