# prova que o scan paralelo por blocos da o mesmo resultado que o laco sequencial original:
# (1) a funcao de scan isolada (saida e gradientes), para A fixo e A dinamico, varios L e blocos;
# (2) o modelo BASE treinado inteiro (mesmos logits) em sinais aleatorios.
import os, sys, importlib.util
import torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
CK = sys.argv[1]   # a trained BASE checkpoint, e.g. mv_base_radioml_IQ.pt
torch.manual_seed(0)


def load(chunk):
    os.environ["MV_VARIANT"] = "BASE"; os.environ["MV_DATASET"] = "radioml"
    os.environ["MV_SCAN_CHUNK"] = str(chunk)
    sp = importlib.util.spec_from_file_location(f"mv_{chunk}", os.path.join(SRC, "mamba_variants.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


loop, par = load(0), load(64)
worst = 0.0
for L in (37, 64, 200, 512):
    for dyn in (False, True):
        Bs, d_in, N = 3, 16, 8
        u = torch.randn(Bs, L, d_in, dtype=torch.float64, requires_grad=True)
        delta = (torch.rand(Bs, L, d_in, dtype=torch.float64) * 2).requires_grad_()
        A = (-torch.rand(Bs, L, N, dtype=torch.float64) * 5 if dyn
             else -torch.rand(d_in, N, dtype=torch.float64) * 30).requires_grad_()
        Bm = torch.randn(Bs, L, N, dtype=torch.float64, requires_grad=True)
        C = torch.randn(Bs, L, N, dtype=torch.float64, requires_grad=True)
        outs, grads = [], []
        for m in (loop, par):
            y = m.MambaBlock._scan_dir(u, delta, A, Bm, C)
            g = torch.autograd.grad(y.pow(2).sum(), (u, delta, A, Bm, C))
            outs.append(y); grads.append(g)
        d_out = (outs[0] - outs[1]).abs().max().item()
        d_grad = max((a - b).abs().max().item() for a, b in zip(*grads))
        worst = max(worst, d_out, d_grad)
        print(f"L={L:3d} A_dinamico={dyn!s:5s}  dif. saida={d_out:.1e}  dif. gradiente={d_grad:.1e}")

# modelo treinado inteiro, em float32 como no treino
x = torch.randn(8, 1024, 2)
logits = []
for m in (loop, par):
    model = m.MambaClassifier(input_mode="IQ").eval()
    model.load_state_dict(torch.load(CK, map_location="cpu", weights_only=True))
    with torch.no_grad():
        logits.append(model(x))
d = (logits[0] - logits[1]).abs().max().item()
same = (logits[0].argmax(-1) == logits[1].argmax(-1)).all().item()
print(f"modelo BASE treinado: dif. max. nos logits={d:.1e}  mesmas classes previstas={same}")
print("EQUIVALENTE" if worst < 1e-9 and d < 1e-3 and same else "DIFERENTE")
