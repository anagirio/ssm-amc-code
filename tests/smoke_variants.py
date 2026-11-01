# teste rapido das variantes de controle: monta, roda forward+backward com dados aleatorios
import os, sys, importlib.util
import torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
for v in ["BASE", "NO_SSM", "TOKENS256", "TRANSFORMER", "TRANSFORMER_MULTIRATE", "TRANSFORMER256",
          "CONVMIX", "CONVMIX_MULTIRATE", "TRANSFORMER_RMS",
          "TRANSFORMER_MULTISCALE", "TOKENS512", "TRANSFORMER512"]:
    os.environ["MV_VARIANT"] = v; os.environ["MV_DATASET"] = "radioml"
    sp = importlib.util.spec_from_file_location(f"mv_{v}", os.path.join(SRC, "mamba_variants.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    torch.manual_seed(0)
    model = m.MambaClassifier(input_mode="IQ").train()
    x = torch.randn(4, 1024, 2); y = torch.randint(0, 24, (4,))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y); loss.backward()
    used = sum(p.numel() for p in model.parameters() if p.grad is not None)
    tokens = 1024 // 2 ** m.CNN_STAGES
    print(f"{v:22s} mixer={m.MIXER:4s} multirate={m.MULTIRATE!s:5s} tokens={tokens:3d} "
          f"batch={m.BATCH_SIZE:3d} params_usados={used/1e6:.3f}M logits={tuple(logits.shape)} "
          f"loss={loss.item():.3f}")
