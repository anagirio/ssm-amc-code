# State-Space Models for Automatic Modulation Classification

This is the code from my research internship at INSA Lyon (Inria, CITI), where I looked at
whether a selective state-space model (Mamba) is actually useful for classifying radio
modulations, and at what cost. Everything here runs on RadioML 2018.01A.

The question was never "can I beat the state of the art". It was: what does the SSM bring,
compared with models that are identical except for the part that mixes the sequence? So most
of this repository is controls, measurements and the scripts that produce them.

## What I found

All numbers are **test** accuracy on RadioML 2018.01A, averaged over all SNRs (−20 to +30 dB),
with 30% of the training set and 12 epochs unless stated otherwise. the range is from the worst to the best seed.

| Sequence mixer (64 tokens) | Seeds | Test accuracy (range) | Params | GFLOPs |
|---|---|---|---|---|
| none (CNN frontend only) | 1 | 36.7% | 0.02 M | 0.003 |
| convolution of the same size as the SSM | 5 | 40.4% (39.7–41.1) | 0.48 M | 0.062 |
| **SSM (Mamba)** | 3 | **47.2% (46.4–48.9)** | 0.54 M | 0.099 |
| Transformer (same frontend) | 5 | 51.4% (49.8–52.8) | 0.56 M | 0.079 |

- **The SSM does contribute.** It adds about 7 points over a convolutional mixer of the same
  size, so the gain is not just "more parameters": the recurrence sees the whole window, the
  convolution only neighbouring tokens.
- **It does not beat attention.** A Transformer built on the same frontend is about 4 points more
  accurate at 64 tokens, and they are about even at 256 tokens (57.5% vs 56.6%) and with the
  full training set (59.4% vs 59.8%).
- **The O(N) advantage only shows up on long sequences.** Counted properly (see below), the SSM
  needs fewer operations than attention only beyond about 256 tokens. With a scan written in
  plain PyTorch it is also slower in wall-clock time at every length I measured.
- **Robustness to the sampling rate comes from the data, not from the SSM.** Trained at one
  rate, all three mixers collapse when the test signal is resampled. Trained on randomly
  resampled signals ("multi-rate"), all three become robust. The one place the SSM does better:
  multi-rate training costs it under one point at the native rate (47.2% → 46.4%), while it
  costs the Transformer 6.2 points and the convolution 4.0. Trained at a single rate, the
  Transformer is also the most fragile: a 1% rate change takes it from 52.1% to 33.4%
  (SSM 47.4% → 45.8%). Curves use a polyphase resampler with anti-aliasing; linear
  interpolation gives the same numbers within a point.
- **Learning the timescale Δ does not give robustness on its own**, most likely because the
  convolutional frontend sits in front of the SSM and is not rate-aware.
- **Two things that are useful beyond this project:**
  - Module-level profilers such as `thop` do not see the selective scan (and report MACs, not
    FLOPs), so they underestimate the cost of a Mamba model by about 3×.
  - Removing the per-example mean of the I/Q signal erases the carrier of AM-with-carrier
    signals in baseband. Normalizing by power only raises AM-SSB-WC from 77% to 88% at high SNR.

I'm careful to call the rate result *robustness*, not *invariance*: it holds over the rates I
tested, which is not a proof.

## What's in here

### The model and training — `src/mamba_variants.py`
The backbone everything else builds on: the balanced train/eval/test split, data loading, the
CNN frontend, the Mamba block (the selective scan `h_t = A_bar*h_{t-1} + B_bar*x_t`), the
classifier and the training loop. Every configuration is a switch chosen with the
`MV_VARIANT` environment variable:

| Variant | What it is |
|---|---|
| `BASE` | CNN frontend (4 strided convs) + 4 bidirectional Mamba blocks, 64 tokens |
| `MULTISCALE` | the best model: frontend with parallel kernels 3/7/15 |
| `NO_SSM` | ablation: frontend only, no sequence mixer |
| `CONVMIX` | control: residual convolutions over the tokens, same size as the SSM |
| `TRANSFORMER` | control: Transformer encoder (4 heads, FFN 2d, learned positions) |
| `TRANSFORMER_MULTISCALE` | the Transformer with the multi-scale frontend (full-data comparison) |
| `TOKENS32`, `TOKENS256`, `TOKENS512` | SSM with 32 / 256 / 512 tokens |
| `TRANSFORMER256`, `TRANSFORMER512` | Transformer with 256 / 512 tokens |
| `MULTIRATE`, `TRANSFORMER_MULTIRATE`, `CONVMIX_MULTIRATE` | trained on signals resampled by a random factor in [0.5, 2] |
| `DELTA_ADAPT`, `PHYS_DELTA` | the Δ experiments (learned per-example scale; scale by bandwidth) |
| `VARLEN` | trained with a random input length |
| `TRANSFORMER_RMS` | Transformer with power-only input normalization (AM carrier test) |

Other switches, also environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `MV_DATASET` | dataset loader (`radioml`) | `radioml` |
| `MV_MODES` | input representation(s): `IQ`, `AP` (amplitude/phase) | `IQ,AP` |
| `MV_SUBSAMPLE` | fraction of the training set (the paper uses `0.30`) | `1.0` |
| `MV_EPOCHS` | number of epochs | `12` |
| `MV_SEED` | random seed (42–44, and 45–46 for the controls, in the paper) | `42` |
| `MV_SCAN_CHUNK` | `0` = original sequential scan; `>0` = parallel scan in chunks of that many tokens | `0` |
| `MAMBA_CKPT_DIR` | where checkpoints are written and read | next to the script |

**About the two scan implementations.** The original scan is a sequential loop over tokens. It
is fine up to 256 tokens, but for backpropagation it keeps every intermediate state, which runs
out of memory at 512 tokens. `MV_SCAN_CHUNK=64` computes the same recurrence in parallel chunks
(a Hillis–Steele scan) and recomputes each chunk in the backward pass instead of storing it,
which is what the fused Mamba kernel does too. `tests/test_scan_equiv.py` checks that both give
the same outputs and gradients (differences around 1e-11) and the same predictions on a trained
model.

### Evaluation and analysis — `src/`
| Script | What it does |
|---|---|
| `eval_test.py` | test accuracy of any set of checkpoints, overall and at SNR ≥ 10 dB, plus the high-SNR confusion matrix; writes `test_accuracy.json`. Each model is tested on the split of its own seed (the split depends on `MV_SEED`) |
| `invariance_radioml.py` | robustness to the sampling rate: resamples the test set by factors 0.6–1.5 and evaluates. `INV_SET=arch` compares SSM, Transformer and convolution, with and without multi-rate training |
| `cost_ssm_vs_transformer.py` | FLOPs counted operator by operator (including the scan) and CPU/GPU latency, SSM vs Transformer, 32–512 tokens; writes `cost_ssm_vs_transformer.json` |
| `cnn_baseline.py` | the stand-alone residual CNN baseline (60.2% IQ, 61.5% amplitude/phase on the full training set) |
| `varlen_length.py` | accuracy as a function of the input length |
| `complexity_ssm.py` | older parameter/FLOP count with `thop`; kept for reference, it **undercounts** the scan |

### Figures — `src/plot_*.py`
All figures share one style (`src/figstyle.py`) and are drawn at their printed width.
| Script | Figure |
|---|---|
| `plot_block_diagram.py` | block diagram of the model |
| `plot_controles.py` | the mixer controls, mean and individual seeds |
| `plot_robustez_split.py` | robustness to the sampling rate, fixed-rate and multi-rate models |
| `plot_cost.py` | measured FLOPs, latency and accuracy vs cost |
| `plot_varlen.py` | accuracy vs input length, fixed-length vs variable-length training |
| `plot_dataset_figs.py` | accuracy vs SNR and the high-SNR confusion matrix |

The rendered figures are in `figures/`.

### Results — `src/*.json`
`test_accuracy.json` (test accuracy of every model, per SNR and confusion matrices),
`invariance_radioml_arch_polyphase.json` (robustness curves per seed, polyphase resampler) and
`cost_ssm_vs_transformer.json` (FLOPs and latency). The figures can be redrawn from these
files without the dataset.

### Running the experiments
- `tests/smoke_variants.py` — builds every variant and runs one training step on random data.
- `tests/test_scan_equiv.py` — the scan equivalence check described above.

## Running it

You need RadioML 2018.01A (`GOLD_XYZ_OSC.0001_1024.hdf5`, ~20 GB) from DeepSig, and its
corrected class list `classes-fixed.json` (included in `src/`; the class order shipped with
the dataset is known to be wrong). Put the HDF5 file next to `mamba_variants.py`.

```
pip install torch h5py numpy scipy matplotlib thop
```

Train the base model and the ablation (light protocol used in the paper):
```
cd src
MV_DATASET=radioml MV_MODES=IQ MV_SUBSAMPLE=0.30 python mamba_variants.py
MV_DATASET=radioml MV_MODES=IQ MV_SUBSAMPLE=0.30 MV_VARIANT=NO_SSM python mamba_variants.py
```

Any other configuration is the same command with another `MV_VARIANT` (and `MV_SEED=43` or
`44` for the other seeds). For 512 tokens add `MV_SCAN_CHUNK=64`.

Then the evaluations:
```
python eval_test.py BASE BASE:_s43 BASE:_s44 NO_SSM CONVMIX TRANSFORMER   # test accuracy
INV_SET=arch python invariance_radioml.py                                  # rate robustness
python cost_ssm_vs_transformer.py && python plot_cost.py                   # cost (no dataset needed)
python ../tests/smoke_variants.py                                          # sanity check
```

## Notes

- **512 tokens.** I did not train models at 512 tokens, where the operation count favours the
  SSM. Even with the parallel scan, one training step takes about 17 s on my laptop GPU; it needs
  the optimized CUDA kernel of Mamba.
- **Hardware.** The original SSM runs were trained on the Grid'5000 cluster (V100 GPUs). The
  controls (convolution, Transformer, their seeds and multi-rate versions) were trained on an
  Apple GPU (MPS) with the same code, split and seeds. All test accuracies in
  `test_accuracy.json` were measured on the same machine.
- **Full-data Transformer.** The 59.4% Transformer trained on the full training set was run with
  a cluster script that I did not keep. The model definition is the same as
  `TRANSFORMER_MULTISCALE` here (same size, 0.58 M parameters), so it can be reproduced with
  `MV_VARIANT=TRANSFORMER_MULTISCALE MV_SUBSAMPLE=1.0`.
- **The data.** RadioML 2018.01A is a public benchmark and is not included because of its size.
