# CIFAR-10 Activation Function Comparison (ReLU vs. GELU vs. SiLU)

A non-noisy control replication of the ConvNet from
[Vivswan/GeLUReLUInterpolation](https://github.com/Vivswan/GeLUReLUInterpolation)
("Leveraging Continuously Differentiable Activation for Learning in Analog
and Quantized Noisy Environments"), built as a standalone PyTorch project.

## What this replicates vs. what it deliberately omits

| | Original repo | This project |
|---|---|---|
| Architecture | 6 conv + 3 linear ConvNet | **Same** (identical channel counts, kernel sizes, pooling, dropout placement) |
| Dataset | CIFAR-10 | **Same** |
| Framework | PyTorch (+ AnalogVNN) | PyTorch only |
| Activations | ReLU/GELU/SiLU via a custom interpolation module (for sweeping analog-noise robustness) | Plain `nn.ReLU` / `nn.GELU` / `nn.SiLU`, evaluated once each (equivalent to their interpolation at factor 0.0 / 1.0) |
| Noise/quantization/clamp layers | Inserted around every conv/linear layer to simulate analog photonic hardware | **Omitted** — this is the "non-noisy control case" |
| Weight noise (`WeightModel`/`PseudoParameter`) | Applied to all parameters | **Omitted** |

This gives you a clean, standard-conditions baseline: same architecture,
same dataset, same framework, testing whether GELU/SiLU still edge out ReLU
under ordinary (noise-free) training — which is exactly the comparison a
course paper baseline needs. The paper can honestly state:

> "We replicated the architecture and activation functions under standard
> conditions; testing under the full analog-noise pipeline is left as
> future work."

## Files

- `model.py` — the `ConvNet` class (6 conv + 3 linear, matches the original's layer shapes and init scheme)
- `activations.py` — selects `nn.ReLU` / `nn.GELU` / `nn.SiLU`
- `data.py` — CIFAR-10 loading (plain `ToTensor()`, no normalization/augmentation, matching the original's preprocessing)
- `train.py` — trains one activation, saves per-epoch metrics to `results/<activation>_seed<seed>.json`
- `compare.py` — aggregates saved results into a summary table + comparison plot

## Setup

```bash
pip install -r requirements.txt
```

## Running the comparison

Run each activation (repeat with different `--seed` values for the
mean ± std numbers your paper should report — see note below):

```bash
python train.py --activation relu --epochs 30
python train.py --activation gelu --epochs 30
python train.py --activation silu --epochs 30
```

Then generate the summary table and plot:

```bash
python compare.py
```

This prints a table like:

```
Activation Runs  Best test acc (mean ± std)     Final test acc (mean ± std)
-------------------------------------------------------------------------------------
relu       1     84.10 ± 0.00%                  83.42 ± 0.00%
gelu       1     85.05 ± 0.00%                  84.61 ± 0.00%
silu       1     84.98 ± 0.00%                  84.30 ± 0.00%
```

and saves `results/comparison_plot.png` with test-accuracy and test-loss
curves for all three activations overlaid.

## Notes for the paper

- **Multiple seeds matter more than you'd think.** On CIFAR-10 at this
  model size, run-to-run variance from a different seed alone is often
  comparable to the gap between activations. Run each activation with 2–3
  seeds (`--seed 0`, `--seed 1`, `--seed 2`) and report mean ± std rather
  than a single number — `compare.py` already does this averaging if
  multiple result files per activation exist in `results/`.
- **Epochs/batch size:** the original repo defaults to `epochs=200,
  batch_size=1024`, tuned for their compute setup. `train.py` defaults to
  `epochs=30, batch_size=128`, which is more realistic for a course project
  on a single GPU/Colab session, and includes early stopping (`--patience`)
  so it won't run the full 30 epochs if test accuracy plateaus. State
  whichever values you actually use in your methodology section — this is
  a deviation from the original worth being explicit about.
- **No normalization/augmentation** is applied, matching the original's
  plain-`ToTensor()` preprocessing exactly. `data.py` has a commented-out
  augmented-and-normalized transform if you want a secondary "with standard
  CIFAR-10 preprocessing" ablation — just note it's not part of the direct
  replication if you use it.
