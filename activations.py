"""
Activation function selector.

The original GeLUReLUInterpolation repo implements ReLU/GELU/SiLU as special
cases of a custom interpolation module (ReLUGeLUInterpolation /
ReLUSiLUInterpolation with interpolate_factor in {0.0, 1.0}) because it needs
to sweep the interpolation factor for its noise-robustness analysis.

Since our control case does not sweep the interpolation factor, we use the
plain, standard PyTorch activation modules directly -- this is mathematically
identical to their interpolation module evaluated at interpolate_factor=0.0
(ReLU) or interpolate_factor=1.0 (GELU / SiLU), just without the unused
machinery.
"""

import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,  # SiLU == Swish
}


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]()
