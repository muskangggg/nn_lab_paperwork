"""
ConvNet architecture, replicated from Vivswan/GeLUReLUInterpolation's
src/run_conv.py -> ConvModel.

What's kept (matches the original exactly):
    - 6 conv layers: channels 3->48->48->96->96->192->192
    - kernel_size=3, with padding=1 on layers 1/3/5 and no padding on 2/4/6
    - MaxPool2d(2, 2) after conv layers 2, 4, 6
    - Dropout(0.25) before conv layers 3 and 5
    - 3 linear layers: flatten -> 512 -> 256 -> num_classes
    - Dropout(0.5) after linear layers 1 and 2
    - Kaiming-uniform init applied only to nn.Linear layers (as in the original)
    - Activation function placed after every conv/linear layer except the
      final classification layer

What's removed (the "non-noisy control" simplification):
    - Clamp (normalization), ReducePrecision (quantization), and
      GaussianNoise layers that the original conditionally inserts around
      every conv/linear layer for analog-hardware simulation
    - The AnalogVNN PseudoParameter weight-noise wrapper (WeightModel)
    - The ReLU<->GELU/SiLU interpolation mechanism (interpolate_factor) --
      we plug in plain nn.ReLU / nn.GELU / nn.SiLU directly instead, which is
      equivalent to evaluating their interpolation at factor 0.0 or 1.0
"""

from typing import Type

import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(
        self,
        activation_fn: Type[nn.Module],
        input_shape=(3, 32, 32),
        num_classes: int = 10,
    ):
        super().__init__()
        act = lambda: activation_fn() if isinstance(activation_fn, type) else activation_fn

        # ---- conv stack (channels: 3 -> 48 -> 48 -> 96 -> 96 -> 192 -> 192) ----
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 48, kernel_size=3, padding=1),
            act(),

            nn.Conv2d(48, 48, kernel_size=3),
            act(),
            nn.MaxPool2d(2, 2),

            nn.Dropout(0.25),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            act(),

            nn.Conv2d(96, 96, kernel_size=3),
            act(),
            nn.MaxPool2d(2, 2),

            nn.Dropout(0.25),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            act(),

            nn.Conv2d(192, 192, kernel_size=3),
            act(),
            nn.MaxPool2d(2, 2),
        )

        # figure out the flattened feature size for this input_shape
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_dim = self.conv(dummy).flatten(1).shape[1]

        self.flatten = nn.Flatten(start_dim=1)

        # ---- linear stack (flat_dim -> 512 -> 256 -> num_classes) ----
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 512),
            act(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            act(),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        # original only kaiming-inits nn.Linear layers, conv layers keep
        # PyTorch's default init
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # quick smoke test: build the model for each activation and check shapes
    from activations import ACTIVATIONS

    for name, act_cls in ACTIVATIONS.items():
        model = ConvNet(activation_fn=act_cls)
        dummy_input = torch.randn(4, 3, 32, 32)  # CIFAR-10 shaped batch
        out = model(dummy_input)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{name:5s}] output shape: {tuple(out.shape)}  |  params: {n_params:,}")
