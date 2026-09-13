"""
Train the control-case ConvNet (no noise/quantization/clamp) on CIFAR-10
for a single activation function.

Usage:
    python train.py --activation relu  --epochs 30
    python train.py --activation gelu  --epochs 30
    python train.py --activation silu  --epochs 30

Results (per-epoch train/test loss & accuracy) are saved to
results/<activation>_seed<seed>.json for later comparison / plotting.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from activations import get_activation
from data import get_dataloaders
from model import ConvNet


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += inputs.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", type=str, required=True, choices=["relu", "gelu", "silu"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--patience", type=int, default=10,
                         help="stop if test accuracy hasn't improved in this many epochs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Activation: {args.activation} | Device: {device} | Seed: {args.seed}")

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)

    # get_activation() returns an instance (e.g. nn.ReLU()); ConvNet accepts
    # either a class or an already-built instance -- see model.py's `act()`
    # helper. Reusing one stateless instance across layers is safe here
    # since ReLU/GELU/SiLU hold no learnable parameters or per-layer state.
    model = ConvNet(activation_fn=get_activation(args.activation)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_test_acc = 0.0
    epochs_since_improvement = 0

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"[{epoch:3d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
              f"test_loss={test_loss:.4f} test_acc={test_acc*100:.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # simple divergence guard, mirroring the spirit of the original
        # repo's early-stopping (it bails if training clearly isn't working)
        if epoch >= 5 and train_acc < 0.15:
            print("Training does not appear to be converging -- stopping early.")
            break

        if epochs_since_improvement >= args.patience:
            print(f"No test accuracy improvement in {args.patience} epochs -- stopping early.")
            break

    elapsed = time.time() - start
    print(f"Done in {elapsed/60:.1f} min. Best test accuracy: {best_test_acc*100:.2f}%")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.activation}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({
            "activation": args.activation,
            "seed": args.seed,
            "epochs_run": len(history["train_loss"]),
            "best_test_acc": best_test_acc,
            "final_test_acc": history["test_acc"][-1],
            "elapsed_minutes": elapsed / 60,
            "history": history,
            "args": vars(args),
        }, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
