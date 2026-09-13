"""
After running train.py for relu / gelu / silu (optionally across multiple
seeds), this collects the saved results/*.json files into a summary table
and a training-curve plot.

Usage:
    python compare.py
    python compare.py --results_dir ./results --out_dir ./results
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path):
    runs = defaultdict(list)
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        runs[data["activation"]].append(data)
    return runs


def print_summary_table(runs):
    print(f"{'Activation':<10} {'Runs':<5} {'Best test acc (mean ± std)':<30} {'Final test acc (mean ± std)'}")
    print("-" * 85)
    for activation in ["relu", "gelu", "silu"]:
        if activation not in runs:
            continue
        best = np.array([r["best_test_acc"] for r in runs[activation]]) * 100
        final = np.array([r["final_test_acc"] for r in runs[activation]]) * 100
        print(f"{activation:<10} {len(runs[activation]):<5} "
              f"{best.mean():.2f} ± {best.std():.2f}%{'':<15} "
              f"{final.mean():.2f} ± {final.std():.2f}%")


def plot_curves(runs, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"relu": "tab:red", "gelu": "tab:blue", "silu": "tab:green"}

    for activation in ["relu", "gelu", "silu"]:
        if activation not in runs:
            continue
        # use the first seed's run for the curve (extend this to average
        # across seeds if you have multiple runs per activation)
        history = runs[activation][0]["history"]
        epochs = range(1, len(history["test_acc"]) + 1)
        axes[0].plot(epochs, [a * 100 for a in history["test_acc"]],
                     label=activation.upper(), color=colors[activation])
        axes[1].plot(epochs, history["test_loss"],
                     label=activation.upper(), color=colors[activation])

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("Test accuracy vs. epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Test loss")
    axes[1].set_title("Test loss vs. epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--out_dir", type=str, default="./results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = load_results(results_dir)

    if not runs:
        print(f"No result files found in {results_dir}. Run train.py first.")
        return

    print_summary_table(runs)
    plot_curves(runs, Path(args.out_dir) / "comparison_plot.png")


if __name__ == "__main__":
    main()
