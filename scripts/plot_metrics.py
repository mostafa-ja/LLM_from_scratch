"""Plot training and validation loss from ``out/metrics.json``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="out/metrics.json")
    parser.add_argument("--output", default="assets/loss_curve.png")
    parser.add_argument("--title", default="Tiny GPT-2 on Tiny Shakespeare")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    history = metrics["history"]
    steps = history["step"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.plot(steps, train_loss, label="train", linewidth=2, color="#2563eb")
    ax.plot(steps, val_loss, label="validation", linewidth=2, color="#f97316")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(args.title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
