"""Training curves from a run's metrics.jsonl: smoothed training loss and validation loss, best step marked."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alignment.io import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("out")
    ap.add_argument("--window", type=int, default=25)
    args = ap.parse_args()
    rows = read_jsonl(args.metrics)
    train = [r for r in rows if r["kind"] == "train"]
    val = [r for r in rows if r["kind"] == "val"]
    steps, loss = [r["step"] for r in train], [r["loss"] for r in train]
    smooth = [sum(loss[max(0, i - args.window + 1) : i + 1]) / len(loss[max(0, i - args.window + 1) : i + 1]) for i in range(len(loss))]
    best = min(val, key=lambda r: (r["val_loss"], r["step"]))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(steps, loss, color="#9ab", lw=0.6, label="train (per step)")
    ax.plot(steps, smooth, color="#246", lw=1.4, label=f"train ({args.window}-step moving average)")
    ax.plot([r["step"] for r in val], [r["val_loss"] for r in val], "o-", color="#c43", lw=1.4, ms=4, label="validation (1,000 rows)")
    ax.axvline(best["step"], color="#c43", ls=":", lw=1)
    ax.annotate(f"best {best['val_loss']:.4f} @ {best['step']}", (best["step"], best["val_loss"]), textcoords="offset points",
                xytext=(-110, 18), fontsize=8, color="#c43")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("response-token NLL (nats)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)


if __name__ == "__main__":
    main()
