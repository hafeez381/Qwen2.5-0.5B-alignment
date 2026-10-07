"""DPO training curves from a run's metrics.jsonl: loss, reward accuracy and margin, and policy log-probs on validation."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alignment.io import read_jsonl


def moving(xs, w):
    return [sum(xs[max(0, i - w + 1) : i + 1]) / len(xs[max(0, i - w + 1) : i + 1]) for i in range(len(xs))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("out")
    ap.add_argument("--title", default="")
    ap.add_argument("--window", type=int, default=25)
    args = ap.parse_args()
    rows = read_jsonl(args.metrics)
    tr = [r for r in rows if r["kind"] == "train"]
    va = [r for r in rows if r["kind"] == "val"]
    best = max(va, key=lambda r: (r["val_accuracy"], r["val_margin"], -r["step"]))
    ts, vs = [r["step"] for r in tr], [r["step"] for r in va]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].plot(ts, moving([r["loss"] for r in tr], args.window), color="#246", lw=1.2, label=f"train ({args.window}-step avg)")
    ax[0].plot(vs, [r["val_loss"] for r in va], "o-", color="#c43", ms=3, label="validation")
    ax[0].axhline(0.6931, color="#999", ls=":", lw=1)
    ax[0].set_title("DPO loss")
    ax[1].plot(ts, moving([r["accuracy"] for r in tr], args.window), color="#246", lw=1.2, label="train accuracy (avg)")
    ax[1].plot(vs, [r["val_accuracy"] for r in va], "o-", color="#c43", ms=3, label="val accuracy")
    ax1b = ax[1].twinx()
    ax1b.plot(vs, [r["val_margin"] for r in va], "s--", color="#683", ms=3, label="val margin")
    ax1b.set_ylabel("margin $r_w - r_l$", color="#683")
    ax[1].set_title("reward accuracy and margin")
    ax[2].plot(vs, [r["val_logp_policy_w"] - va[0]["val_logp_policy_w"] for r in va], "o-", color="#246", ms=3, label=r"$\Delta\log\pi_\theta(y_w)$")
    ax[2].plot(vs, [r["val_logp_policy_l"] - va[0]["val_logp_policy_l"] for r in va], "o-", color="#c43", ms=3, label=r"$\Delta\log\pi_\theta(y_l)$")
    ax[2].axhline(0, color="#999", ls=":", lw=1)
    ax[2].set_title("validation log-prob change vs step 0")
    for a in ax:
        a.axvline(best["step"], color="#c43", ls=":", lw=1)
        a.set_xlabel("optimizer step")
        a.grid(alpha=0.25)
        a.legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(f"{args.title}  (best step {best['step']}: val acc {best['val_accuracy']:.3f}, margin {best['val_margin']:.3f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)


if __name__ == "__main__":
    main()
