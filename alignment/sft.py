"""Supervised fine-tuning with response-only loss, gradient accumulation, fp16 autocast with fp32 weights and
gradient scaling, clipping, cosine warmup, periodic validation, best/final checkpoints and resumable snapshots."""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from alignment.batching import IGNORE, build_lm_batch
from alignment.format import render_prompt
from alignment.io import read_jsonl, repo_path, sha256_file
from alignment.logprobs import response_logprobs
from alignment.results import finalize, reconcile, start_run
from alignment.runmode import add_mode_arg, resolve_model

SMOKE = {"train": "results/_smoke/data/prepared/sft_train.jsonl", "val": "results/_smoke/data/prepared/sft_val.jsonl",
         "output": "outputs/_smoke/sft", "results": "results/_smoke/sft", "grad_accum": 2, "eval_every": 1,
         "snapshot_every": 1, "max_train_rows": 8, "max_val_rows": 4}
VAL_RISES_MAX = 3


def load_config(path: str, mode: str) -> dict:
    cfg = yaml.safe_load(repo_path(path).read_text())
    if mode == "smoke":
        cfg.update(SMOKE)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def window_backward(model, batches, scaler, autocast) -> tuple[float, int]:
    """Backprop one accumulation window so its gradient equals one big batch's token-mean loss."""
    n_tokens = sum(int(b["labels"][:, 1:].ne(IGNORE).sum()) for b in batches)
    total = 0.0
    for b in batches:
        with autocast:
            logp, _ = response_logprobs(model, b)
        loss = -logp.sum() / max(n_tokens, 1)
        scaler.scale(loss).backward()
        total += loss.item()
    return total, n_tokens


@torch.no_grad()
def evaluate(model, tok, rows, cfg, autocast) -> tuple[float, int]:
    model.eval()
    nll, n = 0.0, 0
    for i in range(0, len(rows), cfg["eval_batch"]):
        part = rows[i : i + cfg["eval_batch"]]
        b = build_lm_batch(tok, [render_prompt(tok, r["prompt"]) for r in part], [r["response"] for r in part],
                           cfg["max_length"], pad_to="longest")
        with autocast:
            logp, cnt = response_logprobs(model, b)
        nll -= logp.sum().item()
        n += int(cnt.sum())
    model.train()
    return nll / max(n, 1), n


def save_model(model, tok, path: Path) -> None:
    tmp = path.with_name(path.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tok.save_pretrained(tmp)
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def save_snapshot(model, tok, opt, scaler, sched, state, path: Path) -> None:
    tmp = path.with_name(path.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tok.save_pretrained(tmp)
    torch.save({"optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "scheduler": sched.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, tmp / "trainer.pt")
    (tmp / "state.json").write_text(json.dumps(state))
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)
    for f in ("model.safetensors", "trainer.pt"):
        print(f"SNAPSHOT step={state['step']} {f} sha256={sha256_file(path / f)}", flush=True)


def summarize(rows: list[dict]) -> dict:
    train = [r for r in rows if r["kind"] == "train"]
    val = [r for r in rows if r["kind"] == "val"]
    best = min(val, key=lambda r: (r["val_loss"], r["step"]))
    rises = max_rises = 0
    for a, b in zip(val, val[1:]):
        rises = rises + 1 if b["val_loss"] > a["val_loss"] else 0
        max_rises = max(max_rises, rises)
    return {"n_steps": len(train), "first_train_loss": train[0]["loss"], "final_train_loss": train[-1]["loss"],
            "first_val_loss": val[0]["val_loss"], "final_val_loss": val[-1]["val_loss"], "best_step": best["step"],
            "best_val_loss": best["val_loss"], "skipped_steps": sum(r["skipped"] for r in train),
            "max_consecutive_val_rises": max_rises}


def train(config: str, mode: str, resume: bool) -> int:
    cfg = load_config(config, mode)
    set_seed(cfg["seed"])
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=cuda)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    out, snap = repo_path(cfg["output"]), repo_path(cfg["output"]) / "snapshot"
    source = snap if resume else resolve_model(cfg["model"], mode)
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32).to(device)
    model.train()
    if {p.dtype for p in model.parameters()} != {torch.float32}:
        raise RuntimeError("trainable weights must be fp32 when using gradient scaling")

    train_rows = read_jsonl(cfg["train"])[: cfg.get("max_train_rows")]
    val_rows = read_jsonl(cfg["val"])[: cfg.get("max_val_rows")]
    collate = lambda rows: build_lm_batch(tok, [render_prompt(tok, r["prompt"]) for r in rows],
                                          [r["response"] for r in rows], cfg["max_length"], pad_to="longest")
    loader = DataLoader(train_rows, batch_size=cfg["per_device_batch"], shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(cfg["seed"]))
    accum = cfg["grad_accum"]
    n_steps = math.ceil(len(loader) / accum) * cfg["epochs"]

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"])
    sched = get_cosine_schedule_with_warmup(opt, round(cfg["warmup_ratio"] * n_steps), n_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    state = {"step": 0, "micro": 0, "best_val": float("inf"), "train_seconds": 0.0}

    run = start_run(cfg["results"], mode=mode, cfg=cfg, seed=cfg["seed"], resume=resume)
    log = run["outdir"] / "metrics.jsonl"
    if resume:
        ckpt = torch.load(snap / "trainer.pt", weights_only=False)
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        sched.load_state_dict(ckpt["scheduler"])
        torch.set_rng_state(ckpt["torch_rng"])
        if cuda and ckpt["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        state = json.loads((snap / "state.json").read_text())
        kept = [r for r in read_jsonl(log) if r["step"] <= state["step"]]
        log.write_text("".join(json.dumps(r) + "\n" for r in kept))
    else:
        out.mkdir(parents=True, exist_ok=True)

    def append(row):
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def validate():
        val_loss, n = evaluate(model, tok, val_rows, cfg, autocast)
        append({"kind": "val", "step": state["step"], "val_loss": val_loss, "val_tokens": n})
        print(f"val step={state['step']} loss={val_loss:.4f}", flush=True)
        if val_loss < state["best_val"]:
            state["best_val"] = val_loss
            save_model(model, tok, out / "best")

    if state["step"] == 0:
        validate()
    t0, window = time.perf_counter(), []
    for i, b in enumerate(loader):
        if i < state["micro"]:
            continue
        window.append(b)
        if len(window) < accum and i + 1 < len(loader):
            continue
        loss, n_tokens = window_backward(model, window, scaler, autocast)
        window = []
        scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]))
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        lr = sched.get_last_lr()[0]
        sched.step()
        opt.zero_grad(set_to_none=True)
        state["step"] += 1
        state["micro"] = i + 1
        state["train_seconds"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        append({"kind": "train", "step": state["step"], "loss": loss, "lr": lr, "grad_norm": grad_norm,
                "scale": scale, "skipped": bool(cuda and scaler.get_scale() < scale), "tokens": n_tokens,
                "seconds": state["train_seconds"]})
        print(f"step {state['step']}/{n_steps} loss={loss:.4f} lr={lr:.2e} grad_norm={grad_norm:.3f} "
              f"t={state['train_seconds']:.0f}s", flush=True)
        if state["step"] % cfg["eval_every"] == 0 or state["step"] == n_steps:
            validate()
        if state["step"] % cfg["snapshot_every"] == 0 and state["step"] < n_steps:
            save_snapshot(model, tok, opt, scaler, sched, state, snap)

    save_model(model, tok, out / "final")
    rows = read_jsonl(log)
    summary = summarize(rows)
    finalize(run, log, summary, summarize, extra={
        "n_train_rows": len(train_rows), "n_val_rows": len(val_rows), "n_optimizer_steps_planned": n_steps,
        "train_seconds": round(state["train_seconds"], 1),
        "best_sha256": sha256_file(out / "best" / "model.safetensors"),
        "final_sha256": sha256_file(out / "final" / "model.safetensors"),
        "device": torch.cuda.get_device_name() if cuda else "cpu"})
    reconcile(run["outdir"], {"metrics.jsonl", "summary.json"})
    ok = summary["max_consecutive_val_rises"] < VAL_RISES_MAX
    print(f"{'PASS' if ok else 'BREACH'}: max consecutive validation-loss rises {summary['max_consecutive_val_rises']} "
          f"(limit {VAL_RISES_MAX}); best step {summary['best_step']} val {summary['best_val_loss']:.4f}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    return train(args.config, args.mode, args.resume)


if __name__ == "__main__":
    sys.exit(main())
