"""Direct Preference Optimization from the SFT checkpoint: trainable policy, frozen reference copy, response-only
summed log-probs, per-pair validation diagnostics, best checkpoint by validation reward accuracy."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.batching import build_lm_batch
from alignment.format import render_prompt
from alignment.guards import GuardError, assert_same_init
from alignment.io import read_jsonl, repo_path, sha256_file, write_jsonl
from alignment.logprobs import response_logprobs
from alignment.losses import dpo_loss_from_logps, dpo_metrics
from alignment.results import finalize, reconcile, start_run
from alignment.runmode import add_mode_arg, resolve_model
from alignment.train_utils import load_snapshot, save_model, save_snapshot, set_seed

LN2 = math.log(2)


def load_config(path: str, mode: str) -> dict:
    cfg = yaml.safe_load(repo_path(path).read_text())
    if mode == "smoke":
        cfg.update(init=resolve_model("Qwen/Qwen2.5-0.5B", "smoke"), init_sha256=None,
                   train=cfg["train"].replace("data/prepared", "results/_smoke/data/prepared"),
                   val=cfg["val"].replace("data/prepared", "results/_smoke/data/prepared"),
                   output=cfg["output"].replace("outputs/", "outputs/_smoke/"),
                   results=cfg["results"].replace("results/", "results/_smoke/"),
                   grad_accum=2, eval_every=1, snapshot_every=1, max_train_rows=4, max_val_rows=8, early_gate_step=None)
    return cfg


def pair_batch(tok, rows: list[dict], max_length: int) -> dict:
    """Chosen rows first, then rejected rows, so log-probs split as [:k] and [k:]."""
    prompts = [render_prompt(tok, r["prompt"]) for r in rows]
    return build_lm_batch(tok, prompts + prompts, [r["chosen"] for r in rows] + [r["rejected"] for r in rows],
                          max_length, pad_to="longest")


def param_checksum(model) -> list[float]:
    with torch.no_grad():
        return [float(p.double().sum()) + float(p.double().pow(2).sum()) for p in model.parameters()]


def assert_reference_frozen(ref, opt, checksum: list[float] | None = None) -> None:
    if any(p.requires_grad for p in ref.parameters()):
        raise GuardError("reference parameters must not require grad")
    trained = {id(p) for g in opt.param_groups for p in g["params"]}
    if any(id(p) in trained for p in ref.parameters()):
        raise GuardError("reference parameters are in the optimizer")
    if checksum is not None and param_checksum(ref) != checksum:
        raise GuardError("reference weights changed during training")


@torch.no_grad()
def validate(policy, ref, tok, rows, cfg, autocast, ref_cache: dict) -> list[dict]:
    policy.eval()
    out = []
    for i in range(0, len(rows), cfg["eval_batch"]):
        part = rows[i : i + cfg["eval_batch"]]
        k = len(part)
        b = pair_batch(tok, part, cfg["max_length"])
        with autocast:
            lp, n = response_logprobs(policy, b)
            if not all(r["id"] in ref_cache for r in part):
                rlp, _ = response_logprobs(ref, b)
                for j, r in enumerate(part):
                    ref_cache[r["id"]] = (float(rlp[j]), float(rlp[k + j]))
        rc = torch.tensor([ref_cache[r["id"]][0] for r in part])
        rr = torch.tensor([ref_cache[r["id"]][1] for r in part])
        pc, pr = lp[:k].cpu(), lp[k:].cpu()
        m = dpo_metrics(pc, pr, rc, rr, cfg["beta"])
        losses = -F.logsigmoid(m["margin"])
        for j, r in enumerate(part):
            out.append({"id": r["id"], "logp_policy_w": float(pc[j]), "logp_policy_l": float(pr[j]),
                        "logp_ref_w": float(rc[j]), "logp_ref_l": float(rr[j]), "r_w": float(m["reward_chosen"][j]),
                        "r_l": float(m["reward_rejected"][j]), "margin": float(m["margin"][j]),
                        "correct": bool(m["correct"][j]), "loss": float(losses[j]),
                        "len_w": int(n[j]), "len_l": int(n[k + j])})
    policy.train()
    return out


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    return {"val_loss": mean("loss"), "val_accuracy": sum(r["correct"] for r in rows) / n, "val_margin": mean("margin"),
            "val_r_w": mean("r_w"), "val_r_l": mean("r_l"), "val_logp_policy_w": mean("logp_policy_w"),
            "val_logp_policy_l": mean("logp_policy_l"), "n_pairs": n}


def summarize(rows: list[dict]) -> dict:
    train = [r for r in rows if r["kind"] == "train"]
    val = [r for r in rows if r["kind"] == "val"]
    best = max(val, key=lambda r: (r["val_accuracy"], r["val_margin"], -r["step"]))
    first, last = val[0], val[-1]
    return {"n_steps": len(train), "skipped_steps": sum(r["skipped"] for r in train),
            "step0": {k: first[k] for k in ("val_loss", "val_accuracy", "val_margin")},
            "best_step": best["step"], "best_val_accuracy": best["val_accuracy"], "best_val_margin": best["val_margin"],
            "best_val_loss": best["val_loss"], "final_val_accuracy": last["val_accuracy"],
            "final_val_margin": last["val_margin"], "final_val_loss": last["val_loss"],
            "delta_logp_policy_w_at_best": best["val_logp_policy_w"] - first["val_logp_policy_w"],
            "delta_logp_policy_l_at_best": best["val_logp_policy_l"] - first["val_logp_policy_l"]}


def train(config: str, mode: str, resume: bool, ignore_early_gate: bool) -> int:
    cfg = load_config(config, mode)
    set_seed(cfg["seed"])
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=cuda)
    out, snap = repo_path(cfg["output"]), repo_path(cfg["output"]) / "snapshot"
    if cfg["init_sha256"]:
        assert_same_init(repo_path(cfg["init"]), cfg["init_sha256"])
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    policy = AutoModelForCausalLM.from_pretrained(snap if resume else cfg["init"], torch_dtype=torch.float32).to(device)
    ref = AutoModelForCausalLM.from_pretrained(cfg["init"], torch_dtype=torch.float32).to(device).eval()
    ref.requires_grad_(False)
    policy.train()
    if {p.dtype for p in policy.parameters()} != {torch.float32}:
        raise RuntimeError("trainable weights must be fp32 when using gradient scaling")

    train_rows = read_jsonl(cfg["train"])[: cfg.get("max_train_rows")]
    val_rows = read_jsonl(cfg["val"])[: cfg.get("max_val_rows")]
    loader = DataLoader(train_rows, batch_size=1, shuffle=True, collate_fn=lambda rows: rows,
                        generator=torch.Generator().manual_seed(cfg["seed"]))
    accum = cfg["grad_accum"]
    n_steps = math.ceil(len(loader) / accum) * cfg["epochs"]
    opt = torch.optim.RMSprop(policy.parameters(), lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    assert_reference_frozen(ref, opt)
    ref_checksum = param_checksum(ref)
    state = {"step": 0, "micro": 0, "best": [-1.0, -float("inf")], "train_seconds": 0.0, "early_gate_done": False}

    run = start_run(cfg["results"], mode=mode, cfg=cfg, seed=cfg["seed"], resume=resume)
    log = run["outdir"] / "metrics.jsonl"
    if resume:
        state = load_snapshot({"optimizer": opt, "scaler": scaler}, snap)
        kept = [r for r in read_jsonl(log) if r["step"] <= state["step"]]
        log.write_text("".join(json.dumps(r) + "\n" for r in kept))
    else:
        out.mkdir(parents=True, exist_ok=True)

    def append(row):
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")

    ref_cache: dict = {}

    def run_validation() -> dict:
        rows = validate(policy, ref, tok, val_rows, cfg, autocast, ref_cache)
        name = f"val/step_{state['step']:04d}.jsonl"
        write_jsonl(run["outdir"] / name, rows)
        agg = aggregate(rows)
        append({"kind": "val", "step": state["step"], **agg})
        print(f"val step={state['step']} loss={agg['val_loss']:.4f} acc={agg['val_accuracy']:.4f} "
              f"margin={agg['val_margin']:.4f} logp_w={agg['val_logp_policy_w']:.2f} logp_l={agg['val_logp_policy_l']:.2f}", flush=True)
        if [agg["val_accuracy"], agg["val_margin"]] > state["best"]:
            state["best"] = [agg["val_accuracy"], agg["val_margin"]]
            save_model(policy, tok, out / "best")
        return agg

    if state["step"] == 0:
        agg = run_validation()
        ok = agg["val_accuracy"] == 0.0 and abs(agg["val_loss"] - LN2) < 1e-4 and abs(agg["val_margin"]) < 1e-6
        print(f"{'PASS' if ok else 'BREACH'}: step-0 policy equals reference: accuracy {agg['val_accuracy']} (need 0.0), "
              f"loss {agg['val_loss']:.6f} (need ln 2 = {LN2:.6f}), margin {agg['val_margin']:.2e}", flush=True)
        if not ok:
            return 1

    t0, window = time.perf_counter(), []
    for i, batch in enumerate(loader):
        if i < state["micro"]:
            continue
        window.append(batch[0])
        if len(window) < accum and i + 1 < len(loader):
            continue
        stats = {"loss": 0.0, "correct": 0, "margin": 0.0, "r_w": 0.0, "r_l": 0.0}
        for row in window:
            b = pair_batch(tok, [row], cfg["max_length"])
            with autocast:
                lp, _ = response_logprobs(policy, b)
                with torch.no_grad():
                    rlp, _ = response_logprobs(ref, b)
            loss = dpo_loss_from_logps(lp[:1], lp[1:], rlp[:1], rlp[1:], cfg["beta"])
            scaler.scale(loss / len(window)).backward()
            m = dpo_metrics(lp[:1].detach(), lp[1:].detach(), rlp[:1], rlp[1:], cfg["beta"])
            stats["loss"] += loss.item()
            stats["correct"] += int(m["correct"].sum())
            for key, src in (("margin", "margin"), ("r_w", "reward_chosen"), ("r_l", "reward_rejected")):
                stats[key] += float(m[src].sum())
        n_pairs = len(window)
        window = []
        scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["grad_clip"]))
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        state["step"] += 1
        state["micro"] = i + 1
        state["train_seconds"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        append({"kind": "train", "step": state["step"], "loss": stats["loss"] / n_pairs, "accuracy": stats["correct"] / n_pairs,
                "margin": stats["margin"] / n_pairs, "r_w": stats["r_w"] / n_pairs, "r_l": stats["r_l"] / n_pairs,
                "grad_norm": grad_norm, "scale": scale, "skipped": bool(cuda and scaler.get_scale() < scale),
                "pairs": n_pairs, "seconds": state["train_seconds"]})
        print(f"step {state['step']}/{n_steps} loss={stats['loss'] / n_pairs:.4f} acc={stats['correct'] / n_pairs:.3f} "
              f"grad_norm={grad_norm:.3f} t={state['train_seconds']:.0f}s", flush=True)
        if state["step"] % cfg["eval_every"] == 0 or state["step"] == n_steps:
            agg = run_validation()
            gate = cfg["early_gate_step"]
            if gate and state["step"] >= gate and not state["early_gate_done"] and not ignore_early_gate:
                state["early_gate_done"] = True
                stuck = agg["val_accuracy"] < cfg["early_gate_min_accuracy"] and agg["val_margin"] < cfg["early_gate_min_margin"]
                print(f"{'BREACH' if stuck else 'PASS'}: early gate at step {state['step']}: accuracy {agg['val_accuracy']:.4f} "
                      f"(min {cfg['early_gate_min_accuracy']}), margin {agg['val_margin']:.4f} (min {cfg['early_gate_min_margin']})", flush=True)
                if stuck:
                    save_snapshot(policy, tok, {"optimizer": opt, "scaler": scaler}, state, snap)
                    return 2
        if state["step"] % cfg["snapshot_every"] == 0 and state["step"] < n_steps:
            save_snapshot(policy, tok, {"optimizer": opt, "scaler": scaler}, state, snap)

    assert_reference_frozen(ref, opt, ref_checksum)
    print("PASS: reference weights unchanged and outside the optimizer", flush=True)
    save_model(policy, tok, out / "final")
    rows = read_jsonl(log)
    finalize(run, log, summarize(rows), summarize, extra={
        "n_train_pairs": len(train_rows), "n_val_pairs": len(val_rows), "n_optimizer_steps_planned": n_steps,
        "train_seconds": round(state["train_seconds"], 1), "init": cfg["init"], "init_sha256": cfg["init_sha256"],
        "best_sha256": sha256_file(out / "best" / "model.safetensors"),
        "final_sha256": sha256_file(out / "final" / "model.safetensors"),
        "device": torch.cuda.get_device_name() if cuda else "cpu"})
    expected = {"metrics.jsonl", "summary.json"} | {str(p.relative_to(run["outdir"])) for p in (run["outdir"] / "val").glob("*.jsonl")}
    reconcile(run["outdir"], expected)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    add_mode_arg(ap)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ignore-early-gate", action="store_true")
    args = ap.parse_args()
    return train(args.config, args.mode, args.resume, args.ignore_early_gate)


if __name__ == "__main__":
    sys.exit(main())
