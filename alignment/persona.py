"""Persona style tuning with TRL's DPOTrainer and a LoRA adapter (the one task where TRL and PEFT are allowed).
The reference model is the base model with the adapter disabled (ref_model=None + peft_config)."""
from __future__ import annotations

import argparse
import random
import sys

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from alignment.io import read_json, read_jsonl, repo_path, sha256_file, write_jsonl
from alignment.results import finalize, reconcile, start_run
from alignment.runmode import add_mode_arg, resolve_model


def conversational(rows: list[dict]) -> Dataset:
    return Dataset.from_list([{"prompt": [{"role": "user", "content": r["prompt"]}],
                               "chosen": [{"role": "assistant", "content": r["chosen"]}],
                               "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in rows])


def split(rows: list[dict], holdout: int, seed: int) -> tuple[list[int], list[int]]:
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    return sorted(idx[holdout:]), sorted(idx[:holdout])


def summarize(log: list[dict]) -> dict:
    """One held-out evaluation per epoch, from the trainer's log history."""
    evals = [r for r in log if "eval_rewards/accuracies" in r]
    trains = [r for r in log if "loss" in r]
    keys = ("epoch", "eval_loss", "eval_rewards/accuracies", "eval_rewards/margins", "eval_logps/chosen", "eval_logps/rejected")
    return {"n_logged_train": len(trains), "mean_train_loss": next(r["train_loss"] for r in log if "train_loss" in r),
            "evals": [{k: r[k] for k in keys} for r in evals]}


def main() -> int:
    ap = argparse.ArgumentParser()
    add_mode_arg(ap)
    ap.add_argument("--config", default="configs/lora_persona.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(repo_path(args.config).read_text())
    smoke = args.mode == "smoke"
    if smoke:
        cfg.update(output="outputs/_smoke/persona", results="results/_smoke/persona", epochs=1, max_rows=24, holdout=8)
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()

    rows = read_json(cfg["data"])[: cfg.get("max_rows")]
    train_idx, eval_idx = split(rows, cfg["holdout"], cfg["seed"])
    tok = AutoTokenizer.from_pretrained(resolve_model(cfg["model"], args.mode))
    model = AutoModelForCausalLM.from_pretrained(resolve_model(cfg["model"], args.mode), torch_dtype=torch.float32)
    peft_config = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                             target_modules=cfg["target_modules"], task_type="CAUSAL_LM")
    out = repo_path(cfg["output"])
    targs = DPOConfig(output_dir=str(out / "trainer"), beta=cfg["beta"], loss_type="sigmoid", learning_rate=cfg["lr"],
                      lr_scheduler_type=cfg["lr_scheduler"], warmup_ratio=cfg["warmup_ratio"],
                      per_device_train_batch_size=cfg["per_device_batch"], per_device_eval_batch_size=cfg["per_device_batch"],
                      gradient_accumulation_steps=cfg["grad_accum"], num_train_epochs=cfg["epochs"],
                      max_length=cfg["max_length"], max_prompt_length=cfg["max_prompt_length"],
                      bf16=bf16, fp16=cuda and not bf16, gradient_checkpointing=False, eval_strategy="epoch",
                      logging_steps=5, save_strategy="no", report_to=[], seed=cfg["seed"], use_cpu=not cuda)
    trainer = DPOTrainer(model=model, ref_model=None, args=targs, processing_class=tok, peft_config=peft_config,
                         train_dataset=conversational([rows[i] for i in train_idx]),
                         eval_dataset=conversational([rows[i] for i in eval_idx]))
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"trainable params {trainable:,} / {total:,} = {100 * trainable / total:.2f}%", flush=True)

    run = start_run(cfg["results"], mode=args.mode, cfg=cfg, seed=cfg["seed"])
    step0 = trainer.evaluate()
    print(f"step-0 held-out accuracy {step0['eval_rewards/accuracies']} (LoRA B = 0, so policy = reference, must be 0), "
          f"loss {step0['eval_loss']:.6f}", flush=True)
    trainer.train()
    trainer.model.save_pretrained(out / "adapter")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")

    log_path = run["outdir"] / "log_history.jsonl"
    write_jsonl(log_path, trainer.state.log_history)
    summary = summarize(read_jsonl(log_path))
    ep1 = summary["evals"][0]
    overfit = ep1["eval_rewards/accuracies"] >= 1.0 and ep1["eval_logps/chosen"] < step0["eval_logps/chosen"]
    print(f"{'BREACH' if overfit else 'PASS'}: epoch-1 held-out accuracy {ep1['eval_rewards/accuracies']:.3f}, "
          f"held-out logp(chosen) {step0['eval_logps/chosen']:.2f} -> {ep1['eval_logps/chosen']:.2f}", flush=True)
    finalize(run, log_path, summary, summarize, extra={
        "step0_eval": {k: step0[k] for k in ("eval_loss", "eval_rewards/accuracies", "eval_rewards/margins", "eval_logps/chosen")},
        "trainable_params": trainable, "total_params": total, "train_ids": train_idx, "eval_ids": eval_idx,
        "precision": "bf16" if bf16 else "fp16" if cuda else "fp32",
        "merged_sha256": sha256_file(out / "merged" / "model.safetensors")})
    reconcile(run["outdir"], {"log_history.jsonl", "summary.json"})
    return 1 if overfit else 0


if __name__ == "__main__":
    sys.exit(main())
