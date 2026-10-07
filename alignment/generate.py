"""Greedy generation on a fixed evaluation set, written row by row so an interrupted run resumes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from alignment.format import render_prompt
from alignment.io import read_json, read_jsonl, repo_path, write_json
from alignment.results import finalize, reconcile, start_run
from alignment.runmode import add_mode_arg, resolve_model


def generation_config(cfg: dict, tok) -> dict:
    stop = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None, tok.unk_token_id})
    return {"do_sample": False, "max_new_tokens": cfg["max_new_tokens"], "repetition_penalty": cfg["repetition_penalty"],
            "eos_token_id": stop, "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else stop[0]}


def config_hash(gen_cfg: dict) -> str:
    return hashlib.sha256(json.dumps(gen_cfg, sort_keys=True).encode()).hexdigest()[:16]


@torch.no_grad()
def generate_batch(model, tok, prompts: list[str], gen_cfg: dict) -> list[dict]:
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    model.generation_config = GenerationConfig(**gen_cfg)
    out = model.generate(**enc)[:, enc["input_ids"].shape[1]:]
    rows = []
    for new in out.tolist():
        cut = next((i for i, t in enumerate(new) if t in gen_cfg["eos_token_id"]), None)
        kept = new if cut is None else new[:cut]
        rows.append({"output": tok.decode(kept, skip_special_tokens=True).strip(), "n_new_tokens": len(kept),
                     "stopped_on_eos": cut is not None})
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {"n": n, "mean_new_tokens": sum(r["n_new_tokens"] for r in rows) / n,
            "stop_rate": sum(r["stopped_on_eos"] for r in rows) / n,
            "empty_rate": sum(not r["output"] for r in rows) / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    add_mode_arg(ap)
    ap.add_argument("--model", required=True, help="hub id or local checkpoint directory")
    ap.add_argument("--tokenizer", help="defaults to --model")
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="configs/generate.yaml")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(repo_path(args.config).read_text())
    smoke = args.mode == "smoke"
    if smoke:
        cfg.update(max_new_tokens=16, batch_size=2)
    torch.manual_seed(cfg["seed"])
    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = AutoModelForCausalLM.from_pretrained(resolve_model(args.model, args.mode), torch_dtype=torch.float16 if cuda else torch.float32)
    model = model.to("cuda" if cuda else "cpu").eval()
    gen_cfg = generation_config(cfg, tok)
    items = read_json(args.eval_set)[: (2 if smoke else args.limit)]

    name = f"{args.generator}__{args.dataset}" + (f"__first{args.limit}" if args.limit else "")
    run = start_run(f"results/{'_smoke/' if smoke else ''}generations/{name}", mode=args.mode,
                    cfg={**cfg, "model": args.model, "eval_set": args.eval_set, "generation_config": gen_cfg},
                    seed=cfg["seed"], resume=args.resume)
    rows_path = run["outdir"] / "generations.jsonl"
    done = {r["id"] for r in read_jsonl(rows_path)} if args.resume and rows_path.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    for i in range(0, len(todo), cfg["batch_size"]):
        part = todo[i : i + cfg["batch_size"]]
        outs = generate_batch(model, tok, [render_prompt(tok, it["instruction"]) for it in part], gen_cfg)
        with rows_path.open("a") as f:
            for it, o in zip(part, outs):
                row = {"id": it["id"], "instruction": it["instruction"], "output": o["output"],
                       "generator": args.generator, "dataset": args.dataset,
                       **({"category": it["category"]} if "category" in it else {}),
                       "n_new_tokens": o["n_new_tokens"], "stopped_on_eos": o["stopped_on_eos"],
                       "generation_config_hash": config_hash(gen_cfg)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{min(i + cfg['batch_size'], len(todo))}/{len(todo)}", flush=True)

    rows = read_jsonl(rows_path)
    if [r["id"] for r in rows] != [it["id"] for it in items]:
        raise AssertionError("generation rows do not match the eval set ids and order")
    write_json(run["outdir"] / "generations.json", rows)
    finalize(run, rows_path, summarize(rows), summarize, extra={"generation_config_hash": config_hash(gen_cfg)})
    reconcile(run["outdir"], {"generations.jsonl", "generations.json", "summary.json"})
    print(json.dumps(summarize(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
