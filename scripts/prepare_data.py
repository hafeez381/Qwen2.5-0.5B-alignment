"""Build the frozen train/validation splits and the AlpacaEval subset.

Full mode writes data/splits/*.jsonl (row ids and sha256 only), eval_sets/alpaca_eval_300_seed42.json,
results/data/ (per-row token statistics, dropped rows, summary), data/prepared/*.jsonl (texts, gitignored)
and manifests/sha256.json.
--verify rebuilds everything in memory, checks it equals the committed files, then writes data/prepared/.
Smoke mode draws tiny splits from the small test files and writes everything under results/_smoke/data/.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from alignment.data import hh_row, row_sha, uf_row, ultrachat_row
from alignment.format import render_prompt
from alignment.guards import MANIFEST, assert_fixed_files, normalize_prompt
from alignment.io import read_json, read_jsonl, repo_path, sha256_file, write_json, write_jsonl
from alignment.results import finalize, reconcile, start_run
from alignment.runmode import add_mode_arg

SEED = 42
TOKENIZER = "Qwen/Qwen2.5-0.5B"
MAX_PROMPT_TOKENS = 256
MAX_LENGTH = 512
HH_SUBSETS = ["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"]
REVISIONS = {
    "HuggingFaceH4/ultrachat_200k": "8049631c405ae6576f93f445c6b8166f76f5505a",
    "HuggingFaceH4/ultrafeedback_binarized": "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85",
    "Anthropic/hh-rlhf": "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa",
    "tatsu-lab/alpaca_eval": "2edc6fad8be6b14ea7230aabfd08188da6b8b814",
}
SOURCES = {
    "sft": ("HuggingFaceH4/ultrachat_200k", ultrachat_row, {"train": "data/train_sft-*.parquet", "val": "data/test_sft-*.parquet"}),
    "hh": ("Anthropic/hh-rlhf", hh_row, {"train": "train.jsonl.gz", "val": "test.jsonl.gz"}),
    "uf": ("HuggingFaceH4/ultrafeedback_binarized", uf_row, {"train": "data/train_prefs-*.parquet", "val": "data/test_prefs-*.parquet"}),
}
SIZES = {"full": {"train": {"sft": 30000, "hh": 15000, "uf": 15000}, "val": 1000},
         "smoke": {"train": {"sft": 40, "hh": 40, "uf": 40}, "val": 10}}
ALPACA_N = 300
HH_POOL_MIN = 16000
UF_TRUNC_MAX = 0.25
JACCARD_MAX = 0.8


def paths(mode: str) -> dict:
    if mode == "full":
        return {"splits": "data/splits", "alpaca": "eval_sets/alpaca_eval_300_seed42.json",
                "prepared": "data/prepared", "results": "results/data"}
    root = "results/_smoke/data"
    return {"splits": f"{root}/splits", "alpaca": f"{root}/alpaca_eval_300_seed42.json",
            "prepared": f"{root}/prepared", "results": f"{root}/summary"}


def load_parts(name: str, split: str, mode: str) -> list[tuple[str, object]]:
    repo, _, files = SOURCES[name]
    pattern = files["val" if mode == "smoke" else split]
    base = f"hf://datasets/{repo}@{REVISIONS[repo]}"
    if name == "hh":
        return [(s, load_dataset("json", data_files=f"{base}/{s}/{pattern}", split="train")) for s in HH_SUBSETS]
    return [("", load_dataset("parquet", data_files=f"{base}/{pattern}", split="train"))]


def build_alpaca() -> list[dict]:
    repo = "tatsu-lab/alpaca_eval"
    path = hf_hub_download(repo, "alpaca_eval.json", repo_type="dataset", revision=REVISIONS[repo])
    rows = read_json(path)
    picked = random.Random(SEED).sample(range(len(rows)), ALPACA_N)
    return [{"id": f"alpaca-42-{k:04d}", "instruction": rows[i]["instruction"], "dataset": "alpaca_eval",
             "source_dataset": rows[i]["dataset"], "source_index": i} for k, i in enumerate(picked)]


def _jaccard(a: str, b: str) -> float:
    x, y = set(normalize_prompt(a).split()), set(normalize_prompt(b).split())
    return len(x & y) / max(len(x | y), 1)


def check_persona_eval(persona_eval: list[dict]) -> None:
    train = [r["prompt"] for r in read_json("data/persona_lora/ramsay_style_preferences.json")]
    hits = [(e["id"], t) for e in persona_eval for t in train if _jaccard(e["instruction"], t) >= JACCARD_MAX]
    if hits:
        raise SystemExit(f"BREACH: persona eval prompt(s) near-duplicate a training prompt: {hits[:3]}")
    counts = Counter(e["category"] for e in persona_eval)
    print(f"PASS: persona eval has {len(persona_eval)} prompts {dict(counts)}, none near a training prompt")


def select(name: str, split: str, n: int, tok, blocked: set[str], mode: str):
    repo, convert, _ = SOURCES[name]
    parts = load_parts(name, split, mode)
    keys = [(p, i) for p, (_, ds) in enumerate(parts) for i in range(len(ds))]
    random.Random(SEED).shuffle(keys)
    split_name = f"{name}_{split}"
    records, stats, texts, drops, pool = [], [], [], [], 0
    for p, i in keys:
        subset, ds = parts[p]
        row = convert(ds[i])
        pool += row is not None
        if len(records) == n:
            continue
        index = f"{subset}:{i}" if subset else str(i)
        if row is None:
            drops.append({"split_name": split_name, "index": index, "reason": "converter_rejected"})
            continue
        if normalize_prompt(row["prompt"]) in blocked:
            drops.append({"split_name": split_name, "index": index, "reason": "eval_or_val_overlap"})
            continue
        prompt_tokens = len(tok.encode(render_prompt(tok, row["prompt"]), add_special_tokens=False))
        if prompt_tokens > MAX_PROMPT_TOKENS:
            drops.append({"split_name": split_name, "index": index, "reason": "prompt_over_256_tokens"})
            continue
        rid = f"{split_name}-{len(records):05d}"
        records.append({"id": rid, "source": repo, "revision": REVISIONS[repo], "index": index, "sha256": row_sha(row)})
        stat = {"split_name": split_name, "id": rid, "prompt_tokens": prompt_tokens}
        for key in ("response", "chosen", "rejected"):
            if key in row:
                n_tok = len(tok.encode(row[key], add_special_tokens=False)) + 1
                stat[f"{key}_tokens"] = n_tok
                stat[f"{key}_truncated"] = prompt_tokens + n_tok > MAX_LENGTH
        if subset:
            stat["subset"] = subset
        stats.append(stat)
        texts.append({"id": rid, **row, **({"subset": subset} if subset else {})})
    if len(records) < n:
        raise SystemExit(f"BREACH: {split_name} has only {len(records)} usable rows, need {n}")
    return records, stats, texts, drops, pool


def summarize(rows: list[dict]) -> dict:
    out = {}
    for name in sorted({r["split_name"] for r in rows}):
        part = [r for r in rows if r["split_name"] == name]
        s = {"n": len(part), "mean_prompt_tokens": sum(r["prompt_tokens"] for r in part) / len(part)}
        for key in ("response", "chosen", "rejected"):
            if f"{key}_tokens" in part[0]:
                s[f"mean_{key}_tokens"] = sum(r[f"{key}_tokens"] for r in part) / len(part)
                s[f"{key}_truncated_rate"] = sum(r[f"{key}_truncated"] for r in part) / len(part)
        if "subset" in part[0]:
            s["subsets"] = dict(sorted(Counter(r["subset"] for r in part).items()))
        out[name] = s
    return out


def build(mode: str):
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    alpaca = build_alpaca()
    eval_prompts = [r["instruction"] for r in alpaca] + [r["instruction"] for r in read_json("eval_sets/hh_rlhf_eval_300_seed42.json")]
    if mode == "full":
        persona_eval = read_json("eval_sets/persona_style_eval.json")
        check_persona_eval(persona_eval)
        eval_prompts += [r["instruction"] for r in persona_eval]
    sizes = SIZES[mode]
    out = {"records": {}, "stats": [], "texts": {}, "drops": [], "pool": {}}
    for name in SOURCES:
        blocked = {normalize_prompt(p) for p in eval_prompts}
        for split in ("val", "train"):
            n = sizes["val"] if split == "val" else sizes["train"][name]
            records, stats, texts, drops, pool = select(name, split, n, tok, blocked, mode)
            if split == "val":
                blocked |= {normalize_prompt(t["prompt"]) for t in texts}
            key = f"{name}_{split}"
            out["records"][key], out["texts"][key], out["pool"][key] = records, texts, pool
            out["stats"] += stats
            out["drops"] += drops
            print(f"{key}: kept {len(records)}, single-turn pool {pool}, dropped {Counter(d['reason'] for d in drops)}")
    return alpaca, out


def write_prepared(p: dict, texts: dict) -> None:
    shutil.rmtree(repo_path(p["prepared"]), ignore_errors=True)
    for key, rows in texts.items():
        write_jsonl(f"{p['prepared']}/{key}.jsonl", rows)


def gates(summary: dict, pool: dict, mode: str) -> bool:
    if mode != "full":
        return True
    ok = True
    hh_pool = pool["hh_train"]
    print(f"{'PASS' if hh_pool >= HH_POOL_MIN else 'BREACH'}: HH single-turn train pool {hh_pool} (need >= {HH_POOL_MIN})")
    ok &= hh_pool >= HH_POOL_MIN
    uf_trunc = summary["uf_train"]["chosen_truncated_rate"]
    print(f"{'PASS' if uf_trunc <= UF_TRUNC_MAX else 'BREACH'}: UF chosen truncated at {MAX_LENGTH} tokens {uf_trunc:.4f} (max {UF_TRUNC_MAX})")
    return ok and uf_trunc <= UF_TRUNC_MAX


def main() -> int:
    ap = argparse.ArgumentParser()
    add_mode_arg(ap)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    p = paths(args.mode)
    alpaca, out = build(args.mode)

    if args.verify:
        bad = [k for k, recs in out["records"].items() if read_jsonl(f"{p['splits']}/{k}.jsonl") != recs]
        if read_json(p["alpaca"]) != alpaca:
            bad.append("alpaca")
        if bad:
            print(f"BREACH: rebuilt data differs from committed files: {bad}")
            return 1
        write_prepared(p, out["texts"])
        if args.mode == "full":
            assert_fixed_files("full")
        print("PASS: rebuilt splits and AlpacaEval subset equal the committed files; texts written to", p["prepared"])
        return 0

    run = start_run(p["results"], mode=args.mode, cfg={"seed": SEED, "tokenizer": TOKENIZER, "max_prompt_tokens": MAX_PROMPT_TOKENS,
                                                        "max_length": MAX_LENGTH, "sizes": SIZES[args.mode], "revisions": REVISIONS}, seed=SEED)
    shutil.rmtree(repo_path(p["splits"]), ignore_errors=True)
    for key, recs in out["records"].items():
        write_jsonl(f"{p['splits']}/{key}.jsonl", recs)
    write_json(p["alpaca"], alpaca)
    write_prepared(p, out["texts"])
    rows_path, drops_path = run["outdir"] / "rows.jsonl", run["outdir"] / "drops.jsonl"
    write_jsonl(rows_path, out["stats"])
    write_jsonl(drops_path, out["drops"])
    summary = summarize(out["stats"])
    dropped = {}
    for d in read_jsonl(drops_path):
        dropped.setdefault(d["split_name"], Counter())[d["reason"]] += 1
    finalize(run, rows_path, summary, summarize, extra={
        "single_turn_pool": out["pool"], "dropped": {k: dict(v) for k, v in sorted(dropped.items())},
        "alpaca_subset": {"n": len(alpaca), "source_dataset": dict(Counter(r["source_dataset"] for r in alpaca))}})
    reconcile(run["outdir"], {"rows.jsonl", "drops.jsonl", "summary.json"})

    if args.mode == "full":
        fixed = sorted([*map(str, ["eval_sets/hh_rlhf_eval_300_seed42.json", p["alpaca"], "eval_sets/persona_style_eval.json",
                                    "data/persona_lora/ramsay_style_preferences.json"]),
                        *(f"{p['splits']}/{k}.jsonl" for k in out["records"])])
        write_json(MANIFEST, {rel: sha256_file(rel) for rel in fixed})
        assert_fixed_files("full")
    return 0 if gates(summary, out["pool"], args.mode) else 1


if __name__ == "__main__":
    sys.exit(main())
