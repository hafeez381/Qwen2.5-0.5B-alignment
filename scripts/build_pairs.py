"""Blinded pairwise files from two generation files on the same prompts.

Writes results/eval/<name>/: pairs.jsonl (seeded random A/B order), pairs_swapped.jsonl (every pair with A and B
exchanged, for the position-consistency check), and mapping.json (pair_id -> which model is A in pairs.jsonl).
The judge reads only the pairs files; model identities live only in mapping.json.
"""
from __future__ import annotations

import argparse
import random
import sys

from alignment.io import read_jsonl, write_json, write_jsonl

PAIR_KEYS = {"pair_id", "instruction", "category", "response_a", "response_b"}


def build(focal: list[dict], other: list[dict], seed: int) -> tuple[list[dict], list[dict], dict]:
    if [r["id"] for r in focal] != [r["id"] for r in other]:
        raise SystemExit("BREACH: the two generation files differ in prompt ids or order")
    hashes = {r["generation_config_hash"] for r in focal + other}
    if len(hashes) != 1:
        raise SystemExit(f"BREACH: compared models used different decoding configs {sorted(hashes)}")
    rng = random.Random(seed)
    pairs, swapped, mapping = [], [], {}
    for f, o in zip(focal, other):
        focal_first = rng.random() < 0.5
        a, b = (f, o) if focal_first else (o, f)
        row = {"pair_id": f["id"], "instruction": f["instruction"], "category": f.get("category", ""),
               "response_a": a["output"], "response_b": b["output"]}
        pairs.append(row)
        swapped.append({**row, "response_a": row["response_b"], "response_b": row["response_a"]})
        mapping[f["id"]] = {"A": a["generator"], "B": b["generator"]}
    for row in pairs + swapped:
        assert set(row) == PAIR_KEYS
    return pairs, swapped, mapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", required=True, help="generations.jsonl of the model under test (DPO or LoRA)")
    ap.add_argument("--baseline", required=True, help="generations.jsonl of the comparison model (SFT or base)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    focal, other = read_jsonl(args.focal), read_jsonl(args.baseline)
    pairs, swapped, mapping = build(focal, other, args.seed)
    out = f"results/eval/{args.name}"
    write_jsonl(f"{out}/pairs.jsonl", pairs)
    write_jsonl(f"{out}/pairs_swapped.jsonl", swapped)
    write_json(f"{out}/mapping.json", {"focal": focal[0]["generator"], "baseline": other[0]["generator"],
                                       "seed": args.seed, "pairs": mapping})
    share = sum(m["A"] == focal[0]["generator"] for m in mapping.values()) / len(mapping)
    print(f"{args.name}: {len(pairs)} pairs, focal model in position A for {share:.3f} of pairs")
    return 0 if 0.4 <= share <= 0.6 else 1


if __name__ == "__main__":
    sys.exit(main())
