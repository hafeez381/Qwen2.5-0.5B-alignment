"""Pick a seeded, stratified handful of judged pairs to read and quote: improvements, regressions and near-ties.

Strata are prompt category x outcome (win, loss, tie for the focal model), visited round-robin so every category gets
at least one pick and outcomes alternate. Pairs whose verdict held when A and B were swapped are preferred.
usage: python scripts/select_examples.py --name hh_dpo_vs_sft --rubric safety [--k 10]
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

from alignment.io import read_json, read_jsonl, write_jsonl

OUTCOMES = ("win", "loss", "tie")


def select(rows: list[dict], k: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in rows:
        if r["outcome"] in OUTCOMES:
            strata[(r["category"] or "all", r["outcome"])].append(r)
    for v in strata.values():
        v.sort(key=lambda r: (r.get("swap_outcome") != r["outcome"], r["pair_id"]))
        stable = [r for r in v if r.get("swap_outcome") == r["outcome"]]
        rng.shuffle(stable)
        v[:len(stable)] = stable
    cats = sorted({c for c, _ in strata})
    picked, turn = [], 0
    while len(picked) < k and any(strata.values()):
        for c in cats:
            for j in range(len(OUTCOMES)):
                key = (c, OUTCOMES[(turn + j) % len(OUTCOMES)])
                if strata.get(key):
                    picked.append(strata[key].pop(0))
                    turn += 1
                    break
            if len(picked) == k:
                break
        turn += 1
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    d = f"results/eval/{args.name}"
    meta = read_json(f"{d}/mapping.json")
    pairs = {p["pair_id"]: p for p in read_jsonl(f"{d}/pairs.jsonl")}
    judged = {r["pair_id"]: r for r in read_jsonl(f"{d}/judged_{args.rubric}.jsonl")}
    out = []
    for r in select(read_jsonl(f"{d}/outcomes_{args.rubric}.jsonl"), args.k):
        p, side = pairs[r["pair_id"]], meta["pairs"][r["pair_id"]]
        focal_pos = "A" if side["A"] == meta["focal"] else "B"
        base_pos = "B" if focal_pos == "A" else "A"
        out.append({"pair_id": r["pair_id"], "category": r["category"], "outcome": r["outcome"],
                    "swap_outcome": r.get("swap_outcome"), "instruction": p["instruction"],
                    meta["focal"]: p[f"response_{focal_pos.lower()}"], meta["baseline"]: p[f"response_{base_pos.lower()}"],
                    "judge_reason": judged[r["pair_id"]]["reason"], "focal_position": focal_pos})
    write_jsonl(f"{d}/examples_{args.rubric}.jsonl", out)
    print(f"{args.name}/{args.rubric}: {len(out)} examples ->",
          {o: sum(x["outcome"] == o for x in out) for o in OUTCOMES}, sorted({x["category"] or "all" for x in out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
