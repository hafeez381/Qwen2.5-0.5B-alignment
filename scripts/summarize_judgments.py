"""Join judge verdicts with the hidden A/B mapping and summarize wins, ties, losses, unknowns and failures.

Writes per-pair rows (verdict translated to the model that won) and a summary recomputed from those rows:
win rate with ties counted as half and without ties, Wilson 95% intervals, per-category rates, the share of decided
verdicts that went to position A, and, given the swapped pass, how often the same model wins in both orders.
Prints PASS/BREACH for judge reliability (unknown + failure <= 5%, swap consistency >= 70%, A-position rate <= 65% in
each pass) and, with --compare, for whether two rubrics on the same pairs agree on at most 90% of them; exits 1 on a
breach after writing the files, so the numbers are kept and reported.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict

from alignment.io import read_json, read_jsonl, write_json, write_jsonl


def wilson(k: float, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def outcome(verdict: str, mapping: dict, focal: str) -> str:
    if verdict in ("A", "B"):
        return "win" if mapping[verdict] == focal else "loss"
    return verdict


def rates(rows: list[dict]) -> dict:
    c = defaultdict(int)
    for r in rows:
        c[r["outcome"]] += 1
    w, t, l = c["win"], c["tie"], c["loss"]
    decided = w + t + l
    return {"n": len(rows), "win": w, "tie": t, "loss": l, "unknown": c["unknown"], "failure": c["failure"],
            "win_rate_with_ties": (w + 0.5 * t) / decided if decided else None,
            "win_rate_without_ties": w / (w + l) if w + l else None,
            "ci95_with_ties": wilson(w + 0.5 * t, decided), "ci95_without_ties": wilson(w, w + l)}


def summarize(rows: list[dict]) -> dict:
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"] or "all"].append(r)
    decided = [r for r in rows if r["verdict"] in ("A", "B")]
    swap = [r for r in rows if r.get("swap_outcome") in ("win", "loss", "tie")]
    swap_decided = [r for r in rows if r.get("swap_verdict") in ("A", "B")]
    return {**rates(rows),
            "a_position_rate": sum(r["verdict"] == "A" for r in decided) / len(decided) if decided else None,
            "a_position_rate_swapped": (sum(r["swap_verdict"] == "A" for r in swap_decided) / len(swap_decided)
                                        if swap_decided else None),
            "swap_pairs": len(swap),
            "swap_consistency": sum(r["swap_outcome"] == r["outcome"] for r in swap) / len(swap) if swap else None,
            "by_category": {k: rates(v) for k, v in sorted(by_cat.items())} if len(by_cat) > 1 else {}}


def gates(s: dict) -> list[tuple[bool, str]]:
    bad = (s["unknown"] + s["failure"]) / s["n"]
    out = [(bad <= 0.05, f"unknown + failure {bad:.3f} (max 0.05)")]
    if s["swap_consistency"] is not None:
        out.append((s["swap_consistency"] >= 0.70, f"swap consistency {s['swap_consistency']:.3f} (min 0.70)"))
    for key in ("a_position_rate", "a_position_rate_swapped"):
        if s[key] is not None:
            out.append((s[key] <= 0.65, f"{key} {s[key]:.3f} (max 0.65)"))
    return out


def agreement(a: list[dict], b: list[dict]) -> float:
    other = {r["pair_id"]: r["outcome"] for r in b}
    shared = [r for r in a if r["pair_id"] in other]
    return sum(r["outcome"] == other[r["pair_id"]] for r in shared) / len(shared)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="results/eval/<name>")
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--compare", help="another rubric judged on the same pairs; gate: outcomes agree on <= 90%")
    args = ap.parse_args()
    d = f"results/eval/{args.name}"
    meta = read_json(f"{d}/mapping.json")
    focal, mapping = meta["focal"], meta["pairs"]
    pairs = read_jsonl(f"{d}/pairs.jsonl")

    def load(suffix):
        try:
            judged = {r["pair_id"]: r for r in read_jsonl(f"{d}/judged_{args.rubric}{suffix}.jsonl")}
        except FileNotFoundError:
            return None, set()
        try:
            failed = {r["pair_id"] for r in read_jsonl(f"{d}/judged_{args.rubric}{suffix}_failures.jsonl")}
        except FileNotFoundError:
            failed = set()
        return judged, failed

    main_j, main_f = load("")
    swap_j, _ = load("_swapped")
    if main_j is None:
        raise SystemExit(f"no judgments in {d}/judged_{args.rubric}.jsonl")
    rows = []
    for p in pairs:
        pid = p["pair_id"]
        j = main_j.get(pid)
        verdict = j["verdict"] if j else ("failure" if pid in main_f else "missing")
        row = {"pair_id": pid, "category": p["category"], "verdict": verdict,
               "outcome": outcome(verdict, mapping[pid], focal) if verdict != "missing" else "failure"}
        if swap_j and pid in swap_j:
            flipped = {"A": mapping[pid]["B"], "B": mapping[pid]["A"]}
            row["swap_verdict"] = swap_j[pid]["verdict"]
            row["swap_outcome"] = outcome(swap_j[pid]["verdict"], flipped, focal)
        rows.append(row)
    write_jsonl(f"{d}/outcomes_{args.rubric}.jsonl", rows)
    summary = summarize(read_jsonl(f"{d}/outcomes_{args.rubric}.jsonl"))
    if summary != summarize(rows):
        raise AssertionError("summary does not match the persisted rows")
    judge_models = sorted({r["judge_model"] for r in main_j.values()})
    write_json(f"{d}/summary_{args.rubric}.json", {"focal": focal, "baseline": meta["baseline"], "rubric": args.rubric,
                                                    "judge_models": judge_models, "rows": f"{d}/outcomes_{args.rubric}.jsonl",
                                                    "summary": summary})
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}))
    checks = gates(summary)
    if args.compare:
        agree = agreement(rows, read_jsonl(f"{d}/outcomes_{args.compare}.jsonl"))
        checks.append((agree <= 0.90, f"{args.rubric}/{args.compare} outcome agreement {agree:.3f} (max 0.90)"))
    for ok, msg in checks:
        print(("PASS: " if ok else "BREACH: ") + msg)
    return 0 if all(ok for ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
