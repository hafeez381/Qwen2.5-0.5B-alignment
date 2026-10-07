"""Gates on a generation file: stop/template health and the repetition-loop rate that decides the penalty once."""
from __future__ import annotations

import argparse
import sys

from alignment.io import read_jsonl

NO_STOP_MAX = 0.10
LOOP_MAX = 0.10


def has_loop(text: str, max_period: int = 50, repeats: int = 3) -> bool:
    """True if some span of 1..max_period words occurs `repeats` times back to back."""
    w = text.split()
    for p in range(1, max_period + 1):
        for i in range(len(w) - p * repeats + 1):
            if all(w[i + k * p : i + (k + 1) * p] == w[i : i + p] for k in range(1, repeats)):
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="generations.jsonl files, pooled")
    args = ap.parse_args()
    rows = [r for f in args.files for r in read_jsonl(f)]
    n = len(rows)
    bad = [r["id"] for r in rows if not r["stopped_on_eos"] or "<|im_start|>" in r["output"]]
    loops = [r["id"] for r in rows if has_loop(r["output"])]
    ok_stop = len(bad) / n <= NO_STOP_MAX
    print(f"{'PASS' if ok_stop else 'BREACH'}: no-stop or template leak {len(bad)}/{n} = {len(bad) / n:.3f} (max {NO_STOP_MAX}) {bad[:10]}")
    print(f"loop rate {len(loops)}/{n} = {len(loops) / n:.3f} -> repetition_penalty "
          f"{'1.1' if len(loops) / n > LOOP_MAX else '1.0'} {loops[:10]}")
    return 0 if ok_stop else 1


if __name__ == "__main__":
    sys.exit(main())
