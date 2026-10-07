"""Library boundary for the from-scratch SFT and DPO code: no TRL, no PEFT, no HF Trainer, no built-in model loss.

Scans alignment/*.py except persona.py (the TRL + LoRA task). Exit 1 and list the violations if any are found.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXEMPT = {"persona.py"}
BANNED_MODULES = ("trl", "peft")
BANNED_NAMES = {"Trainer", "SFTTrainer", "Seq2SeqTrainer", "DPOTrainer"}
# ponytail: a model call is recognized by its callee name; rename-proof detection would need type inference.
MODEL_NAMES = ("model", "policy", "reference", "ref")


def _callee(node: ast.Call) -> str:
    f = node.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


def violations(path: Path) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out += [f"import {a.name}" for a in node.names if a.name.split(".")[0] in BANNED_MODULES]
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in BANNED_MODULES:
                out.append(f"from {node.module} import ...")
            out += [f"from {node.module} import {a.name}" for a in node.names if a.name in BANNED_NAMES]
        elif isinstance(node, ast.Call):
            if _callee(node) in BANNED_NAMES:
                out.append(f"call {_callee(node)}()")
            if _callee(node).lower().endswith(MODEL_NAMES) and any(k.arg == "labels" for k in node.keywords):
                out.append(f"{_callee(node)}(..., labels=...) uses the model's built-in loss")
    return [f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}:{v}" for v in out]


def main(paths: list[str]) -> int:
    files = [Path(p) for p in paths] or [p for p in sorted((ROOT / "alignment").glob("*.py")) if p.name not in EXEMPT]
    found = [v for f in files for v in violations(f)]
    print("\n".join(found) if found else f"PASS: library boundary holds in {len(files)} file(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
