"""Runtime integrity guards. Each one raises: a violation invalidates the run."""
from __future__ import annotations

import re
from pathlib import Path

from alignment.io import REPO_ROOT, read_json, sha256_file

MANIFEST = "manifests/sha256.json"


class GuardError(RuntimeError):
    pass


def assert_fixed_files(mode: str = "full", root: str | Path = REPO_ROOT, entries: dict | None = None) -> list[str]:
    """Check fixed files against the sha256 manifest. A missing file is fatal in full mode; returns missing paths."""
    root = Path(root)
    entries = read_json(root / MANIFEST) if entries is None else entries
    missing = [rel for rel in sorted(entries) if not (root / rel).exists()]
    changed = [rel for rel, h in sorted(entries.items()) if rel not in missing and sha256_file(root / rel) != h]
    if changed:
        raise GuardError("Fixed files changed: " + ", ".join(changed))
    if missing and mode == "full":
        raise GuardError("Fixed files missing: " + ", ".join(missing))
    return missing


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def assert_no_eval_overlap(train_prompts, eval_prompts) -> None:
    """No training prompt may equal an evaluation prompt after normalization."""
    evals = {normalize_prompt(p) for p in eval_prompts}
    hits = sorted({p for p in map(normalize_prompt, train_prompts) if p in evals})
    if hits:
        raise GuardError(f"{len(hits)} training prompt(s) equal an eval prompt, e.g. {hits[0][:80]!r}")
