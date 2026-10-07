"""Seeding, atomic checkpoint writes and resumable snapshots shared by the SFT and DPO loops."""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from alignment.io import sha256_file


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_model(model, tok, path: Path) -> None:
    tmp = path.with_name(path.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tok.save_pretrained(tmp)
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def save_snapshot(model, tok, trainer: dict, state: dict, path: Path) -> None:
    """Weights, `trainer` state_dicts, RNG states and loop state, written to a temp dir then renamed."""
    tmp = path.with_name(path.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tok.save_pretrained(tmp)
    torch.save({**{k: v.state_dict() for k, v in trainer.items()}, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, tmp / "trainer.pt")
    (tmp / "state.json").write_text(json.dumps(state))
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)
    for f in ("model.safetensors", "trainer.pt"):
        print(f"SNAPSHOT step={state['step']} {f} sha256={sha256_file(path / f)}", flush=True)


def load_snapshot(trainer: dict, path: Path) -> dict:
    ckpt = torch.load(path / "trainer.pt", weights_only=False)
    for k, v in trainer.items():
        v.load_state_dict(ckpt[k])
    torch.set_rng_state(ckpt["torch_rng"])
    if torch.cuda.is_available() and ckpt["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    return json.loads((path / "state.json").read_text())
