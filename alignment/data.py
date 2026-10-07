"""Convert the three source datasets to single-turn rows: {prompt, response} for SFT, {prompt, chosen, rejected} for DPO."""
from __future__ import annotations

import json

from alignment.io import sha256_text

HUMAN, ASSISTANT = "\n\nHuman:", "\n\nAssistant:"


def row_sha(row: dict) -> str:
    keys = ("prompt", "response", "chosen", "rejected")
    return sha256_text(json.dumps({k: row[k] for k in keys if k in row}, ensure_ascii=False, sort_keys=True))


def ultrachat_row(ex: dict) -> dict | None:
    msgs = ex["messages"]
    if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
        return None
    prompt, response = msgs[0]["content"].strip(), msgs[1]["content"].strip()
    return {"prompt": prompt, "response": response} if prompt and response else None


def _hh_split(text: str) -> tuple[str, str] | None:
    if text.count(HUMAN) != 1 or text.count(ASSISTANT) != 1:
        return None
    before, response = text.rsplit(ASSISTANT, 1)
    return before.split(HUMAN, 1)[1].strip(), response.strip()


def hh_row(ex: dict) -> dict | None:
    """Single-turn HH-RLHF pair: one Human turn, one Assistant turn, same prompt on both sides."""
    c, r = _hh_split(ex["chosen"]), _hh_split(ex["rejected"])
    if c is None or r is None or c[0] != r[0]:
        return None
    (prompt, chosen), (_, rejected) = c, r
    if not (prompt and chosen and rejected) or chosen == rejected:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def uf_row(ex: dict) -> dict | None:
    if ex["score_chosen"] == ex["score_rejected"]:
        return None
    prompt = ex["prompt"].strip()
    chosen, rejected = ex["chosen"][-1]["content"].strip(), ex["rejected"][-1]["content"].strip()
    if not (prompt and chosen and rejected) or chosen == rejected:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
