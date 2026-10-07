"""Right-padded LM batches with response-only labels. Every mask is built from positions, never from token ids,
so a tokenizer whose pad token equals its EOS keeps the EOS label."""
from __future__ import annotations

import torch

IGNORE = -100


def encode_pair(tokenizer, prompt: str, response: str, max_length: int) -> tuple[list[int], int]:
    """Token ids of prompt + response + EOS, truncated to max_length, and the number of prompt tokens kept."""
    p = list(tokenizer.encode(prompt, add_special_tokens=False))
    r = list(tokenizer.encode(response, add_special_tokens=False))
    if tokenizer.eos_token_id is not None:
        r.append(tokenizer.eos_token_id)
    ids = (p + r)[:max_length]
    return ids, min(len(p), len(ids))


def pad_id(tokenizer) -> int:
    for tid in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        if tid is not None:
            return tid
    return 0


def build_lm_batch(tokenizer, prompts, responses, max_length: int, pad_to: str = "max_length") -> dict:
    encoded = [encode_pair(tokenizer, p, r, max_length) for p, r in zip(prompts, responses)]
    width = max_length if pad_to == "max_length" else max(1, max(len(ids) for ids, _ in encoded))
    input_ids = torch.full((len(encoded), width), pad_id(tokenizer), dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
    labels = torch.full((len(encoded), width), IGNORE, dtype=torch.long)
    for row, (ids, n_prompt) in enumerate(encoded):
        t = torch.tensor(ids, dtype=torch.long)
        input_ids[row, : len(ids)] = t
        attention_mask[row, : len(ids)] = 1
        labels[row, n_prompt : len(ids)] = t[n_prompt:]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_dpo_batch(tokenizer, prompts, chosen, rejected, max_length: int, pad_to: str = "max_length") -> dict:
    return {
        "chosen": build_lm_batch(tokenizer, prompts, chosen, max_length, pad_to),
        "rejected": build_lm_batch(tokenizer, prompts, rejected, max_length, pad_to),
    }
