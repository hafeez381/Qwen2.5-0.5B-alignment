from __future__ import annotations

import torch

from alignment.batching import IGNORE


def response_logprobs(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed log-prob of the response tokens of each row, and how many response tokens were scored.

    Position t predicts token t+1, so logits[:, :-1] are scored against labels[:, 1:].
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logp = logits[:, :-1].float().log_softmax(-1)
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE)
    token_logp = logp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    token_logp = torch.where(mask, token_logp, torch.zeros_like(token_logp))
    return token_logp.sum(-1), mask.sum(-1)
