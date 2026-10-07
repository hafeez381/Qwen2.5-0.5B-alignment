from __future__ import annotations

import torch
import torch.nn.functional as F

from alignment.logprobs import response_logprobs


def sft_loss(model, batch: dict) -> torch.Tensor:
    """Mean negative log-prob over every response token in the batch."""
    logp, n_tokens = response_logprobs(model, batch)
    return -logp.sum() / n_tokens.sum().clamp_min(1)


def dpo_logits(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta: float) -> torch.Tensor:
    return beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))


def dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta: float) -> torch.Tensor:
    return -F.logsigmoid(dpo_logits(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)).mean()


@torch.no_grad()
def dpo_metrics(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta: float) -> dict:
    """Per-pair implicit rewards; accuracy uses the strict r_w > r_l, so identical models score 0."""
    r_w = beta * (policy_chosen - reference_chosen)
    r_l = beta * (policy_rejected - reference_rejected)
    return {"reward_chosen": r_w, "reward_rejected": r_l, "correct": r_w > r_l, "margin": r_w - r_l}


def pair_logprobs(policy, reference, batch: dict) -> tuple[torch.Tensor, ...]:
    """Policy log-probs with gradients; reference log-probs under no_grad, moved to the policy's device."""
    pc, _ = response_logprobs(policy, batch["chosen"])
    pr, _ = response_logprobs(policy, batch["rejected"])
    with torch.no_grad():
        rc, _ = response_logprobs(reference, batch["chosen"])
        rr, _ = response_logprobs(reference, batch["rejected"])
    return pc, pr, rc.to(pc.device), rr.to(pc.device)


def dpo_loss(policy, reference, batch: dict, beta: float) -> torch.Tensor:
    return dpo_loss_from_logps(*pair_logprobs(policy, reference, batch), beta)
