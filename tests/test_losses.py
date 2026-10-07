import math

import torch

from alignment.batching import build_lm_batch
from alignment.logprobs import response_logprobs
from alignment.losses import dpo_loss_from_logps, dpo_metrics, sft_loss
from tests.helpers import BigramLM
from tests.toy_lm import ToyTokenizer

TOK = ToyTokenizer()
LOG2 = math.log(2)


def test_sft_loss_is_the_token_mean():
    model = BigramLM(TOK.vocab_size)
    batch = build_lm_batch(TOK, ["Do: ", "Say: "], ["a", "longer"], 16)
    logp, n = response_logprobs(model, batch)
    torch.testing.assert_close(sft_loss(model, batch), -logp.sum() / n.sum())


def test_sft_loss_with_no_response_tokens_is_finite_zero():
    model = BigramLM(TOK.vocab_size)
    loss = sft_loss(model, build_lm_batch(TOK, ["long prompt: "], ["x"], 4))
    assert loss.item() == 0.0


def test_dpo_hand_computed_value():
    t = lambda v: torch.tensor([v])
    loss = dpo_loss_from_logps(t(-1.0), t(-3.0), t(-2.0), t(-2.0), beta=0.1)
    torch.testing.assert_close(loss, torch.tensor(math.log1p(math.exp(-0.2))))
    assert abs(loss.item() - 0.598139) < 1e-6


def test_dpo_limits_and_sign():
    pc, pr = torch.tensor([-1.0, -2.0]), torch.tensor([-3.0, -4.0])
    rc, rr = torch.tensor([-2.0, -2.0]), torch.tensor([-2.0, -2.0])
    assert abs(dpo_loss_from_logps(pc, pc, pc, pc, 0.5).item() - LOG2) < 1e-7
    assert abs(dpo_loss_from_logps(pc, pr, rc, rr, 0.0).item() - LOG2) < 1e-7
    assert dpo_loss_from_logps(pr, pc, rc, rr, 0.1) > dpo_loss_from_logps(pc, pr, rc, rr, 0.1)


def test_dpo_keeps_float64_and_is_a_batch_mean():
    x = torch.tensor([-1.0, -2.0], dtype=torch.float64)
    y = torch.tensor([-3.0, -1.0], dtype=torch.float64)
    z = torch.zeros(2, dtype=torch.float64)
    loss = dpo_loss_from_logps(x, y, z, z, 0.1)
    assert loss.dtype == torch.float64 and loss.ndim == 0
    each = [dpo_loss_from_logps(x[i:i + 1], y[i:i + 1], z[:1], z[:1], 0.1) for i in range(2)]
    torch.testing.assert_close(loss, (each[0] + each[1]) / 2)


def test_identical_policy_and_reference_give_zero_accuracy():
    v = torch.tensor([-5.0, -7.0, -1.0])
    m = dpo_metrics(v, v - 1, v, v - 1, 0.1)
    assert m["correct"].float().mean().item() == 0.0
    assert m["margin"].abs().max().item() == 0.0
