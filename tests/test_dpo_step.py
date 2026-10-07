import copy

import torch

from alignment.batching import build_dpo_batch
from alignment.losses import dpo_loss, pair_logprobs
from tests.helpers import BigramLM
from tests.toy_lm import ToyTokenizer

TOK = ToyTokenizer()


def test_one_step_widens_the_preference_and_leaves_the_reference_alone():
    policy = BigramLM(TOK.vocab_size, seed=1)
    reference = copy.deepcopy(policy)
    ref_before = reference.table.detach().clone()
    batch = build_dpo_batch(TOK, ["Pick: ", "Choose: "], ["yes", "good"], ["no", "bad"], 16)
    pc, pr, rc, rr = pair_logprobs(policy, reference, batch)
    torch.testing.assert_close(pc - pr, rc - rr)
    loss = dpo_loss(policy, reference, batch, beta=0.5)
    assert abs(loss.item() - torch.log(torch.tensor(2.0)).item()) < 1e-6
    loss.backward()
    with torch.no_grad():
        policy.table -= 1.0 * policy.table.grad
    pc2, pr2, _, _ = pair_logprobs(policy, reference, batch)
    assert ((pc2 - pr2) > (pc - pr)).all()
    assert reference.table.grad is None
    assert torch.equal(reference.table, ref_before)


def test_reference_logprobs_carry_no_graph():
    policy, reference = BigramLM(TOK.vocab_size, 1), BigramLM(TOK.vocab_size, 2)
    batch = build_dpo_batch(TOK, ["Q: "], ["a"], ["b"], 8)
    pc, _, rc, rr = pair_logprobs(policy, reference, batch)
    assert pc.requires_grad and not rc.requires_grad and not rr.requires_grad
