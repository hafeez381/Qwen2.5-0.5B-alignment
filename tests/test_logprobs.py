import math

import torch

from alignment.batching import build_lm_batch
from alignment.logprobs import response_logprobs
from tests.helpers import BigramLM
from tests.toy_lm import ToyTokenizer

TOK = ToyTokenizer()


def test_matches_a_hand_computed_bigram_logprob():
    model = BigramLM(TOK.vocab_size)
    batch = build_lm_batch(TOK, ["ab"], ["c"], max_length=8)
    a, b, c, eos = TOK.encode("a")[0], TOK.encode("b")[0], TOK.encode("c")[0], TOK.eos_token_id
    table = model.table.detach().log_softmax(-1)
    expected = table[b, c] + table[c, eos]
    logp, n = response_logprobs(model, batch)
    assert n.tolist() == [2]
    torch.testing.assert_close(logp[0], expected)


def test_only_response_tokens_count():
    model = BigramLM(TOK.vocab_size)
    with torch.no_grad():
        logp1, _ = response_logprobs(model, build_lm_batch(TOK, ["xyz: "], ["ok"], max_length=16))
        model.table[TOK.encode("x")[0], TOK.encode("y")[0]] += 50.0
        logp2, _ = response_logprobs(model, build_lm_batch(TOK, ["xyz: "], ["ok"], max_length=16))
    torch.testing.assert_close(logp1, logp2)


def test_empty_prompt_leaves_the_first_response_token_unscored():
    model = BigramLM(TOK.vocab_size)
    _, n = response_logprobs(model, build_lm_batch(TOK, [""], ["ok"], max_length=8))
    assert n.tolist() == [2]


def test_padding_width_does_not_change_logprobs():
    model = BigramLM(TOK.vocab_size)
    args = (TOK, ["Q: ", "Longer task: "], ["a", "bcd"], 32)
    full, _ = response_logprobs(model, build_lm_batch(*args))
    longest, _ = response_logprobs(model, build_lm_batch(*args, pad_to="longest"))
    torch.testing.assert_close(full, longest)


def test_row_order_and_batch_vs_single_on_a_real_transformer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", torch_dtype=torch.float32).eval()
    prompts = ["Say hi.", "Write one long sentence about the sea and the sky."]
    responses = ["Hi!", "The sea mirrors the sky."]
    with torch.no_grad():
        batched, _ = response_logprobs(model, build_lm_batch(tok, prompts, responses, 48))
        single = torch.cat([response_logprobs(model, build_lm_batch(tok, [p], [r], 48))[0] for p, r in zip(prompts, responses)])
        swapped, _ = response_logprobs(model, build_lm_batch(tok, prompts[::-1], responses[::-1], 48))
    torch.testing.assert_close(batched, single, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(batched, swapped.flip(0), rtol=1e-4, atol=1e-4)
    assert all(math.isfinite(v) and v < 0 for v in batched.tolist())
