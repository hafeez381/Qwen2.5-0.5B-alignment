import contextlib

import torch

from alignment.batching import build_lm_batch
from alignment.losses import sft_loss
from alignment.sft import window_backward
from tests.helpers import BigramLM
from tests.toy_lm import ToyTokenizer

TOK = ToyTokenizer()
PROMPTS = ["Q: ", "Task: ", "Do it: ", "Hi: "]
RESPONSES = ["a", "a much longer answer", "mid", "xy"]


def test_window_gradient_equals_one_big_batch():
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    small = BigramLM(TOK.vocab_size, seed=3)
    big = BigramLM(TOK.vocab_size, seed=3)
    micro = [build_lm_batch(TOK, PROMPTS[i:i + 2], RESPONSES[i:i + 2], 32, pad_to="longest") for i in (0, 2)]
    loss, n = window_backward(small, micro, scaler, contextlib.nullcontext())
    full = sft_loss(big, build_lm_batch(TOK, PROMPTS, RESPONSES, 32))
    full.backward()
    assert abs(loss - full.item()) < 1e-5
    torch.testing.assert_close(small.table.grad, big.table.grad, rtol=1e-5, atol=1e-7)


def test_per_microbatch_mean_would_differ():
    a = BigramLM(TOK.vocab_size, seed=3)
    micro = [build_lm_batch(TOK, PROMPTS[i:i + 2], RESPONSES[i:i + 2], 32, pad_to="longest") for i in (0, 2)]
    naive = sum(sft_loss(a, b) for b in micro) / 2
    full = sft_loss(a, build_lm_batch(TOK, PROMPTS, RESPONSES, 32))
    assert abs(naive.item() - full.item()) > 1e-3


def test_tiny_model_overfits_eight_rows():
    torch.manual_seed(0)
    model = BigramLM(TOK.vocab_size, seed=4)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    batch = build_lm_batch(TOK, ["Q: "] * 8, ["abcab"] * 8, 16)
    first = sft_loss(model, batch).item()
    for _ in range(50):
        opt.zero_grad()
        sft_loss(model, batch).backward()
        opt.step()
    assert sft_loss(model, batch).item() < 0.2 * first
