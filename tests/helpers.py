from types import SimpleNamespace

import torch
from torch import nn

from tests.toy_lm import ToyTokenizer


class BigramLM(nn.Module):
    """Logits at position t depend on the token at t, so a wrong causal shift changes the answer."""

    def __init__(self, vocab_size, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.table = nn.Parameter(torch.randn(vocab_size, vocab_size, generator=g))

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.table[input_ids])


class PadIsEosTokenizer(ToyTokenizer):
    pad_token = ToyTokenizer.eos_token
    pad_token_id = ToyTokenizer.eos_token_id


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None
