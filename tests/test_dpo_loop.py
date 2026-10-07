import contextlib
import copy
import math

import pytest
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.batching import build_dpo_batch
from alignment.dpo import aggregate, assert_reference_frozen, param_checksum, validate
from alignment.guards import GuardError, assert_same_init
from alignment.io import REPO_ROOT, sha256_file
from alignment.losses import dpo_loss, dpo_metrics, pair_logprobs
from tests.helpers import BigramLM
from tests.toy_lm import ToyTokenizer

ROWS = [{"id": f"p{i}", "prompt": p, "chosen": c, "rejected": r} for i, (p, c, r) in enumerate([
    ("How do I boil an egg?", "Simmer it for nine minutes.", "No idea."),
    ("Name a colour.", "Blue.", "I refuse to answer that."),
    ("What is 2+2?", "4", "5"),
])]


def test_step_zero_policy_equals_reference_on_a_real_transformer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    policy = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
    ref = copy.deepcopy(policy).eval().requires_grad_(False)
    cfg = {"eval_batch": 2, "max_length": 64, "beta": 0.1}
    agg = aggregate(validate(policy, ref, tok, ROWS, cfg, contextlib.nullcontext(), {}))
    assert agg["val_accuracy"] == 0.0
    assert abs(agg["val_loss"] - math.log(2)) < 1e-6 and agg["val_margin"] == 0.0


def test_toy_dpo_overfits_and_the_reference_never_moves():
    tok = ToyTokenizer()
    policy = BigramLM(tok.vocab_size, seed=5)
    ref = copy.deepcopy(policy).requires_grad_(False)
    opt = torch.optim.SGD(policy.parameters(), lr=1e-2)
    assert_reference_frozen(ref, opt)
    before = param_checksum(ref)
    batch = build_dpo_batch(tok, ["Q: ", "Ask: ", "Do: ", "Hi: "], ["good", "yes", "fine", "ok"],
                            ["bad", "no", "awful", "nope"], 16, pad_to="longest")
    for _ in range(30):
        opt.zero_grad()
        dpo_loss(policy, ref, batch, beta=0.5).backward()
        opt.step()
    m = dpo_metrics(*pair_logprobs(policy, ref, batch), 0.5)
    assert m["correct"].all() and (m["margin"] > 0).all()
    assert_reference_frozen(ref, opt, before)


def test_reference_in_optimizer_or_changed_is_caught():
    policy, ref = BigramLM(10, 1), BigramLM(10, 2).requires_grad_(False)
    with pytest.raises(GuardError):
        assert_reference_frozen(ref, torch.optim.SGD(list(policy.parameters()) + list(ref.parameters()), lr=0.1))
    before = param_checksum(ref)
    with torch.no_grad():
        ref.table += 1e-3
    with pytest.raises(GuardError, match="changed"):
        assert_reference_frozen(ref, torch.optim.SGD(policy.parameters(), lr=0.1), before)


def test_same_init_guard(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    assert_same_init(tmp_path, sha256_file(tmp_path / "model.safetensors"))
    with pytest.raises(GuardError):
        assert_same_init(tmp_path, "0" * 64)


def test_hh_and_uf_configs_differ_only_in_data_and_output_paths():
    hh = yaml.safe_load((REPO_ROOT / "configs/dpo_hh.yaml").read_text())
    uf = yaml.safe_load((REPO_ROOT / "configs/dpo_uf.yaml").read_text())
    assert hh.keys() == uf.keys()
    assert {k for k in hh if hh[k] != uf[k]} == {"train", "val", "output", "results"}
