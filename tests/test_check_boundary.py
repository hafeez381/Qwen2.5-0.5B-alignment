import importlib.util

from alignment.io import REPO_ROOT

spec = importlib.util.spec_from_file_location("check_boundary", REPO_ROOT / "scripts/check_boundary.py")
check_boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_boundary)


def test_flags_trl_trainer_and_builtin_loss(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from trl import DPOTrainer\nimport peft\nfrom transformers import Trainer\n"
        "out = policy_model(input_ids=x, labels=y)\n"
    )
    found = " | ".join(check_boundary.violations(bad))
    for expected in ("from trl", "import peft", "import Trainer", "labels=..."):
        assert expected in found, found


def test_allows_plain_transformers_and_own_labels_argument(tmp_path):
    ok = tmp_path / "ok.py"
    ok.write_text(
        "from transformers import AutoModelForCausalLM\n"
        "logits = model(input_ids=x, attention_mask=m).logits\n"
        "lp = response_logprobs(model, x, m, labels=y)\n"
    )
    assert check_boundary.violations(ok) == []


def test_repo_passes():
    assert check_boundary.main([]) == 0
