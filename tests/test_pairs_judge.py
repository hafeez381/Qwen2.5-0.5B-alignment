import ast
import importlib.util
import json

import pytest

from alignment.io import REPO_ROOT


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bp, jk, sj = load("build_pairs"), load("judge_pairwise_kaggle"), load("summarize_judgments")


def gens(generator, n=40, hash_="h1"):
    return [{"id": f"q{i}", "instruction": f"question {i}", "output": f"{generator} answer {i}", "generator": generator,
             "category": "c1" if i % 2 else "c2", "generation_config_hash": hash_} for i in range(n)]


def test_pairs_round_trip_order_is_mixed_and_blind():
    pairs, swapped, mapping = bp.build(gens("dpo"), gens("sft"), 42)
    a_share = sum(m["A"] == "dpo" for m in mapping.values()) / len(mapping)
    assert 0.3 < a_share < 0.7
    for p, s in zip(pairs, swapped):
        m = mapping[p["pair_id"]]
        side = {"A": p["response_a"], "B": p["response_b"]}
        assert side["A"].startswith(m["A"]) and side["B"].startswith(m["B"])
        assert (s["response_a"], s["response_b"]) == (p["response_b"], p["response_a"])
        assert set(p) == bp.PAIR_KEYS and "dpo" not in p["pair_id"] and "sft" not in p["pair_id"]


def test_pairs_refuse_mismatched_prompts_or_decoding():
    with pytest.raises(SystemExit):
        bp.build(gens("dpo"), gens("sft")[::-1], 42)
    with pytest.raises(SystemExit):
        bp.build(gens("dpo"), gens("sft", hash_="h2"), 42)


def test_rendered_prompt_hides_identities():
    pairs, _, _ = bp.build(gens("dpo", 5), gens("sft", 5), 42)
    for p in pairs:
        p["response_a"] = p["response_b"] = "neutral text"
    text = jk.render(pairs, "general")
    assert "Response A" in text and "Response B" in text
    assert not any(w in text.lower() for w in ("dpo", "sft", "lora", "qwen", "smollm"))


class FakeClient:
    def __init__(self, fail_first=0, always_fail=False):
        self.calls, self.fail_first, self.always_fail = 0, fail_first, always_fail

    def __call__(self, prompt):
        self.calls += 1
        if self.always_fail or self.calls <= self.fail_first:
            raise RuntimeError("proxy denied")
        ids = [line.split("pair_id=")[1] for line in prompt.splitlines() if line.startswith("### Item pair_id=")]
        return json.dumps({"verdicts": [{"pair_id": i, "winner": "A", "reason": "ok"} for i in ids]})


def write_pairs(tmp_path, n=12):
    pairs, _, _ = bp.build(gens("dpo", n), gens("sft", n), 42)
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in pairs))
    return path


def test_judge_retries_sleeps_resumes_and_logs_failures(tmp_path):
    pairs, out = write_pairs(tmp_path), tmp_path / "judged.jsonl"
    sleeps = []
    client = FakeClient(fail_first=2)
    jk.judge_file(pairs, out, client, "m", "general", batch_size=5, sleep_seconds=5, max_retries=3, sleep=sleeps.append)
    rows = jk.read_jsonl(out)
    assert len(rows) == 12 and rows[0]["attempts"] == 3 and all(s >= 5 for s in sleeps)
    calls = client.calls
    jk.judge_file(pairs, out, client, "m", "general", reuse=True, sleep=sleeps.append)
    assert client.calls == calls
    out2 = tmp_path / "judged2.jsonl"
    jk.judge_file(pairs, out2, FakeClient(always_fail=True), "m", "general", max_retries=3, sleep=sleeps.append)
    assert jk.read_jsonl(out2) == [] and len(jk.read_jsonl(tmp_path / "judged2_failures.jsonl")) == 12


def test_judge_script_cannot_generate():
    tree = ast.parse((REPO_ROOT / "scripts/judge_pairwise_kaggle.py").read_text())
    names = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.split(".")[0] in {"torch", "transformers", "alignment"} for m in names)


def test_parse_drops_malformed_verdicts():
    text = 'noise {"verdicts": [{"pair_id": "q1", "winner": "b"}, {"pair_id": "zz", "winner": "A"}, {"pair_id": "q2", "winner": "maybe"}]}'
    assert jk.parse(text, ["q1", "q2"]) == {"q1": {"winner": "B", "reason": ""}}


def test_summary_rates_and_swap_consistency():
    rows = [{"pair_id": "1", "category": "x", "verdict": "A", "outcome": "win", "swap_outcome": "win"},
            {"pair_id": "2", "category": "x", "verdict": "B", "outcome": "loss", "swap_outcome": "win"},
            {"pair_id": "3", "category": "y", "verdict": "tie", "outcome": "tie", "swap_outcome": "tie"},
            {"pair_id": "4", "category": "y", "verdict": "failure", "outcome": "failure"}]
    s = sj.summarize(rows)
    assert (s["win"], s["loss"], s["tie"], s["failure"]) == (1, 1, 1, 1)
    assert s["win_rate_with_ties"] == pytest.approx(0.5) and s["win_rate_without_ties"] == pytest.approx(0.5)
    assert s["a_position_rate"] == pytest.approx(0.5) and s["swap_consistency"] == pytest.approx(2 / 3)
    assert sj.outcome("A", {"A": "dpo", "B": "sft"}, "dpo") == "win"
    assert sj.outcome("A", {"A": "sft", "B": "dpo"}, "dpo") == "loss"


def test_gates_and_rubric_agreement():
    rows = [{"pair_id": str(i), "category": None, "verdict": "A", "outcome": "win", "swap_verdict": "B",
             "swap_outcome": "win"} for i in range(9)]
    rows.append({"pair_id": "9", "category": None, "verdict": "failure", "outcome": "failure", "attempts": None})
    rows[0]["attempts"], rows[1]["swap_attempts"] = 2, 3
    s = sj.summarize(rows)
    assert (s["failure_rate"], s["unknown_rate"], s["retried_pairs"], s["retried_pairs_swapped"]) == (0.1, 0.0, 1, 1)
    assert s["a_position_rate"] == 1.0 and s["a_position_rate_swapped"] == 0.0
    verdicts = {msg.split()[0]: ok for ok, msg in sj.gates(s)}
    assert verdicts == {"unknown": False, "swap": True, "a_position_rate": False, "a_position_rate_swapped": True}
    other = [{"pair_id": str(i), "outcome": "win" if i < 5 else "loss"} for i in range(10)]
    assert sj.agreement(rows, other) == pytest.approx(5 / 10)


def test_select_examples_covers_categories_and_prefers_swap_stable():
    se = load("select_examples")
    rows = [{"pair_id": f"{c}{i}", "category": c, "outcome": o, "swap_outcome": o if i % 2 else "tie"}
            for c in "xyz" for i, o in enumerate(["win", "win", "loss", "loss", "tie", "tie"])]
    picked = se.select(rows, k=6)
    assert len(picked) == 6 and {r["category"] for r in picked} == set("xyz")
    assert {r["outcome"] for r in picked} == {"win", "loss", "tie"}
    assert all(r["swap_outcome"] == r["outcome"] for r in picked)
    assert se.select(rows, k=6) == picked
