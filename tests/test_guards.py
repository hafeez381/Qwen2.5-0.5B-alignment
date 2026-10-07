import pytest

from alignment import guards
from alignment.guards import GuardError
from alignment.io import sha256_file


def test_fixed_files_catch_a_flipped_byte(tmp_path):
    f = tmp_path / "eval.json"
    f.write_bytes(b'{"a": 1}')
    entries = {"eval.json": sha256_file(f)}
    guards.assert_fixed_files("full", root=tmp_path, entries=entries)
    f.write_bytes(b'{"a": 2}')
    with pytest.raises(GuardError, match="changed"):
        guards.assert_fixed_files("full", root=tmp_path, entries=entries)


def test_fixed_files_missing_is_fatal_only_in_full(tmp_path):
    entries = {"gone.json": "0" * 64}
    assert guards.assert_fixed_files("smoke", root=tmp_path, entries=entries) == ["gone.json"]
    with pytest.raises(GuardError, match="missing"):
        guards.assert_fixed_files("full", root=tmp_path, entries=entries)


def test_eval_overlap_ignores_case_and_whitespace():
    guards.assert_no_eval_overlap(["How do I bake bread?"], ["How do I bake a cake?"])
    with pytest.raises(GuardError):
        guards.assert_no_eval_overlap(["How do  I bake\nbread? "], ["how do i bake bread?"])
