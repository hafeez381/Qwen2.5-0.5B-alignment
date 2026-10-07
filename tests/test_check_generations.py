import importlib.util

from alignment.io import REPO_ROOT

spec = importlib.util.spec_from_file_location("check_generations", REPO_ROOT / "scripts/check_generations.py")
cg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cg)


def test_loop_detection():
    assert cg.has_loop("a b c d a b c d a b c d end")
    assert not cg.has_loop("a b c d a b c d end a b c d")
    assert not cg.has_loop("short answer")
