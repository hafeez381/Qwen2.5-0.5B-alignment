import importlib.util

from alignment.io import REPO_ROOT

spec = importlib.util.spec_from_file_location("check_generations", REPO_ROOT / "scripts/check_generations.py")
cg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cg)


def test_loops_of_any_period_are_found():
    assert cg.has_loop("a b c d a b c d a b c d end")
    assert cg.has_loop("Ingredients: " + "- 1/2 cup brown sugar " * 6)
    sentence = "The onboarding process is designed to ensure that the new employee is fully equipped. "
    assert cg.has_loop("Intro. " + sentence * 3)
    stanza = "Their approach was simple and clear, / A way to empower teams to thrive, / With AI, they could create, / A world where Agile coaching was possible. / "
    assert cg.has_loop(stanza * 3)


def test_ordinary_text_is_not_a_loop():
    assert not cg.has_loop("a b c d a b c d end a b c d")
    assert not cg.has_loop("short answer")
    assert not cg.has_loop("Preheat the oven. Mix the flour and sugar. Add the eggs and the butter. Bake for 12 minutes.")
