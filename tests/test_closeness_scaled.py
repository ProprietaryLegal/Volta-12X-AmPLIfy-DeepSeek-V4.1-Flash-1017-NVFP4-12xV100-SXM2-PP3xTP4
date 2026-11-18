"""closeness.scaled_word: the rule of line c2 (PLAN Amendment 12), on rows worked out by hand.
Rows are (tokens compared, first divergence, mean |dlogprob| before it or None)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("closeness", Path(__file__).resolve().parents[1] / "tools" / "closeness.py")
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)


def test_holds_at_exactly_half_and_twice():
    # base medians: first 100, mean 0.02; arm medians: first 50 (= half), mean 0.04 (= twice)
    base = [(200, 80, 0.01), (200, 100, 0.02), (200, 120, 0.03)]
    arm = [(200, 40, 0.03), (200, 50, 0.04), (200, 60, 0.05)]
    assert cl.scaled_word(arm, base) == ("HOLDS", 50, 0.04, 100, 0.02)


def test_worse_when_the_mean_is_over_twice():
    base = [(200, 80, 0.01), (200, 100, 0.02), (200, 120, 0.03)]
    arm = [(200, 80, 0.03), (200, 100, 0.0401), (200, 120, 0.05)]
    assert cl.scaled_word(arm, base)[0] == "WORSE"


def test_worse_when_the_divergence_comes_under_half_as_late():
    base = [(200, 80, 0.01), (200, 100, 0.02), (200, 120, 0.03)]
    arm = [(200, 40, 0.01), (200, 49, 0.02), (200, 60, 0.03)]
    assert cl.scaled_word(arm, base)[0] == "WORSE"


def test_cannot_decide_when_the_comparator_did_not_move():
    base = [(200, 200, 0.0), (200, 200, 0.0)]
    arm = [(200, 65, 0.04), (200, 124, 0.03)]
    assert cl.scaled_word(arm, base)[0] == "CANNOT DECIDE"


def test_cannot_decide_without_prompts_or_without_a_mean():
    assert cl.scaled_word([], [])[0] == "CANNOT DECIDE"
    # every divergence at token 0: no token before it, no mean
    assert cl.scaled_word([(200, 0, None)], [(200, 0, None)])[0] == "CANNOT DECIDE"


def test_window_o1_figures_as_printed():
    # o1, closeness.py's own lines: F3c against F2 (65, 0.0403) (124, 0.0333) (8, 0.0245); F2 against PL (8, 0.0184)
    # (60, 0.0702) (8, 0.0117). With the default as the base: medians 65 / 0.0333 against 8 / 0.0184: 65 >= 4 and
    # 0.0333 <= 0.0368, so HOLDS.
    arm = [(200, 65, 0.0403), (164, 124, 0.0333), (160, 8, 0.0245)]
    base = [(200, 8, 0.0184), (165, 60, 0.0702), (160, 8, 0.0117)]
    assert cl.scaled_word(arm, base) == ("HOLDS", 65, 0.0333, 8, 0.0184)
