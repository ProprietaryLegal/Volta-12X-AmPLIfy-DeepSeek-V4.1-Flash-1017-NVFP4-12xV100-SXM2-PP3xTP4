"""opt2_decide.decide against PLAN Amendment 12's sentences, case by case (ceiling 31,698)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("opt2_decide", Path(__file__).resolve().parents[1] / "tools" / "opt2_decide.py")
od = importlib.util.module_from_spec(spec)
spec.loader.exec_module(od)
C = 31698


def pick(*a):
    return od.decide(*a, C)[0]


def test_neither_passed_or_over_the_ceiling_is_two_slots():
    assert pick(None, None, None, None) == "F2"
    assert pick(31699, None, 3000, None) == "F2"      # passed its lines but one MiB over the ceiling
    assert pick(31846, 31932, 3400, 3400) == "F2"


def test_one_alone():
    assert pick(30600, None, 3000, None) == "F3c"
    assert pick(None, 31698, None, 3100) == "F3o"     # exactly at the ceiling qualifies
    assert pick(30600, 31700, 3000, 3100) == "F3c"    # F3o passed but is over the ceiling


def test_both_more_than_100_apart_the_lower_wins_whatever_the_read():
    assert pick(30600, 31200, 2000, 3400) == "F3c"
    assert pick(31300, 31199, 3400, 2000) == "F3o"


def test_both_within_100_the_faster_read_and_equal_reads_go_to_f3o():
    assert pick(31200, 31300, 3401, 3400) == "F3c"    # exactly 100 apart counts as within
    assert pick(31200, 31300, 3400, 3400) == "F3o"
    assert pick(31250, 31200, 3000, 3400) == "F3o"


def test_both_within_100_without_a_read_speed_cannot_decide():
    assert od.decide(31200, 31250, None, 3400, C) == (
        "F2", "both qualify within 100 MiB but a prompt-read speed could not be read: COULD NOT DECIDE between them; two slots stays")
