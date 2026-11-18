"""tools/oom_lines.py on REAL lines: the twelve lines window k1's limited boot logged and the old counter took for
out-of-memory errors (tests/data/card-limit-lines-k1.txt, from the build's run/state/P3k1-S.log, 2026-11-17), and one real
out-of-memory error of the eight-card build (tests/data/oom-line-W5-piqx.txt, its W5-piqx.log, 2026-11-10). The owner's
acceptance for the fix: 0 on k1's log, 1 on the same log with one real out-of-memory line appended; the whole log
is checked the same way outside the repository (it stays out of it)."""
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("oom_lines", ROOT / "tools" / "oom_lines.py")
oom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oom)
LIMIT = (ROOT / "tests" / "data" / "card-limit-lines-k1.txt").read_text()
REAL = (ROOT / "tests" / "data" / "oom-line-W5-piqx.txt").read_text()


def test_the_fixtures_are_what_they_are_said_to_be():
    assert len(LIMIT.splitlines()) == 12 and all("card_limit.py" in line for line in LIMIT.splitlines())
    assert len(REAL.splitlines()) == 1 and "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate" in REAL
    # the old pattern is what went wrong: it matches all thirteen
    old = re.compile("OutOfMemoryError|CUDA out of memory")
    assert sum(1 for line in (LIMIT + REAL).splitlines() if old.search(line)) == 13


def test_the_limits_announcement_is_no_out_of_memory_error():
    assert oom.count(LIMIT) == 0


def test_a_real_error_counts_also_among_the_announcements():
    assert oom.count(REAL) == 1
    assert oom.count(LIMIT + REAL) == 1
    assert oom.count(LIMIT + REAL + REAL) == 2
    assert oom.count("") == 0
