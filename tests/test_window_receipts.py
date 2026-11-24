"""tools/mhc_window_receipt.py and tools/control_ahead_window_receipt.py on real log lines.

tests/data/kit11-markers-aw7.txt holds every line of window aw7's qualification boot (kit 11, twelve workers) that
either counter reads: the experts' sizing line each worker prints at boot (two per worker), the mHC kernel's
first-dispatch line and the control header's first-transfer line, copied from the server log unchanged and sorted by
time. Expected counts come from the build's layout, three stages of four ranks, not from the tools' output.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "kit11-markers-aw7.txt"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MHC = load("mhc_window_receipt")
CTL = load("control_ahead_window_receipt")
LINES = DATA.read_text().splitlines()
RANKS = 3 * 4


def write(tmp_path, lines):
    log = tmp_path / "boot.log"
    log.write_text("\n".join(lines) + "\n")
    return log


def test_real_lines_hold_one_marker_of_each_kind_per_rank():
    assert sum(MHC.MARKER in line for line in LINES) == RANKS
    assert sum(CTL.MARKER in line for line in LINES) == RANKS


def test_counts_on_the_real_boot(tmp_path):
    log = write(tmp_path, LINES)
    assert MHC.counts(log) == (RANKS, RANKS)
    assert CTL.counts(log) == (RANKS, RANKS)


def test_receipts_and_exit_codes(tmp_path):
    log = write(tmp_path, LINES)
    assert MHC.main(log, tmp_path / "mhc.tsv") == 0
    assert (tmp_path / "mhc.tsv").read_text() == f"mhc-workers\t{RANKS}\nmhc-dispatch\t{RANKS}\n"
    assert CTL.main(log, tmp_path / "ctl.tsv") == 0
    assert (tmp_path / "ctl.tsv").read_text() == f"control-ahead-workers\t{RANKS}\ncontrol-ahead-transfers\t{RANKS}\n"
    # A receipt is written once: a second run on the same receipt path is INCONCLUSIVE, never an overwrite.
    assert MHC.main(log, tmp_path / "mhc.tsv") == 34


def test_one_missing_transfer_fails_the_line(tmp_path):
    last = [line for line in LINES if CTL.MARKER in line][-1]
    log = write(tmp_path, [line for line in LINES if line != last])
    assert CTL.main(log, tmp_path / "ctl.tsv") == 33
    assert (tmp_path / "ctl.tsv").read_text() == f"control-ahead-workers\t{RANKS}\ncontrol-ahead-transfers\t{RANKS - 1}\n"


def test_one_missing_dispatch_is_counted(tmp_path):
    last = [line for line in LINES if MHC.MARKER in line][-1]
    log = write(tmp_path, [line for line in LINES if line != last])
    assert MHC.counts(log) == (RANKS, RANKS - 1)


@pytest.mark.parametrize("tool", [MHC, CTL])
def test_a_marker_from_another_process_is_refused(tmp_path, tool):
    markers = [line for line in LINES if tool.MARKER in line]
    first_pid = markers[0].split("pid=", 1)[1].split(")", 1)[0]
    last_pid = markers[-1].split("pid=", 1)[1].split(")", 1)[0]
    assert first_pid != last_pid
    moved = markers[-1].replace(f"pid={last_pid})", f"pid={first_pid})")
    log = write(tmp_path, [moved if line == markers[-1] else line for line in LINES])
    with pytest.raises(ValueError, match="PID differs"):
        tool.counts(log)


@pytest.mark.parametrize("tool", [MHC, CTL])
def test_a_repeated_marker_is_refused(tmp_path, tool):
    marker = [line for line in LINES if tool.MARKER in line][0]
    log = write(tmp_path, LINES + [marker])
    with pytest.raises(ValueError, match="repeated"):
        tool.counts(log)


@pytest.mark.parametrize("tool", [MHC, CTL])
def test_a_boot_without_every_rank_is_refused(tmp_path, tool):
    lines = [line for line in LINES if not line.startswith("(Worker_PP2_TP3 ")]
    with pytest.raises(ValueError, match="twelve distinct ranks"):
        tool.counts(write(tmp_path, lines))
