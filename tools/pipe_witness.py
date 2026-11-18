#!/usr/bin/env python3
"""pipe_witness.py LOG --mid {0|1} --hold N [--turn N] [--after-read]: arm P's witness that kit 8's three switches are
in effect, or not, on the boot whose server log is LOG (run_window_pp3.sh, phase armP).

The switches are read once, at start, by the processes that act on them, and each of those processes says so in the
server log (a titled worker's /proc environment is blank: arm G's corrected witness, PLAN Amendment 29; so the log's
own lines are the evidence, as there):

  VLLM_V41_PP_MID_PREFILL_SKIP=1   every worker of the stage between the first and the last logs, at construction
      (READ models/deepseek_v41/model.py, _init_mid_prefill_skip):
        (Worker_PP1_TP<t> pid=N) INFO ... VLLM_V41_PP_MID_PREFILL_SKIP=1 active (stage 1 of 3, TP rank <t>): ...
      --mid 1: exactly one such line from each of Worker_PP1_TP0..3, its TP rank the worker's, and none from a
      worker of another stage. --mid 0: no line that names the switch at all.
  VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD=N   the engine core (the scheduler) logs once
      (READ v1/core/sched/v41_multi_seq.py, V41SchedulerMultiSeq.__init__):
        (EngineCore pid=N) INFO ... VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD=<N> active (scheduler): ...
      --hold N (N > 0): exactly one such line, from the engine core, with that N. --hold 0: no line that names it.
  VLLM_V41_PP_TURN_CHUNKS=N   the engine core (the scheduler) logs once
      (READ v1/core/sched/v41_multi_seq.py, V41SchedulerMultiSeq.__init__):
        (EngineCore pid=N) INFO ... VLLM_V41_PP_TURN_CHUNKS=<N> active (scheduler): ...
      and each worker of the stage between the first and the last logs once, at its first prompt-sized hand-off
      (READ v1/worker/gpu_worker.py, Worker._pp_send_overlaps: the stage awaits its hand-off before its next step):
        (Worker_PP1_TP<t> pid=N) INFO ... VLLM_V41_PP_TURN_CHUNKS=<N> active (worker): ...
      --turn N (N > 0): exactly one scheduler line, from the engine core, with that N; with --after-read also
      exactly one worker line with that N from each of Worker_PP1_TP0..3 and none from a worker of another stage
      (the first stage keeps its overlap; the last stage sends nothing). --turn 0 (the default): no line that names
      the switch.
  --after-read (after a prompt read with skipped chunks): the first skipped step's line
        (Worker_PP<s>_TP0 pid=N) INFO ... VLLM_V41_PREFILL_SKIP_ROWS: first skipped prefill step (layers > 20).
      (model.py, prefill_skip_bind: TP rank 0 of a stage that skips, once) must be there from Worker_PP2_TP0 (the
      last stage, as on every boot) and, with --mid 1, from Worker_PP1_TP0; with --mid 0 never from Worker_PP1_TP0.

  FAIL          a line that must be there is missing or doubled, a line that must not be there is there, a rank or a
                count does not match
  INCONCLUSIVE  no FAIL, and the log cannot say: it cannot be read, it holds no line of the engine core or of a
                middle-stage worker at all (not this kind of boot's log), or (--after-read) no stage logged a first
                skipped step (no prompt with skipped chunks was read on this boot)
  PASS          otherwise
Exits 0 PASS, 1 FAIL, 2 INCONCLUSIVE. Pure reading; never observed on a card before arm P: the two active lines.
"""
import argparse
import re
import sys

PREFIX = re.compile(r"\((?P<proc>[A-Za-z_0-9]+) pid=\d+\)")
MID_KNOB = "VLLM_V41_PP_MID_PREFILL_SKIP"
HOLD_KNOB = "VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD"
MID_ACTIVE = re.compile(MID_KNOB + r"=1 active \(stage (?P<stage>\d+) of (?P<stages>\d+), TP rank (?P<rank>\d+)\): ")
HOLD_ACTIVE = re.compile(HOLD_KNOB + r"=(?P<n>\d+) active \(scheduler\): ")
TURN_KNOB = "VLLM_V41_PP_TURN_CHUNKS"
TURN_ACTIVE = re.compile(TURN_KNOB + r"=(?P<n>\d+) active \((?P<who>scheduler|worker)\): ")
FIRST_SKIP = "VLLM_V41_PREFILL_SKIP_ROWS: first skipped prefill step (layers > 20)."
MIDDLE = tuple(f"Worker_PP1_TP{t}" for t in range(4))


def read(lines, mid, hold, after_read, turn=0):
    """(fails, unreadable, notes) from the server log's lines (pure)."""
    fails, unread = [], []
    procs = set()
    mid_lines, mid_other, hold_lines, hold_other, skips = {}, [], [], [], {}
    turn_sched, turn_work, turn_other = [], {}, []
    def writer(line, at):
        """The process whose line holds position ``at``: the last "(Name pid=N)" before it. A worker's line can
        follow another's unfinished one on the same physical line (MEASURED on run/state/P3st4-KD1.log: a graph
        capture's progress bar, then "(Worker_PP2_TP1 pid=...) INFO ..."), so the line's start does not say."""
        found = [m["proc"] for m in PREFIX.finditer(line) if m.end() <= at]
        return found[-1] if found else None

    for line in lines:
        procs.update(m["proc"] for m in PREFIX.finditer(line))
        if MID_KNOB in line:
            a = MID_ACTIVE.search(line)
            proc = writer(line, a.start()) if a else None
            if proc:
                mid_lines.setdefault(proc, []).append((int(a["stage"]), int(a["stages"]), int(a["rank"])))
            else:
                mid_other.append(line.strip()[:160])
        if HOLD_KNOB in line:
            a = HOLD_ACTIVE.search(line)
            proc = writer(line, a.start()) if a else None
            if proc:
                hold_lines.append((proc, int(a["n"])))
            else:
                hold_other.append(line.strip()[:160])
        if TURN_KNOB in line:
            a = TURN_ACTIVE.search(line)
            proc = writer(line, a.start()) if a else None
            if proc and a["who"] == "scheduler":
                turn_sched.append((proc, int(a["n"])))
            elif proc:
                turn_work.setdefault(proc, []).append(int(a["n"]))
            else:
                turn_other.append(line.strip()[:160])
        if FIRST_SKIP in line:
            proc = writer(line, line.index(FIRST_SKIP))
            if proc:
                skips[proc] = skips.get(proc, 0) + 1
    if "EngineCore" not in procs:
        unread.append("no line of the engine core")
    if not any(p in procs for p in MIDDLE):
        unread.append("no line of a middle-stage worker (Worker_PP1_TP0..3)")
    # the middle-stage skip
    if mid:
        for t, name in enumerate(MIDDLE):
            got = mid_lines.get(name, [])
            if got != [(1, 3, t)]:
                fails.append(f"{name}: {len(got)} active line(s) {got}, want one for stage 1 of 3, TP rank {t}")
        for name in sorted(set(mid_lines) - set(MIDDLE)):
            fails.append(f"{name}: an active line of {MID_KNOB} from a worker that is not of the middle stage")
    elif mid_lines or mid_other:
        fails.append(f"{sum(map(len, mid_lines.values())) + len(mid_other)} line(s) name {MID_KNOB} on a boot without it")
    if mid and mid_other:
        fails.append(f"a line names {MID_KNOB} and is not the active line: {mid_other[0]}")
    # the prompt-end hold
    if hold:
        if hold_lines != [("EngineCore", hold)]:
            fails.append(f"active lines of {HOLD_KNOB}: {hold_lines}, want one from the engine core with {hold}")
    elif hold_lines or hold_other:
        fails.append(f"{len(hold_lines) + len(hold_other)} line(s) name {HOLD_KNOB} on a boot without it")
    if hold and hold_other:
        fails.append(f"a line names {HOLD_KNOB} and is not the active line: {hold_other[0]}")
    # the turn chunks
    if turn:
        if turn_sched != [("EngineCore", turn)]:
            fails.append(f"scheduler lines of {TURN_KNOB}: {turn_sched}, want one from the engine core with {turn}")
        for name in sorted(set(turn_work) - set(MIDDLE)):
            fails.append(f"{name}: a worker line of {TURN_KNOB} from a worker that is not of the middle stage")
        for name in MIDDLE:
            got = turn_work.get(name, [])
            if got not in ([], [turn]) or (after_read and got != [turn]):
                fails.append(f"{name}: worker line(s) of {TURN_KNOB} {got}, want {'one' if after_read else 'none or one'} with {turn}")
        if turn_other:
            fails.append(f"a line names {TURN_KNOB} and is not an active line: {turn_other[0]}")
    elif turn_sched or turn_work or turn_other:
        fails.append(f"{len(turn_sched) + sum(map(len, turn_work.values())) + len(turn_other)} line(s) name {TURN_KNOB} on a boot without it")
    # the first skipped step, after a read
    if after_read:
        last = skips.get("Worker_PP2_TP0", 0)
        want = 1 if mid else 0
        if last == 0 and not skips:
            # the last stage skips on every boot: without its line no prompt with skipped chunks was read yet
            unread.append("no first skipped step from the last stage: no prompt with skipped chunks was read on this boot")
        else:
            if last != 1:
                fails.append(f"Worker_PP2_TP0 logged the first skipped step {last} time(s), want 1")
            if skips.get("Worker_PP1_TP0", 0) != want:
                fails.append(f"Worker_PP1_TP0 logged the first skipped step {skips.get('Worker_PP1_TP0', 0)} time(s), want {want}")
        for name in sorted(set(skips) - {"Worker_PP1_TP0", "Worker_PP2_TP0"}):
            fails.append(f"{name} logged a first skipped step: only TP rank 0 of a stage that skips does")
    notes = [f"active lines: {MID_KNOB} from {sorted(mid_lines) or 'no worker'}; {HOLD_KNOB} {hold_lines or 'none'}; "
             f"{TURN_KNOB} scheduler {turn_sched or 'none'}, workers {sorted(turn_work) or 'none'}; "
             f"first skipped step from {sorted(skips) or 'no worker'}; processes with lines: {len(procs)}"]
    return fails, unread, notes


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--mid", type=int, choices=(0, 1), required=True)
    ap.add_argument("--hold", type=int, required=True)
    ap.add_argument("--turn", type=int, default=0)
    ap.add_argument("--after-read", action="store_true")
    a = ap.parse_args(argv)
    if a.hold < 0:
        ap.error("--hold is 0 or the switch's number of decode steps")
    if not 0 <= a.turn <= 8:
        ap.error("--turn is 0 or the switch's number of chunks (1 to 8)")
    try:
        with open(a.log, encoding="utf-8", errors="replace") as f:
            fails, unread, notes = read(f, a.mid, a.hold, a.after_read, a.turn)
    except OSError as e:
        fails, unread, notes = [], [f"{a.log} cannot be read: {e.strerror}"], []
    verdict = "FAIL" if fails else "INCONCLUSIVE" if unread else "PASS"
    for line in notes + [f"FAIL {x}" for x in fails] + [f"UNREADABLE {x}" for x in unread]:
        print("  " + line)
    print(f"PIPE WITNESS mid={a.mid} hold={a.hold} turn={a.turn}{' after a read' if a.after_read else ''}: {verdict}")
    return {"PASS": 0, "FAIL": 1}.get(verdict, 2)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
