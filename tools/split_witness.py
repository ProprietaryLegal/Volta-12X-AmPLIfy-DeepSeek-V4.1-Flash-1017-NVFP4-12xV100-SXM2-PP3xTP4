#!/usr/bin/env python3
"""split_witness.py {log LOG | procs PGID | both LOG PGID} WANT: arm G's witness that the dense multiply library's
switch TM_GEMM_TUNE=WANT (max_splits=1) reached the arm's serving workers (owner's rule, 2026-11-17; corrected after
window g1, 2026-11-17 19:24Z, where the first version's environment clause failed on all twelve workers).

WHY NOT /proc/<pid>/environ (the first version's per-worker clause): a worker sets its title with setproctitle
(vllm/utils/system_utils.py:184-198, called from v1/executor/multiproc_executor.py:1037/1065), and setproctitle takes
the memory of argv AND of the environment block for the title; /proc/<pid>/environ shows that block. MEASURED on
a CPU machine 2026-11-17 with the engine's own setproctitle 1.3.8 (cpython-312 .so from the build's venv), a spawned child
with TM_GEMM_TUNE in its environment: /proc/<pid>/environ held 6 variables (TM_GEMM_TUNE among them) before
setproctitle("VLLM::Worker_PP0_TP0") and 0 after (184 bytes of zeros), while os.environ and libc getenv() in the
same process still returned max_splits=1. So /proc/<pid>/environ of a titled vLLM process is empty by construction:
window g1's "unset" on twelve of twelve was this, not a missing switch. Nothing in the engine removes TM_GEMM_* (READ:
sm70_native.py is the only Python file that names it; no os.environ.pop/del of it anywhere in the package).

WHY NOT THE LIBRARY'S TRACE (TM_GEMM_TRACE): unchanged from the first version (unprefixed, interleaved, a host cost
on every eager dispatch, a limit per process).

WHAT IS READ (all in the arm's own server log; nothing per process is read but the titles):
  1 the value every worker receives: every 'TM_GEMM_TUNE' field of the log (the "SM70 acceleration status" lines of
    the API server, the engine core and the last stage's four workers, and the engine core's "Initializing a V1 LLM
    engine ... with config:" line) must be WANT, with at least one line of the engine core. These are the raw inputs
    the parent captured from its environment (vllm/config/sm70_native.py capture_inputs) and sends to every worker in
    the configuration; each worker's NativeBindings (vllm/_sm70/policy.py:196-240) turns them into the prepared policy
    token that every TurboMind call carries (csrc/sm70_policy.h PolicyScope, with_policy).
  2 the library's own parse: TurboMind builds one Gemm per (card, policy) in each worker
    (csrc/sm70_turbomind/ops/gemm_runtime.cpp:719-750 get_gemm), and its constructor (kernels/gemm/gemm.cu:536-558)
    parses TM_GEMM_TUNE from the active policy (or, for a call without a token, from the process's getenv: the same
    value, MEASURED above) with ParseTuningParams, which PRINTS "max_splits <v>" to stdout (tuner/params.cc:18).
    Window g1: 16 such lines; windows s4, sv1, mv1 (the switch off): 0. Every one must read the value WANT names;
    at least 12 (one per worker: every stage runs TurboMind GEMMs) must be present.
  3 no line "Failed to parse `TM_GEMM_TUNE`" (gemm.cu: on a failure the library uses its defaults).
  REPORT ONLY (not judged): the parse lines are unprefixed (C++ stdout), so they name no worker; each worker's
    one-time stderr line "V41 sum2 generic push active ... rank=<t>" names its TP rank, and in window g1 every one of
    the twelve such lines directly follows one or two parse lines (three runs over ranks 0-3). The pairing is printed
    as a hint; it rests on write order, not on a name.
  procs: the twelve processes titled "VLLM::Worker_PP<p>_TP<t>" in the arm's process group, each exactly once (that
    the twelve workers live); their environment is NOT read (see above).
  FAIL          a field other than WANT, a parse line with another value, or a parse-failure line
  INCONCLUSIVE  no FAIL, and: no engine-core line with the field, a status line without it, fewer than 12 parse lines
                (a worker's stdout not yet flushed cannot be told from a worker that never parsed), or not the twelve
                titled workers
  PASS          otherwise. WHAT A PASS SHOWS: the value every worker received, and at least twelve library parses of
                that value in a twelve-worker boot. IT DOES NOT NAME THE STAGE of any parse: no line of the engine
                says per worker which split count its library uses (see the engine-side line proposed in the arm-g
                notes).
Exits 0 PASS, 1 FAIL, 2 INCONCLUSIVE."""
import os
import re
import sys

STATUS = "SM70 acceleration status: "
PREFIX = re.compile(r"^\((?P<proc>[A-Za-z_0-9]+) pid=\d+\)")
FIELD = re.compile(r"""'TM_GEMM_TUNE': (?P<q>['"])(?P<v>.*?)(?P=q)""")
PARSE_FAIL = "Failed to parse `TM_GEMM_TUNE`"
PARSED = re.compile(r"^(?P<key>max_splits|max_waves|swizzle|top_k|clusters|min_iter|max_iter|max_time) (?P<v>\S+)\s*$")
SUM2 = re.compile(r"^V41 sum2 generic push active bytes=\d+ blocks=\d+ rank=(?P<r>\d+)\s*$")
MIN_PARSES = 12
TITLE = re.compile(r"^VLLM::(?P<name>Worker_PP[0-2]_TP[0-3])$")
WORKERS = tuple(f"Worker_PP{p}_TP{t}" for p in range(3) for t in range(4))
UNSET = r"\x1f"   # how repr() prints the engine's unset marker inside a status line


def want_pairs(want):
    """{'max_splits': '1'} from 'max_splits=1' (the TM_GEMM_TUNE syntax: comma-separated key=value)."""
    out = {}
    for part in want.split(","):
        k, sep, v = part.partition("=")
        if not sep or not k.strip():
            raise ValueError(f"WANT '{want}' is not key=value[,key=value]")
        out[k.strip()] = v.strip()
    return out


def log_lines(lines, want):
    """(fails, unreadable, notes) from the server log's lines (pure)."""
    fails, unread, per = [], [], {}
    nparse = 0
    pairs = want_pairs(want)
    parsed, other, run, paired = [], [], 0, []
    for line in lines:
        raw = line.rstrip("\n")
        pm = PARSED.match(raw)
        if pm:
            if pm["key"] in pairs:
                (parsed if pm["v"] == pairs[pm["key"]] else other).append(raw)
                run += 1
            continue
        sm = SUM2.match(raw)
        if sm:
            paired.append((sm["r"], run))
            run = 0
        elif raw and not raw.startswith("("):
            pass                                # other unprefixed lines (graph capture bars) do not end a run
        if PARSE_FAIL in line:
            nparse += 1
        vals = [f["v"] for f in FIELD.finditer(line)]
        if not vals and STATUS not in line:
            continue
        m = PREFIX.match(line)
        proc = m["proc"] if m else "(no prefix)"
        if not vals:
            unread.append(f"a status line of {proc} holds no TM_GEMM_TUNE field")
            continue
        per.setdefault(proc, []).extend(vals)
    for proc, vals in sorted(per.items()):
        bad = sorted({("unset" if v == UNSET else repr(v)) for v in vals if v != want})
        if bad:
            fails.append(f"{proc}: TM_GEMM_TUNE {', '.join(bad)} in its status line(s), want {want!r}")
    if not any(p == "EngineCore" for p in per):
        unread.append("no status line of the engine core (the parent whose capture every worker receives)")
    if nparse:
        fails.append(f"{nparse} line(s) '{PARSE_FAIL}': the library fell back to its defaults")
    if other:
        fails.append(f"{len(other)} library parse line(s) with another value: {sorted(set(other))[:3]}")
    elif len(parsed) < MIN_PARSES:
        unread.append(f"{len(parsed)} library parse line(s) '{' '.join(next(iter(pairs.items())))}', want at least "
                      f"{MIN_PARSES} (one per worker; an unflushed stdout cannot be told from a worker that never parsed)")
    hint = ", ".join(f"rank {r}: {k}" for r, k in paired)
    notes = [f"lines with the field: {', '.join(f'{p} x{len(v)}' for p, v in sorted(per.items())) or 'none'}; "
             f"parse-failure lines {nparse}",
             f"library parse lines matching {want}: {len(parsed)}; other values: {len(other)}",
             f"REPORT ONLY, by write order: parse lines directly before each 'V41 sum2 ... rank=' line: {hint or 'no sum2 line'}"
             f" ({sum(1 for _, k in paired if k > 0)} of {len(paired)} sum2 lines follow a parse line)"]
    return fails, unread, notes


def procs(pgid, want, proc_root="/proc"):
    """(fails, unreadable, notes): the twelve titled workers of group PGID, each exactly once. Their environment is not
    read: setproctitle empties /proc/<pid>/environ (module docstring)."""
    unread, seen = [], {}
    for pid in os.listdir(proc_root):
        if not pid.isdigit():
            continue
        try:
            stat = open(f"{proc_root}/{pid}/stat", "rb").read().decode(errors="replace")
            if int(stat[stat.rindex(")") + 2:].split()[2]) != pgid:
                continue
            title = open(f"{proc_root}/{pid}/cmdline", "rb").read().split(b"\0")[0].decode(errors="replace").strip()
        except (OSError, ValueError, IndexError):
            continue   # a process that ended or is not ours to read; the twelve are counted below
        m = TITLE.match(title)
        if m:
            seen.setdefault(m["name"], []).append(pid)
    for name in WORKERS:
        if len(seen.get(name, [])) != 1:
            unread.append(f"{name}: {len(seen.get(name, []))} processes in group {pgid}, want 1")
    found = sum(len(seen.get(n, [])) == 1 for n in WORKERS)
    return [], unread, [f"workers titled in group {pgid}: {found} of 12 (their environment is not read)"]


def main(argv):
    if len(argv) < 3 or argv[0] not in ("log", "procs", "both") or (argv[0] == "both") != (len(argv) == 4):
        print(__doc__.split("\n")[0])
        return 2
    mode, want = argv[0], argv[-1]
    fails, unread, notes = [], [], []
    if mode in ("log", "both"):
        try:
            with open(argv[1], encoding="utf-8", errors="replace") as f:
                r = log_lines(f, want)
        except OSError as e:
            r = [], [f"{argv[1]} cannot be read: {e.strerror}"], []
        except ValueError as e:
            r = [], [str(e)], []
        fails += r[0]; unread += r[1]; notes += r[2]
    if mode in ("procs", "both"):
        g = argv[1] if mode == "procs" else argv[2]
        if not g.isdigit():
            unread.append(f"process group '{g}' is not a number")
        else:
            r = procs(int(g), want)
            fails += r[0]; unread += r[1]; notes += r[2]
    v = "FAIL" if fails else "INCONCLUSIVE" if unread else "PASS"
    for line in notes + [f"FAIL {x}" for x in fails] + [f"UNREADABLE {x}" for x in unread]:
        print("  " + line)
    print(f"SPLIT WITNESS ({mode}) TM_GEMM_TUNE={want}: {v}")
    return {"PASS": 0, "FAIL": 1}.get(v, 2)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
