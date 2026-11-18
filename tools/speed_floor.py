#!/usr/bin/env python3
"""speed_floor.py SUMMARY LABEL READ [--n 4]: arm G's speed line (owner's rule, 2026-11-17): each of five figures at
least 95 % of window s4's, the standing configuration's acceptance window (arm F4c, PLAN Amendment 19):

  one request, fresh text        128.1 tok/s    ("one alone" of the fresh row of slots_cells.py summary)
  one request, repeated text     176.6 tok/s    ("one alone" of the repeated row)
  four together, fresh, SUM      306.6 tok/s    (SUM of the fresh row)
  four together, repeated, SUM   453.4 tok/s    (SUM of the repeated row)
  prompt read                    3,284 tok/s    (the LADDER line of stage/pex-speed/ctx_ladder.py, as armF4c line f)

The figures are read from the PRINTED outputs, as window s4 printed its own (S4-RESULT.txt, read-F4c.out):
SUMMARY is the output of `slots_cells.py summary LABEL=DIR`, READ the output of ctx_ladder.py. The comparison is
exact on the printed decimals: figure x 100 >= s4's x 95.
  FAIL          a figure is under 95 % of s4's (decides even when another figure is unreadable)
  INCONCLUSIVE  no FAIL, and a figure cannot be read (missing row, "one alone None", no LADDER line, a row of another
                request count than --n)
  PASS          every figure at least 95 %; a gain is reported, never required
Exits 0 PASS, 1 FAIL, 2 INCONCLUSIVE."""
import argparse
import re
import sys
from decimal import Decimal

S4 = (("one request, fresh text", "fresh", "alone", Decimal("128.1")),
      ("one request, repeated text", "repeated", "alone", Decimal("176.6")),
      ("four together, fresh text, SUM", "fresh", "sum", Decimal("306.6")),
      ("four together, repeated text, SUM", "repeated", "sum", Decimal("453.4")),
      ("prompt read", "read", "read", Decimal("3284")))
HEAD = re.compile(r"^(fresh|repeated) text, requests together")
ROW = re.compile(r"^\s+(?P<label>\S+) (?P<n>\d+) together: SUM\s+(?P<sum>\d+(?:\.\d+)?) \(.*\), WALL .*, one alone "
                 r"(?P<alone>\S+), first token ")
LADDER = re.compile(r"^LADDER .* read (?P<read>\d+) tok/s")


def summary_figures(text, label, n):
    """{(kind, 'sum'|'alone'): Decimal or None} from the summary's printed rows of LABEL (pure); None = unreadable."""
    got, kind = {}, None
    for line in text.splitlines():
        h = HEAD.match(line)
        if h:
            kind = h.group(1)
            continue
        m = ROW.match(line)
        if not m or m["label"] != label or kind is None:
            continue
        if (kind, "sum") in got:  # a second row of the same label and kind: not the figure s4 printed once
            got[(kind, "sum")] = got[(kind, "alone")] = None
            continue
        ok = int(m["n"]) == n
        got[(kind, "sum")] = Decimal(m["sum"]) if ok else None
        got[(kind, "alone")] = Decimal(m["alone"]) if ok and re.fullmatch(r"\d+(?:\.\d+)?", m["alone"]) else None
    return got


def read_figure(text):
    """The prompt-read tok/s of the LAST LADDER line (as the driver's sed takes it), or None (pure)."""
    vals = [m["read"] for m in map(LADDER.match, text.splitlines()) if m]
    return Decimal(vals[-1]) if vals else None


def judge(fig):
    """fig: {(kind, what): Decimal or None}. Returns (verdict, lines) (pure)."""
    lines, fail, inc = [], False, False
    for name, kind, what, ref in S4:
        x = fig.get((kind, what))
        floor = ref * Decimal("0.95")
        if x is None:
            lines.append(f"{name}: UNREADABLE (window s4: {ref}; floor {floor})")
            inc = True
            continue
        pct = (x * 100 / ref).quantize(Decimal("0.1"))
        if x * 100 >= ref * 95:
            gain = " (a gain: reported, not required)" if x > ref else ""
            lines.append(f"{name}: {x}, {pct} % of window s4's {ref} (floor {floor}): HOLDS{gain}")
        else:
            lines.append(f"{name}: {x}, {pct} % of window s4's {ref} (floor {floor}): UNDER 95 %")
            fail = True
    return ("FAIL" if fail else "INCONCLUSIVE" if inc else "PASS"), lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("label")
    ap.add_argument("read")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    fig = {}
    try:
        fig.update(summary_figures(open(a.summary, encoding="utf-8", errors="replace").read(), a.label, a.n))
    except OSError as e:
        print(f"speed_floor: {a.summary} cannot be read: {e}")
    try:
        fig[("read", "read")] = read_figure(open(a.read, encoding="utf-8", errors="replace").read())
    except OSError as e:
        print(f"speed_floor: {a.read} cannot be read: {e}")
    v, lines = judge(fig)
    for line in lines:
        print("  " + line)
    print(f"SPEED LINE {a.label}: {v} (each figure at least 95 % of window s4's)")
    sys.exit({"PASS": 0, "FAIL": 1}.get(v, 2))


if __name__ == "__main__":
    main()
