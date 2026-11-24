#!/usr/bin/env python3
"""Count the mHC prefill kernel's first-dispatch witnesses by booted worker (kits 10 and 11). Emit only aggregates."""

import re
from collections import Counter
from pathlib import Path
import sys

WORKER = r"\(Worker_PP([0-2])_TP([0-3]) pid=([1-9][0-9]*)\)"
BOOT = re.compile(WORKER + r".*\[gguf_loader\.py:\d+\] nvfp4_8r: TP rank \d+ layers .* hot / 0 cold experts,")
MARKER = "VLLM_V41_MHC_PREFILL=1 first dispatch rows="
DISPATCH = re.compile(WORKER + r".* " + re.escape(MARKER) + r"([0-9]+)\s*$")
RANKS = {(str(pp), str(tp)) for pp in range(3) for tp in range(4)}


def counts(log: Path) -> tuple[int, int]:
    workers: dict[tuple[str, str], str] = {}
    dispatched: Counter[tuple[str, str, str]] = Counter()
    with log.open("r", encoding="utf-8", errors="replace") as lines:
        for line in lines:
            boot = BOOT.match(line)
            if boot:
                pp, tp, pid = boot.groups()
                if workers.setdefault((pp, tp), pid) != pid:
                    raise ValueError("more than one boot PID for a rank")
            if MARKER in line:
                hit = DISPATCH.match(line)
                if not hit or int(hit[4]) < 30:
                    raise ValueError("malformed first-dispatch marker")
                dispatched[hit[1], hit[2], hit[3]] += 1
    if workers.keys() != RANKS or len(set(workers.values())) != len(RANKS):
        raise ValueError("boot log does not identify all twelve distinct ranks")
    if any(workers[pp, tp] != pid for pp, tp, pid in dispatched):
        raise ValueError("dispatch PID differs from the boot witness")
    if any(count != 1 for count in dispatched.values()):
        raise ValueError("repeated first-dispatch marker on a rank")
    return len(workers), dispatched.total()


def main(log: Path, receipt: Path) -> int:
    try:
        workers, dispatched = counts(log)
        aggregate = f"mhc-workers\t{workers}\nmhc-dispatch\t{dispatched}\n"
        with receipt.open("x") as out:
            out.write(aggregate)
    except (OSError, ValueError) as exc:
        print(f"INCONCLUSIVE mHC dispatch: {exc}")
        return 34
    print(aggregate, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
