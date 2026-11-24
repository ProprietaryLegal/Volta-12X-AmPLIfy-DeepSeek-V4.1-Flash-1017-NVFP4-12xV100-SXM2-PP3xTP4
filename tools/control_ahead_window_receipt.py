#!/usr/bin/env python3
"""Count first early-header transfers by booted worker (kit 11's control-ahead). Emit only aggregates."""

from pathlib import Path
import re
import sys

WORKER = r"\(Worker_PP([0-2])_TP([0-3]) pid=([1-9][0-9]*)\)"
BOOT = re.compile(WORKER + r".*\[gguf_loader\.py:\d+\] nvfp4_8r: TP rank \d+ layers .* hot / 0 cold experts,")
MARKER = "VLLM_V41_PP_CONTROL_AHEAD=1 first transfer"
TRANSFER = re.compile(
    WORKER + r" INFO \d{2}-\d{2} \d{2}:\d{2}:\d{2} \[v41_pp_handoff\.py:\d+\] "
    + re.escape(MARKER) + r" \(stage ([0-2]), TP rank ([0-3])\)\s*$"
)
RANKS = {(str(pp), str(tp)) for pp in range(3) for tp in range(4)}


def counts(log: Path) -> tuple[int, int]:
    workers: dict[tuple[str, str], str] = {}
    transferred: dict[tuple[str, str], str] = {}
    with log.open("r", encoding="utf-8", errors="replace") as lines:
        for line in lines:
            boot = BOOT.match(line)
            if boot:
                pp, tp, pid = boot.groups()
                if workers.setdefault((pp, tp), pid) != pid:
                    raise ValueError("more than one boot PID for a rank")
            if MARKER in line:
                hit = TRANSFER.match(line)
                if not hit or (hit[1], hit[2]) != (hit[4], hit[5]):
                    raise ValueError("malformed early-header transfer marker or mismatched rank")
                rank = (hit[1], hit[2])
                if rank in transferred:
                    raise ValueError("repeated first-transfer marker on a rank")
                transferred[rank] = hit[3]
    if workers.keys() != RANKS or len(set(workers.values())) != len(RANKS):
        raise ValueError("boot log does not identify all twelve distinct ranks")
    if any(workers[rank] != pid for rank, pid in transferred.items()):
        raise ValueError("transfer PID differs from the boot witness")
    return len(workers), len(transferred)


def main(log: Path, receipt: Path) -> int:
    try:
        workers, transferred = counts(log)
        aggregate = f"control-ahead-workers\t{workers}\ncontrol-ahead-transfers\t{transferred}\n"
        with receipt.open("x") as out:
            out.write(aggregate)
    except (OSError, ValueError) as exc:
        print(f"INCONCLUSIVE control-ahead transfer: {exc}")
        return 34
    print(aggregate, end="")
    if transferred != workers:
        print(f"REQUALIFY A: control-ahead first transfer was not witnessed on {workers} of {workers} workers")
        return 33
    print(f"REQUALIFY A: control-ahead first transfer witnessed on {workers} of {workers} workers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
