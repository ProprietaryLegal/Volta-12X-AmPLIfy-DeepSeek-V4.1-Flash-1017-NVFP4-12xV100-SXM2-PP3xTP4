#!/usr/bin/env python3
"""oom_lines.py LOG: how many lines of a server log are a real out-of-memory error. Prints the number.

A real one is the exception as PyTorch prints it at the end of a traceback, e.g. (2026-11-10, the eight-card build's
W5-piqx.log): "... ERROR ... torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 108.00 MiB. GPU 2 has a
total capacity of ...". So a line counts when it holds "OutOfMemoryError: " (the class, a colon, the message) or
"CUDA out of memory".

It is NOT one when it is the card limit's own announcement at the end of warm-up (kit 6c, card_limit.py: "V4.1
card memory after warm-up (...): ...; VLLM_V41_CARD_LIMIT_MIB=N: allocator limit ... (headroom ...); beyond it an
allocation fails with torch.OutOfMemoryError"). Window k1 (2026-11-17) counted those twelve lines, one per worker,
as twelve out-of-memory lines with the old pattern "OutOfMemoryError|CUDA out of memory": the limited boot was
refused and booted again without the limit although nothing had failed. One reader for the driver and the chains.
Exit 0 with the number printed; 9 bad arguments; an unreadable log raises (never a 0 by default)."""
import sys

REAL = ("OutOfMemoryError: ", "CUDA out of memory")
ANNOUNCEMENT = "V4.1 card memory after warm-up"


def is_oom(line):
    """True for a real out-of-memory error line, False for anything else. (Pure.)"""
    return ANNOUNCEMENT not in line and any(mark in line for mark in REAL)


def count(text):
    return sum(1 for line in text.splitlines() if is_oom(line))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(9)
    print(count(open(sys.argv[1], errors="replace").read()))
