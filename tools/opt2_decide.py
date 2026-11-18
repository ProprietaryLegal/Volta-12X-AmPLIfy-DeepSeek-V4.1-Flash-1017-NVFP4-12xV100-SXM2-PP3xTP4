#!/usr/bin/env python3
"""opt2_decide.py HIGH_F3C HIGH_F3O READ_F3C READ_F3O CEIL: what window o2 serves (PLAN Amendment 12). HIGH_x is the
highest card sample (MiB) of an arm that PASSED every line, READ_x its prompt-read speed (tok/s); "-" = the arm did
not pass (or the figure is absent). Prints "PICK why". A three-slot arm qualifies only with HIGH at or under CEIL.
Both: the lower HIGH; within 100 MiB of each other the faster read; equal reads: F3o (it does not re-cut prompts);
a read speed missing in that case: F2, said as "could not decide". Neither: F2."""
import sys


def decide(hc, ho, rc, ro, ceil):
    qc, qo = hc is not None and hc <= ceil, ho is not None and ho <= ceil
    if qc and qo:
        if abs(hc - ho) > 100:
            return ("F3c" if hc < ho else "F3o"), "both qualify; the lower highest card sample"
        if rc is None or ro is None:
            return "F2", "both qualify within 100 MiB but a prompt-read speed could not be read: COULD NOT DECIDE between them; two slots stays"
        if rc > ro:
            return "F3c", "both qualify within 100 MiB; the faster prompt read"
        return "F3o", "both qualify within 100 MiB; its prompt read is not slower, and it does not re-cut prompts"
    if qc:
        return "F3c", "arm F3c alone qualifies"
    if qo:
        return "F3o", "arm F3o alone qualifies"
    return "F2", f"no three-slot arm passed every line with its highest card sample at or under {ceil} MiB"


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(9)
    hc, ho, rc, ro = (None if a == "-" else int(a) for a in sys.argv[1:5])
    print(*decide(hc, ho, rc, ro, int(sys.argv[5])))
