#!/usr/bin/env python3
"""store_line.py LOG LABEL ROOT [SAME_SET]: the conversation store's boot lines of one boot of the store windows,
judged: the scheduler's store line, and (PLAN Amendment 19) the twelve workers' tier-digest lines.

The line (store owner's form, 2026-11-17, logger vllm.v1.core.sched.v41_store_files), one per boot:

  VLLM_V41_STORE=1 active (scheduler): record set <ROOT>/set-<32 hex> under root <ROOT> (<N> other folders under
  the root, <M> of them record sets, <B> B: not read, not deleted); budgets: ...; boot scan: <P> pinned and <C>
  conversation entries loaded (...), <F> damaged entries removed (...), <X> over the conversation disk budget
  dropped, <U> unlisted chunk directories to delete (...); write failure records ...

What each boot must print (PLAN Amendment 15), ROOT being this build's own root:
  KD1 (the first store boot, on an empty root): 0 other folders, 0 entries loaded, 0 damaged, 0 dropped, 0 unlisted
  KD2, KN, KN2 (a boot that must find the records of the boot before it): entries loaded > 0, 0 damaged, 0 other
       record sets (another set under the root means this boot's fingerprint is not the one that saved them);
       the unlisted count is REPORT ONLY (a chunk whose entry a tool dropped just before the stop is listed and
       deleted at the next boot by design, store owner 2026-11-17), so is the dropped count
SAME_SET (a restart): the record set must be this one, the set of the boot before (`set-<32 hex>`).

The tier digests (owner's condition after window st1; store owner's kit of 2026-11-17): every worker logs
  nvfp4_8r: TP rank R layers A-B two-tier tier digest <64 hex>
and on this build, where every routed expert is on the card, the digest of a stage is by its rule the sha256 of
one zero byte (tier HOT = 0, int8) per expert and layer: 14 x 384 = 5,376 zero bytes on stages 0-13 and 14-27,
12 x 384 = 4,608 on stage 28-39. Required: twelve lines, four per stage, each equal to that value. Anything else
(a rank that put an expert in host memory, a missing line) is wrong: the fingerprint, and so the record set, would
not be the all-NVFP4 build's.

Prints two lines (digests, store line). Exit 0 as required, 3 a required figure is wrong, 2 a line could not be
read (absent, another form): never 0 by default."""
import hashlib
import re
import sys
from dataclasses import dataclass

MARK = "VLLM_V41_STORE=1 active (scheduler): record set "
FORM = re.compile(
    r"VLLM_V41_STORE=1 active \(scheduler\): record set (?P<set>\S+/set-[0-9a-f]{32}) under root (?P<root>\S+) "
    r"\((?P<other>\d+) other folders under the root, (?P<sets>\d+) of them record sets, (?P<other_bytes>\d+) B: "
    r"not read, not deleted\); budgets: .*?; boot scan: (?P<pinned>\d+) pinned and (?P<conv>\d+) conversation "
    r"entries loaded \(.*?\), (?P<damaged>\d+) damaged entries removed \((?P<damaged_bytes>\d+) B.*?\), "
    r"(?P<dropped>\d+) over the conversation disk budget dropped, (?P<unlisted>\d+) unlisted chunk directories "
    r"to delete")
LABELS = ("KD1", "KD2", "KN", "KN2")
DIGEST = re.compile(r"nvfp4_8r: TP rank (\d+) layers (\d+)-(\d+) two-tier tier digest ([0-9a-f]{64})\b")
STAGES = ((0, 13), (14, 27), (28, 39))
EXPERTS = 384


def all_hot_digest(first, last):
    """The tier digest of a stage whose every expert is HOT: sha256 of one zero byte per expert and layer. (Pure.)"""
    return hashlib.sha256(bytes((last - first + 1) * EXPERTS)).hexdigest()


def digests_wrong(log_text):
    """What is wrong with the workers' tier-digest lines, as words; empty when twelve lines, four per stage, each
    the all-hot digest of its stage. (Pure.)"""
    found = [(int(r), int(a), int(b), h) for r, a, b, h in DIGEST.findall(log_text)]
    bad = []
    if len(found) != 12:
        bad.append(f"{len(found)} tier-digest lines, want 12")
    for first, last in STAGES:
        mine = [(r, h) for r, a, b, h in found if (a, b) == (first, last)]
        if sorted(r for r, _ in mine) != [0, 1, 2, 3]:
            bad.append(f"layers {first}-{last}: ranks {sorted(r for r, _ in mine)}, want 0 to 3")
        off = [r for r, h in mine if h != all_hot_digest(first, last)]
        if off:
            bad.append(f"layers {first}-{last}: rank(s) {off} log another digest than the all-NVFP4 plan's")
    return bad


@dataclass(frozen=True)
class StoreLine:
    set: str
    root: str
    other: int
    sets: int
    other_bytes: int
    pinned: int
    conv: int
    damaged: int
    damaged_bytes: int
    dropped: int
    unlisted: int


def parse(log_text):
    """The StoreLine of the LAST store line in the log, or None when there is none or it has another form. (Pure.)"""
    lines = [line for line in log_text.splitlines() if MARK in line]
    if not lines:
        return None
    m = FORM.search(lines[-1])
    if m is None:
        return None
    d = m.groupdict()
    return StoreLine(d["set"], d["root"], *(int(d[k]) for k in (
        "other", "sets", "other_bytes", "pinned", "conv", "damaged", "damaged_bytes", "dropped", "unlisted")))


def wrong(line, label, root):
    """The required figures of this boot that are wrong, as words; empty when all hold. (Pure.)"""
    bad = []
    if line.root != root or not line.set.startswith(root + "/set-"):
        bad.append(f"root {line.root}, want {root}")
    if line.damaged != 0:
        bad.append(f"{line.damaged} damaged entries removed")
    if label == "KD1":
        for name, value in (("other folders", line.other), ("pinned entries loaded", line.pinned),
                            ("conversation entries loaded", line.conv), ("dropped over the disk budget", line.dropped),
                            ("unlisted chunk directories", line.unlisted)):
            if value != 0:
                bad.append(f"{value} {name} on a root that must be empty")
    else:
        if line.pinned + line.conv == 0:
            bad.append("0 entries loaded: the records of the boot before were not found")
        if line.sets != 0:
            bad.append(f"{line.sets} other record set(s) under the root: another fingerprint saved there")
    return bad


def main(argv):
    if len(argv) not in (3, 4) or argv[1] not in LABELS:
        print(__doc__)
        return 2
    log, label, root = argv[:3]
    same = argv[3] if len(argv) == 4 else None
    text = open(log, errors="replace").read()
    dbad = digests_wrong(text)
    print(f"TIER DIGESTS {label}: {'OK' if not dbad else 'FAIL (' + '; '.join(dbad) + ')'}: stages 0-13 and 14-27 "
          f"{all_hot_digest(0, 13)[:12]}..., stage 28-39 {all_hot_digest(28, 39)[:12]}... (all experts on the card)")
    line = parse(text)
    if line is None:
        print(f"STORE LINE {label}: COULD NOT BE READ in {log} (no store line, or not the form this tool knows)")
        return 2
    what = (f"set {line.set.rsplit('/', 1)[1]}; other folders {line.other} ({line.sets} record sets, "
            f"{line.other_bytes} B); loaded {line.pinned} pinned and {line.conv} conversation; damaged removed "
            f"{line.damaged} ({line.damaged_bytes} B); dropped {line.dropped}; unlisted {line.unlisted}")
    bad = wrong(line, label, root)
    if same is not None and line.set.rsplit("/", 1)[1] != same:
        bad.append(f"the record set is {line.set.rsplit('/', 1)[1]}, the boot before had {same}")
    print(f"STORE LINE {label}: {'OK' if not bad else 'FAIL (' + '; '.join(bad) + ')'}: {what}")
    return 3 if (bad or dbad) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
