"""CPU checks of tools/store_line.py. Two kinds of lines: (1) REAL ones, tests/data/store-lines-real.txt: the 34
store lines of the server's form that the store owner's rebuilt engine logged in its CPU runs of 2026-11-17
(store-on-cpu-normal.log and -faults.log of the store's CPU runs in a private working repository; public
text, temporary folders as roots); (2) lines filled into the same form by hand, for the rule's cases the real
runs do not hold on this build's root. Expected verdicts follow PLAN Amendment 15."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("store_line", ROOT / "tools" / "store_line.py")
sl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sl)

R = "/path/to/build/store"
SET = R + "/set-0123456789abcdef0123456789abcdef"


def line(root=R, other=0, sets=0, other_bytes=0, pinned=0, conv=0, damaged=0, dropped=0, unlisted=0, set_=SET):
    return ("(EngineCore pid=1) INFO 11-17 12:00:00 [v41_store_files.py:1] VLLM_V41_STORE=1 active (scheduler): "
            f"record set {set_} under root {root} ({other} other folders under the root, {sets} of them record "
            f"sets, {other_bytes} B: not read, not deleted); budgets: pinned RAM 68719476736 B, conversation RAM "
            "34359738368 B, pinned disk 100000000000 B, conversation disk 100000000000 B; boot scan: "
            f"{pinned} pinned and {conv} conversation entries loaded (0 pinned in RAM, 0 B), {damaged} damaged "
            f"entries removed (0 B of entry files), {dropped} over the conversation disk budget dropped, {unlisted} "
            "unlisted chunk directories to delete (0 B); write failure records /dev/shm/v41-store-failures (run 7).")


def verdict(text, label):
    parsed = sl.parse(text)
    return None if parsed is None else sl.wrong(parsed, label, R)


def test_first_boot_on_an_empty_root():
    assert verdict(line(), "KD1") == []
    assert verdict(line(conv=1), "KD1") != [] and verdict(line(other=1), "KD1") != []
    assert verdict(line(unlisted=1), "KD1") != [] and verdict(line(damaged=1), "KD1") != []


def test_a_restart_must_find_its_records_and_no_damage():
    assert verdict(line(pinned=5, conv=9), "KD2") == []
    assert verdict(line(pinned=5, conv=9, unlisted=3, dropped=1), "KD2") == []      # report only
    assert verdict(line(), "KD2") != []                                             # nothing loaded
    assert verdict(line(pinned=5, damaged=1), "KN2") != []
    assert verdict(line(pinned=5, other=1, sets=1), "KN") != []                     # another fingerprint's set
    assert verdict(line(pinned=5, other=1, sets=0), "KN") == []                     # a folder that is no record set


def test_another_root_or_no_line_is_never_a_pass(tmp_path):
    assert verdict(line(root="/path/to/eight-card-build/store", set_="/path/to/eight-card-build/store/set-" + "a" * 32,
                        pinned=1), "KD2") != []
    assert sl.parse("no store here\n") is None
    assert sl.parse(line().replace("boot scan", "scan")) is None
    log = tmp_path / "a.log"
    log.write_text("x\n")
    assert sl.main([str(log), "KD1", R]) == 2
    log.write_text(digest_lines() + line() + "\n")
    assert sl.main([str(log), "KD1", R]) == 0
    log.write_text(digest_lines() + line(conv=2) + "\n")
    assert sl.main([str(log), "KD1", R]) == 3
    assert sl.main([str(log), "KA", R]) == 2


def test_the_last_line_counts():
    text = line(conv=4) + "\n" + line() + "\n"
    assert sl.parse(text).conv == 0


REAL_LINES = (ROOT / "tests" / "data" / "store-lines-real.txt").read_text().splitlines()


def test_every_real_line_of_the_rebuilt_engine_is_read():
    assert len(REAL_LINES) == 34
    parsed = [sl.parse(line) for line in REAL_LINES]
    assert all(p is not None for p in parsed)
    assert all(p.set.startswith(p.root + "/set-") for p in parsed)
    # the real runs hold every kind the rule distinguishes
    assert sum(1 for p in parsed if p.pinned + p.conv > 0) == 4
    assert sum(1 for p in parsed if p.damaged > 0) == 1
    assert sum(1 for p in parsed if p.sets > 0) == 10
    assert sum(1 for p in parsed if p.unlisted > 0) == 1


def test_the_rule_on_real_lines():
    parsed = [sl.parse(line) for line in REAL_LINES]
    empty = parsed[0]                       # the first real line: a boot on an empty root
    assert (empty.other, empty.pinned, empty.conv, empty.damaged, empty.unlisted) == (0, 0, 0, 0, 0)
    assert sl.wrong(empty, "KD1", empty.root) == []
    assert sl.wrong(empty, "KD1", R) != []                      # not this build's root
    assert sl.wrong(empty, "KD2", empty.root) != []             # a restart that loads nothing
    damaged = next(p for p in parsed if p.damaged > 0)
    assert any("damaged" in w for w in sl.wrong(damaged, "KD2", damaged.root))
    # the four real boots that loaded entries: never the "nothing loaded" complaint (each of them also has another
    # record set or a damaged record beside it in these test runs, which the rule reports; the clean restart on
    # this build's own root is the hand-filled case above)
    loaded = [p for p in parsed if p.pinned + p.conv > 0]
    assert len(loaded) == 4 and all(not any("0 entries loaded" in w for w in sl.wrong(p, "KN2", p.root)) for p in loaded)
    assert all(sl.wrong(p, "KN2", p.root) != [] for p in loaded)
    foreign = next(p for p in parsed if p.sets > 0 and p.damaged == 0)
    assert any("other record set" in w for w in sl.wrong(foreign, "KD2", foreign.root))
    assert sl.wrong(foreign, "KD1", foreign.root) != []


def digest_lines(stage2_last="x", drop=None, cold_rank=None):
    import hashlib
    out = []
    for first, last in ((0, 13), (14, 27), (28, 39)):
        for rank in range(4):
            if drop == (first, rank):
                continue
            h = hashlib.sha256(bytes((last - first + 1) * 384)).hexdigest()
            if cold_rank == (first, rank):      # one expert of one layer in host memory (tier HOST = 2)
                b = bytearray((last - first + 1) * 384); b[5] = 2
                h = hashlib.sha256(bytes(b)).hexdigest()
            out.append(f"(Worker pid=1) INFO 11-17 12:00:00 [gguf_loader.py:158] nvfp4_8r: TP rank {rank} layers "
                       f"{first}-{last} two-tier tier digest {h}")
    return "\n".join(out) + "\n"


def test_all_hot_digest_is_the_sha256_of_zero_bytes():
    # arithmetic, independent of the engine: 14 layers x 384 experts and 12 x 384, one int8 zero (HOT) each
    import hashlib
    assert sl.all_hot_digest(0, 13) == hashlib.sha256(b"\x00" * 5376).hexdigest()
    assert sl.all_hot_digest(14, 27) == sl.all_hot_digest(0, 13)
    assert sl.all_hot_digest(28, 39) == hashlib.sha256(b"\x00" * 4608).hexdigest() != sl.all_hot_digest(0, 13)


def test_twelve_digest_lines_of_the_all_nvfp4_plan():
    assert sl.digests_wrong(digest_lines()) == []
    assert sl.digests_wrong("") != []
    assert any("want 12" in w for w in sl.digests_wrong(digest_lines(drop=(14, 2))))
    assert any("another digest" in w for w in sl.digests_wrong(digest_lines(cold_rank=(28, 1))))


def test_main_with_digests_and_the_same_set(tmp_path):
    log = tmp_path / "b.log"
    name = SET.rsplit("/", 1)[1]
    log.write_text(digest_lines() + line(pinned=3, conv=4) + "\n")
    assert sl.main([str(log), "KD2", R]) == 0
    assert sl.main([str(log), "KD2", R, name]) == 0
    assert sl.main([str(log), "KD2", R, "set-" + "f" * 32]) == 3        # the restart opened another record set
    log.write_text(digest_lines(cold_rank=(0, 0)) + line(pinned=3, conv=4) + "\n")
    assert sl.main([str(log), "KD2", R, name]) == 3                       # a rank is not all NVFP4
    log.write_text(line(pinned=3, conv=4) + "\n")
    assert sl.main([str(log), "KD2", R, name]) == 3                       # no digest lines
