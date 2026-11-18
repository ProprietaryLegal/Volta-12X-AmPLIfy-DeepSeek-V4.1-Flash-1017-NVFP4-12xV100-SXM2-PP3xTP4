#!/usr/bin/env python3
"""serve_checked.py: the record run/state/SERVE-CHECKED of the twelve-card build (owner's rule, 2026-11-17, after the owner made
this build the default server): the public port :18042 never holds a server that is still being checked. The full checks
run on the test port (driver phase servecheck); the serving boot on :18042 (servedefault) sends no request and
needs this record to match the trees AS THEY ARE NOW.

  serve_checked.py fields  [--tree B12] --config C --card-limit N|none
  serve_checked.py write   [--tree B12] --config C --card-limit N|none --window TAG --hot H --capture LINE --high MIB [--source S]
  serve_checked.py match   [--tree B12] --config C --card-limit N|none [--record FILE]
  serve_checked.py record-from WINDOW_DIR [--tree B12] [--record FILE]

THE MATCHED FIELDS (a change of any one makes the record stale and servecheck owed):
  config              F4c, F3c or F2 (the serving configuration)
  card_limit          the card limit of the boot in MiB, or none
  profile             the configuration's profile (its :18043 form; the :18042 port copy differs only in the port line)
  profile_sha_noport  sha256 of that profile with its port line(s) removed
  kit                 patches/CURRENT
  store_kit           patches/STORE, or none (the store's kit in the trees or not)
  eff_rows_sha        sha256 of the effective kit rows, as the driver's eff_rows() prints them (the kit's manifest,
                      with the store kit's manifest stacked over it when it is in the trees)
INFORMATION (written, not matched): time, window, source (servecheck or record-from), hot (per stage, the hot expert
counts servedefault must see again), capture (the accepted capture's summary line), high (the rehearsal's highest
card sample, MiB).
The record is one file, replaced as a whole (written beside it, then renamed) on every pass.
Exits: fields/write 0, or 9 on a bad input; match 0 MATCH, 1 STALE, 2 ABSENT or unreadable;
record-from 0 written, 1 refused (the window's serve did not pass, or the trees no longer match what it logged), 9 usage."""
import argparse
import calendar
import hashlib
import os
import re
import sys
import time
from pathlib import Path

TREE = os.environ.get("B12_ROOT", "/path/to/build")  # the build's tree (a placeholder in the published copy)
PROFILES = {"F4c": "serve-nvfp4-pp3-full-c4096-a4.yaml", "F3c": "serve-nvfp4-pp3-full-c4096-a3.yaml",
            "F2": "serve-nvfp4-pp3-full-a2.yaml"}
MATCHED = ("config", "card_limit", "profile", "profile_sha_noport", "kit", "store_kit", "eff_rows_sha")
INFO = ("time", "window", "source", "hot", "capture", "high")
PORT_LINE = re.compile(rb"^[ \t]*port[ \t]*:")


class Bad(Exception):
    pass


def manifest_rows(path):
    """The driver's rows(): lines not starting '#' with exactly three tab-separated fields."""
    out = []
    for line in Path(path).read_text().split("\n"):
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) == 3:
            out.append(f)
    return out


def eff_rows(tree):
    """The driver's eff_rows() (window/run_window_pp3.sh), line for line: the kit's rows; with the store's kit, a file
    of both takes the store's new sha256 (MISMATCH when the store's base is not the kit's new), and the store's other
    files follow in the store manifest's order. Returns the printed text."""
    kit = (Path(tree) / "patches" / "CURRENT").read_text().strip()
    store = (Path(tree) / "patches" / "STORE").read_text().strip() if (Path(tree) / "patches" / "STORE").exists() else ""
    krows = manifest_rows(Path(tree) / "patches" / kit / "MANIFEST")
    if not store:
        return "".join("\t".join(r) + "\n" for r in krows)
    srows = manifest_rows(Path(tree) / "patches" / store / "MANIFEST")
    sb = {p: b for p, b, _ in srows}
    sn = {p: n for p, _, n in srows}
    order = [p for p, _, _ in srows]
    out, seen = [], set()
    for p, b, n in krows:
        seen.add(p)
        if p in sn:
            out.append((p, b, sn[p]) if sb[p] == n else (p, "MISMATCH", "MISMATCH"))
        else:
            out.append((p, b, n))
    out += [(p, sb[p], sn[p]) for p in order if p not in seen]
    return "".join("\t".join(r) + "\n" for r in out)


def fields(tree, config, card_limit):
    if config not in PROFILES:
        raise Bad(f"configuration '{config}' is not one of {sorted(PROFILES)}")
    if card_limit != "none" and not re.fullmatch(r"[1-9][0-9]{3,5}", card_limit):
        raise Bad(f"card limit '{card_limit}' is not a number of MiB or none")
    t = Path(tree)
    prof = t / "run" / PROFILES[config]
    data = prof.read_bytes()
    kept = b"".join(l for l in data.splitlines(keepends=True) if not PORT_LINE.match(l))
    store = (t / "patches" / "STORE").read_text().strip() if (t / "patches" / "STORE").exists() else "none"
    return {"config": config, "card_limit": card_limit, "profile": PROFILES[config],
            "profile_sha_noport": hashlib.sha256(kept).hexdigest(),
            "kit": (t / "patches" / "CURRENT").read_text().strip(), "store_kit": store,
            "eff_rows_sha": hashlib.sha256(eff_rows(tree).encode()).hexdigest()}


def read_record(path):
    rec = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        k, sep, v = line.partition("=")
        if not sep:
            raise Bad(f"{path}: a line without '=': {line[:80]!r}")
        rec[k] = v
    return rec


def write_record(path, rec):
    for k, v in rec.items():
        if "\n" in v:
            raise Bad(f"field {k} holds a line break")
    p = Path(path)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text("# SERVE-CHECKED: the twelve-card build passed the full serving checks on these trees "
                   "(tools/serve_checked.py; servedefault requires the matched fields to equal the trees now)\n"
                   + "".join(f"{k}={rec[k]}\n" for k in MATCHED + INFO if k in rec))
    os.replace(tmp, p)


def compare(rec, now):
    return [f"{k}: record '{rec.get(k, '(absent)')}', trees now '{now[k]}'" for k in MATCHED if rec.get(k) != now[k]]


def record_from(window, tree):
    """The fields of a record from a window whose serve phase passed (its chain.out and phases.log), checked against
    the trees now. Returns (record, refusals)."""
    w = Path(window)
    tag = w.name
    text = ""
    for name in ("phases.log", "chain.out"):
        if (w / name).exists():
            text += (w / name).read_text(errors="replace") + "\n"
    no = []
    passed = re.search(rf"PP3WIN-{tag} serve EXIT=0 \(", text)
    if not passed:
        no.append(f"window {tag}'s serve phase did not end EXIT=0")
    cfg = (w / "serve-config").read_text().strip() if (w / "serve-config").exists() else ""
    if cfg not in ("F4c", "F3c"):
        no.append(f"window {tag}'s serve-config is '{cfg}': a record is taken only from a configuration whose serve "
                  "compared the accepted capture (F4c, F3c)")
    m = re.search(r"^url http://127\.0\.0\.1:\d+ on \S+ \(loopback only\); model \S+; profile \S*/run/([^/;\s]+);", text, re.M)
    prof = m.group(1) if m else ""
    if cfg in PROFILES and prof != PROFILES[cfg]:
        no.append(f"window {tag} served profile '{prof}', not {PROFILES.get(cfg)}")
    kit = re.search(rf"^CHAIN-\S+ {tag} order \S+ start (\S+); configuration (\S+); kit (\S+); card limit (\S+)$", text, re.M)
    if not kit:
        no.append(f"window {tag}'s chain.out has no start line with its kit and card limit")
        start = None
    else:
        start = calendar.timegm(time.strptime(kit.group(1), "%Y-%m-%dT%H:%M:%SZ"))
        if kit.group(2) != cfg:
            no.append(f"window {tag}'s chain configuration {kit.group(2)} is not its serve-config {cfg}")
    limit = re.search(rf"card limit (\d+) MiB: 12 of 12 workers log their allocator limit", text)
    hot = re.search(r"serving boot: every rank logs 0 cold; hot experts (\S+ \S+ \S+) = 15,360", text)
    same = re.search(r"against the accepted capture \(window o2, arm F3c\): capture exit 0; +(SUMMARY \S+ vs ACC: identical 7 of 7;.*)$", text, re.M)
    high = re.search(rf"rehearsal of P3{tag}-S: exit 0, out-of-memory lines 0, highest card sample (\d+) MiB", text)
    for what, mm in (("the card limit on 12 of 12", limit), ("the hot counts with 0 cold", hot),
                     ("the accepted capture 7 of 7", same), ("a rehearsal with exit 0 and 0 out-of-memory lines", high)):
        if not mm:
            no.append(f"window {tag}'s log does not show {what}")
    two_tier = len(set(re.findall(r"TP rank (\d+) layers (\d+-\d+) two-tier tier digest [0-9a-f]{64}", text)))
    if no:
        return None, no
    now = fields(tree, cfg, limit.group(1))
    # What the window logged, against the trees now: the kit of its start line; the store's kit (the store kit's
    # engine logs a two-tier tier digest per worker: twelve lines with it, none without); the profile, the kit names
    # and the store kit file unchanged since the window started (their mtime before its start: the window logged no
    # sha256 of them).
    if kit.group(3) != now["kit"]:
        no.append(f"window {tag} ran kit {kit.group(3)}; the trees hold {now['kit']}")
    if kit.group(4) != limit.group(1):
        no.append(f"window {tag}'s chain card limit {kit.group(4)} is not its boot's {limit.group(1)}")
    had_store = two_tier >= 12
    if two_tier not in (0, 12) or had_store != (now["store_kit"] != "none"):
        no.append(f"window {tag}'s boot logged {two_tier} two-tier tier digest lines (store kit {'in' if had_store else 'not in'} "
                  f"the trees then); the trees now: store kit {now['store_kit']}")
    t = Path(tree)
    for f in [t / "run" / PROFILES[cfg], t / "patches" / "CURRENT"] + ([t / "patches" / "STORE"] if now["store_kit"] != "none" else []):
        if start is None or f.stat().st_mtime >= start:
            no.append(f"{f} was changed at or after window {tag}'s start: not what that window served")
    if no:
        return None, no
    rec = dict(now)
    rec.update({"time": time.strftime("%FT%TZ", time.gmtime()), "window": tag,
                "source": f"record-from {tag} (serve EXIT=0)", "hot": hot.group(1),
                "capture": same.group(1)[:300], "high": high.group(1)})
    return rec, []


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fields", "write", "match"):
        s = sub.add_parser(name)
        s.add_argument("--tree", default=TREE)
        s.add_argument("--config", required=True)
        s.add_argument("--card-limit", required=True)
        if name == "match":
            s.add_argument("--record")
        if name == "write":
            s.add_argument("--record")
            for k in ("window", "hot", "capture", "high"):
                s.add_argument("--" + k, required=True)
            s.add_argument("--source", default="servecheck")
    s = sub.add_parser("record-from")
    s.add_argument("window")
    s.add_argument("--tree", default=TREE)
    s.add_argument("--record")
    a = ap.parse_args(argv)
    record = getattr(a, "record", None) or str(Path(a.tree) / "run" / "state" / "SERVE-CHECKED")
    try:
        if a.cmd == "record-from":
            rec, no = record_from(a.window, a.tree.rstrip("/"))
            if no:
                for x in no:
                    print(f"  REFUSED {x}")
                print(f"SERVE-CHECKED NOT WRITTEN from {a.window}")
                return 1
            write_record(record, rec)
            print(f"SERVE-CHECKED WRITTEN {record} from window {rec['window']}: " + "; ".join(f"{k}={rec[k]}" for k in MATCHED + INFO))
            return 0
        now = fields(a.tree, a.config, a.card_limit)
        if a.cmd == "fields":
            print("".join(f"{k}={now[k]}\n" for k in MATCHED), end="")
            return 0
        if a.cmd == "write":
            if not re.fullmatch(r"\d+", a.high) or not a.hot or not a.capture:
                raise Bad("--high must be MiB and --hot, --capture must not be empty")
            rec = dict(now)
            rec.update({"time": time.strftime("%FT%TZ", time.gmtime()), "window": a.window, "source": a.source,
                        "hot": a.hot, "capture": a.capture[:300], "high": a.high})
            write_record(record, rec)
            print(f"SERVE-CHECKED WRITTEN {record}: " + "; ".join(f"{k}={rec[k]}" for k in MATCHED + INFO))
            return 0
        # match
        if not Path(record).exists():
            print(f"SERVE-CHECKED ABSENT: {record} (servecheck owed)")
            return 2
        rec = read_record(record)
        diff = compare(rec, now)
        if diff:
            for d in diff:
                print(f"  STALE {d}")
            print(f"SERVE-CHECKED STALE: {record} (window {rec.get('window', '?')}, {rec.get('time', '?')}): servecheck owed")
            return 1
        print(f"SERVE-CHECKED MATCH: {record} (window {rec.get('window', '?')}, {rec.get('time', '?')}, source "
              f"{rec.get('source', '?')}); hot {rec.get('hot', '?')}; rehearsal high {rec.get('high', '?')} MiB")
        return 0
    except (OSError, Bad, ValueError) as e:
        print(f"SERVE-CHECKED UNREADABLE: {e}")
        return 2 if a.cmd == "match" else 9


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
