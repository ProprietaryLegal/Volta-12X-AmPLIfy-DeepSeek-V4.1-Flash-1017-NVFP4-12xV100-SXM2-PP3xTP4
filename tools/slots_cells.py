#!/usr/bin/env python3
"""slots_cells.py: N requests together on the twelve-card build (the owner's decision of 2026-11-17 to test four,
six and eight requests at once). Public prompts only; writes numbers and sha256 digests only (no reply text is kept).

  slots_cells.py run   --n N --out DIR --texts DIR [--url http://127.0.0.1:18043] [--passes 2]
  slots_cells.py judge --n N --dir DIR
  slots_cells.py lineh --n N --out DIR --texts DIR [--url http://127.0.0.1:18043]
  slots_cells.py summary LABEL=DIR [LABEL=DIR ...]

The request code and the first three prompts of each kind are those of the server owner's window tool
(stage/stack-window/win_cells.py: F1 F2 F3, RA RB RC); the other five of each kind are made the same way from the
same two pinned public texts (the Apache licence and vLLM's metrics.py in --texts) and from plain public tasks.

run (one boot), per pass: F1 alone, RA alone, then the N fresh prompts together, then the N repeated-text prompts
together. Before a cell each of its prompts is read once (the same request with max_tokens 1), as in win_cells. A
cell does not start while the server has a request running or waiting. Per request: tokens, tok/s over its own
decode time (first to last token), time to first token. Per cell: SUM = the requests' tok/s added, WALL = all
tokens over the time from the first token of any request to the last token of all, and the most requests the server
reported running at once. Exits: 0 done, 3 server busy, 4 a request failed, 9 bad arguments.

judge: the rule's "no request starved" line for a run (fixed 2026-11-17 before any N-slot data), over every
together cell of every pass:
  FAIL          a request did not finish (no finish reason "stop" or "length"), or its tok/s is under half of the
                cell's mean tok/s per request;
  INCONCLUSIVE  the server was never seen running all N at once in some together cell, or a cell is missing;
  PASS          otherwise.
Exits 0 PASS, 1 FAIL, 2 INCONCLUSIVE, 5 MALFORMED.
STRICT RECEIPT (2026-11-17, after the independent review): the cells of the run are named, not counted. Each pass
1..2 must hold exactly the four cells run() writes, F1 alone, RA alone, the N fresh together and the N repeated
together (expected_cells), each once, each request named as its cell names it. A cell missing, repeated or extra:
INCONCLUSIVE. A record without its fields, a token count that is not a count, or a decode time that is not a finite
positive number: MALFORMED (exit 5; it is never computed, so it can never crash into exit 1, which the driver reads as
FAIL). A request that starved in a well-formed cell is FAIL even when another cell is missing.

summary: one table over several runs (the build at 3, 4, 6, 8 slots): per run and kind of text the
SUM and WALL tok/s (mean of the passes), the slowest and fastest request, and the time to first token (median and
longest) of the together cells; each SUM also as a ratio to the first LABEL's and, when a LABEL "X3" is given, to
X3's. The question it answers: does more than three at once buy anything, and what does a request wait.

lineh: the store owner's line h at N: N chats at once, each answer against the same chat alone, every send a cold
read (a fresh cache salt). Per group (the N fresh, then the N repeated): alone, together, alone again; 300 tokens
each. PASS: every together answer equals its two alone answers and N were seen running together. FAIL: a together
answer differs from two equal alone answers. INCONCLUSIVE: an alone answer does not repeat, N were never seen
together, or a row is missing. Exits 0 PASS, 1 FAIL, 2 INCONCLUSIVE, 5 MALFORMED.
STRICT (2026-11-17): each group's rows are named: every chat's solo1 and solo2 row and the group's together row,
each exactly once (expected_lineh); a missing, repeated or extra row: INCONCLUSIVE (a repeated row no longer
overwrites the first); a row without its fields or with digests that do not match its names: MALFORMED (exit 5).
Any unexpected error of any subcommand exits 5, never 1.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import urllib.request
import uuid

PIN = {"LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
       "metrics.py": "acf306f1b453f33f9cebd13abfa6547dea5dff4a1dc92d4e9b701daf526746ee"}
FRESH = {
    "F1": "Write a Python module that implements an LRU cache class with get, put and delete, a time-to-live option, "
          "type hints and docstrings, then unit tests for it with unittest.",
    "F2": "Explain, for a first-year law student, the elements of adverse possession at common law, with one worked "
          "hypothetical for each element.",
    "F3": "Describe step by step how a B-tree insertion splits a full node, with a small numeric example, then give "
          "pseudocode for insert and split.",
    "F4": "Write a Python function that parses ISO 8601 durations such as P3DT4H30M into seconds, with type hints and "
          "docstrings, then tests for it with pytest.",
    "F5": "Explain, for a first-year law student, the mailbox rule in contract formation, with one worked "
          "hypothetical for each of its limits.",
    "F6": "Describe step by step how Dijkstra's algorithm finds shortest paths with a binary heap, with a small "
          "numeric example, then give pseudocode.",
    "F7": "Write a SQL schema for a library lending system with members, books, copies and loans, then five queries "
          "against it, each with a one-paragraph explanation.",
    "F8": "Explain how TCP congestion control moves through slow start, congestion avoidance and fast recovery, with "
          "a numeric example of the congestion window at each step.",
}
# name: (source text, heading); the licence parts are cut at its numbered headings
REPEATED = {
    "RA": ("LICENSE", "Below is a licence. Write out sections 2 through 5 word for word, changing only 'Licensor' to 'Grantor'."),
    "RB": ("LICENSE", "Below is a licence. Write out section 1 (Definitions) word for word, changing only 'Licensor' to 'Grantor'."),
    "RC": ("metrics.py", "Below is a Python file. Return the whole file with every occurrence of num_drafts changed to "
                         "num_proposals, and nothing else changed."),
    "RD": ("LICENSE", "Below is a licence. Write out sections 6 through 9 word for word, changing only 'Licensor' to 'Grantor'."),
    "RE": ("LICENSE", "Below is a licence. Write out sections 2 through 5 word for word, changing only 'Derivative Works' "
                      "to 'Adapted Works'."),
    "RF": ("LICENSE", "Below is a licence. Write out section 1 (Definitions) word for word, changing only 'Contribution' "
                      "to 'Submission'."),
    "RG": ("metrics.py", "Below is a Python file. Return the whole file with every occurrence of num_accepted_tokens "
                         "changed to num_kept_tokens, and nothing else changed."),
    "RH": ("metrics.py", "Below is a Python file. Return the whole file with every occurrence of spec_decoding_stats "
                         "changed to spec_stats, and nothing else changed."),
}
TOKENS = {"F": 600, "R": 1500}
FINISHED = ("stop", "length")


def names(n):
    if not 2 <= n <= 8:
        raise SystemExit("slots_cells: --n must be 2..8")
    return tuple(sorted(FRESH)[:n]), tuple(sorted(REPEATED)[:n])


def read_pinned(texts, name):
    data = open(os.path.join(texts, name), "rb").read()
    got = hashlib.sha256(data).hexdigest()
    if got != PIN[name]:
        raise SystemExit(f"slots_cells: {name} is {got[:16]}, not the pinned public text {PIN[name][:16]}")
    return data.decode()


def prompts(texts):
    source = {name: read_pinned(texts, name) for name in PIN}
    p = dict(FRESH)
    for name, (text, head) in REPEATED.items():
        p[name] = head + "\n\n" + source[text]
    return p


def metric_sum(url, metric_names):
    text = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
    total, seen = 0.0, 0
    for name in metric_names:
        for m in re.finditer(r"^vllm:" + re.escape(name) + r"\{[^}]*\} (\S+)$", text, re.M):
            total += float(m.group(1))
            seen += 1
    if seen == 0:
        raise SystemExit(f"slots_cells: none of vllm:{metric_names} found at {url}/metrics")
    return total


def one_request(url, model, prompt, max_tokens, res, i, name, salt=None):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if salt is not None:
        body["cache_salt"] = salt  # a fresh salt makes the read cold (no prefix, snapshot or store hit)
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    h = hashlib.sha256()
    first = usage = finish = None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("error"):
                raise RuntimeError(f"stream error: {str(d['error'])[:200]}")
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
                dl = c.get("delta", {})
                piece = dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")
                if piece:
                    if first is None:
                        first = time.time()
                    h.update(piece.encode())
    last = time.time()
    if first is None or usage is None or finish is None:
        raise RuntimeError("no token, no usage block or no finish reason")
    res[i] = {"name": name, "out": usage["completion_tokens"], "prompt": usage["prompt_tokens"],
              "decode_s": round(last - first, 4), "ttft_s": round(first - t0, 4), "first": first, "last": last,
              "finish": finish, "sha256": h.hexdigest()}


def together(url, model, P, cell, max_tokens, salted):
    """Send the cell's requests at once; returns (rows, peak running)."""
    if metric_sum(url, ["num_requests_running", "num_requests_waiting"]) != 0:
        print("slots_cells: the server has a request running or waiting; stopping", file=sys.stderr)
        sys.exit(3)
    res = [None] * len(cell)
    errs = []
    peak = [0]
    stop = threading.Event()

    def guarded(i):
        try:
            one_request(url, model, P[cell[i]], max_tokens(cell[i]), res, i, cell[i],
                        salt=uuid.uuid4().hex[:12] if salted else None)
        except Exception as e:  # reported below; the run stops
            errs.append(f"{cell[i]}: {e!r}")

    def poll():
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], int(metric_sum(url, ["num_requests_running"])))
            except Exception:
                pass
            stop.wait(0.2)

    watcher = threading.Thread(target=poll)
    watcher.start()
    threads = [threading.Thread(target=guarded, args=(i,)) for i in range(len(cell))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    watcher.join()
    if errs:
        print(f"slots_cells: request failed: {errs}", file=sys.stderr)
        sys.exit(4)
    return res, peak[0]


def cell_numbers(req):
    """SUM and WALL tok/s of one cell (pure)."""
    total = sum(r["out"] for r in req)
    span = max(r["last"] for r in req) - min(r["first"] for r in req)
    return round(sum(r["out"] / r["decode_s"] for r in req), 2), round(total / span, 2)


def run(a):
    os.makedirs(a.out)  # a new folder each run
    fresh, repeated = names(a.n)
    P = prompts(a.texts)
    model = json.load(urllib.request.urlopen(a.url + "/v1/models", timeout=10))["data"][0]["id"]
    for p in range(1, a.passes + 1):
        for cell in ((fresh[0],), (repeated[0],), fresh, repeated):
            for n in cell:  # the read: the same request, one token
                one_request(a.url, model, P[n], 1, [None], 0, n)
            req, peak = together(a.url, model, P, cell, lambda n: TOKENS[n[0]], salted=False)
            total, wall = cell_numbers(req)
            row = {"pass": p, "cell": "+".join(cell), "req": req, "peak_running": peak, "sum_tok_s": total,
                   "wall_tok_s": wall, "utc": time.strftime("%FT%TZ", time.gmtime())}
            with open(os.path.join(a.out, "cells.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"CELL pass {p} {len(cell)} together {row['cell']}: " + ", ".join(
                f"{r['name']} {r['out']} tok {r['out'] / r['decode_s']:.1f} tok/s ttft {r['ttft_s']:.1f}s" for r in req)
                + f"; SUM {total:.1f} tok/s, WALL {wall:.1f} tok/s, peak running {peak}", flush=True)
    print("slots_cells: done")


def expected_cells(n, passes=2):
    """[(pass, cell)] that run() writes for N at PASSES (pure)."""
    fresh, repeated = names(n)
    return [(p, "+".join(c)) for p in range(1, passes + 1) for c in ((fresh[0],), (repeated[0],), fresh, repeated)]


def _count(x):
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _positive(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def cells_gaps(rows, n, passes=2):
    """(malformed, incomplete): the reasons a cells receipt is malformed, and those it is incomplete (pure)."""
    bad, gaps = [], []
    seen = {}
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or not all(k in r for k in ("pass", "cell", "req", "peak_running")) \
                or not isinstance(r["req"], list) or not isinstance(r["cell"], str):
            bad.append(f"row {i + 1}: not a cell record (pass, cell, req, peak_running)")
            continue
        for q in r["req"]:
            if not isinstance(q, dict) or not all(k in q for k in ("name", "out", "decode_s", "finish")):
                bad.append(f"pass {r['pass']} {r['cell']}: a request without name, out, decode_s or finish")
            elif not _count(q["out"]) or not _positive(q["decode_s"]):
                bad.append(f"pass {r['pass']} {r['cell']} {q['name']}: out {q['out']!r}, decode_s {q['decode_s']!r}")
        if [q.get("name") for q in r["req"] if isinstance(q, dict)] != r["cell"].split("+"):
            bad.append(f"pass {r['pass']} {r['cell']}: its requests are not the cell's names")
        seen[(r["pass"], r["cell"])] = seen.get((r["pass"], r["cell"]), 0) + 1
    want = expected_cells(n, passes)
    for key in want:
        if seen.get(key, 0) == 0:
            gaps.append(f"pass {key[0]} {key[1]}: missing")
        elif seen[key] > 1:
            gaps.append(f"pass {key[0]} {key[1]}: {seen[key]} times")
    for key in seen:
        if key not in want:
            gaps.append(f"pass {key[0]} {key[1]}: not a cell of this run (n {n}, {passes} passes)")
    return bad, gaps


def starved_verdict(rows, n, passes=2):
    """The "no request starved" line (docstring). Returns (verdict, lines): PASS, FAIL, INCONCLUSIVE or MALFORMED."""
    bad, gaps = cells_gaps(rows, n, passes)
    if bad:
        return "MALFORMED", bad
    lines, fail, inc = list(gaps), False, bool(gaps)
    cells = [r for r in rows if len(r["req"]) == n and len(r["req"]) > 1]
    for r in cells:
        rates = [q["out"] / q["decode_s"] for q in r["req"]]
        mean = sum(rates) / len(rates)
        low = [q["name"] for q, rate in zip(r["req"], rates) if rate < 0.5 * mean]
        unfinished = [q["name"] for q in r["req"] if q["finish"] not in FINISHED]
        tag = f"pass {r['pass']} {r['cell']}"
        if unfinished or low:
            lines.append(f"{tag}: FAIL (unfinished {unfinished}, under half the mean {mean:.1f} tok/s: {low})")
            fail = True
        elif r["peak_running"] < n:
            lines.append(f"{tag}: INCONCLUSIVE (at most {r['peak_running']} of {n} seen running at once)")
            inc = True
        else:
            lines.append(f"{tag}: slowest {min(rates):.1f}, fastest {max(rates):.1f}, mean {mean:.1f} tok/s, "
                         f"SUM {r['sum_tok_s']:.1f}, WALL {r['wall_tok_s']:.1f}")
    return ("FAIL" if fail else "INCONCLUSIVE" if inc else "PASS"), lines


def summary_rows(runs):
    """runs: [(label, rows)]. Returns the table as lists (pure). Per label and kind ("fresh", "repeated"):
    n, SUM and WALL (mean over the passes), slowest and fastest request tok/s, TTFT median and max (seconds)."""
    out = []
    for label, rows in runs:
        for kind, letter in (("fresh", "F"), ("repeated", "R")):
            cells = [r for r in rows if len(r["req"]) > 1 and r["req"][0]["name"][0] == letter]
            alone = [q["out"] / q["decode_s"] for r in rows if len(r["req"]) == 1 for q in r["req"] if q["name"][0] == letter]
            if not cells:
                continue
            rates = sorted(q["out"] / q["decode_s"] for r in cells for q in r["req"])
            ttft = sorted(q["ttft_s"] for r in cells for q in r["req"])
            out.append({
                "label": label, "kind": kind, "n": len(cells[0]["req"]), "passes": len(cells),
                "sum": round(sum(r["sum_tok_s"] for r in cells) / len(cells), 1),
                "wall": round(sum(r["wall_tok_s"] for r in cells) / len(cells), 1),
                "slowest": round(rates[0], 1), "fastest": round(rates[-1], 1),
                "ttft_median": round(ttft[len(ttft) // 2], 2), "ttft_max": round(ttft[-1], 2),
                "alone": round(sum(alone) / len(alone), 1) if alone else None,
            })
    return out


def summary(a):
    runs = []
    for item in a.runs:
        label, _, d = item.partition("=")
        runs.append((label, [json.loads(line) for line in open(os.path.join(d, "cells.jsonl"))]))
    table = summary_rows(runs)
    for kind in ("fresh", "repeated"):
        part = [t for t in table if t["kind"] == kind]
        if not part:
            continue
        first = part[0]["sum"]
        x3 = next((t["sum"] for t in part if t["label"] == "X3"), None)
        print(f"{kind} text, requests together (tok/s; time to first token in seconds):")
        for t in part:
            ratios = f"x{t['sum'] / first:.2f} of {part[0]['label']}" + ("" if x3 is None else f", x{t['sum'] / x3:.2f} of X3")
            print(f"  {t['label']:>4} {t['n']} together: SUM {t['sum']:7.1f} ({ratios}), WALL {t['wall']:7.1f}, per request "
                  f"{t['slowest']:.1f} to {t['fastest']:.1f}, one alone {t['alone']}, first token median {t['ttft_median']:.2f} "
                  f"longest {t['ttft_max']:.2f} ({t['passes']} passes)")


def judge(a):
    try:
        rows = [json.loads(line) for line in open(os.path.join(a.dir, "cells.jsonl"))]
    except (OSError, ValueError) as e:
        print(f"NO REQUEST STARVED at {a.n}: MALFORMED (the receipt cannot be read: {type(e).__name__})")
        sys.exit(5)
    v, lines = starved_verdict(rows, a.n)
    if v == "MALFORMED":
        print(f"NO REQUEST STARVED at {a.n}: MALFORMED")
        for line in lines:
            print("  " + line)
        sys.exit(5)
    for r in rows:
        if len(r["req"]) == 1:
            q = r["req"][0]
            print(f"ALONE pass {r['pass']} {q['name']}: {q['out'] / q['decode_s']:.1f} tok/s")
    print(f"NO REQUEST STARVED at {a.n}: {v}")
    for line in lines:
        print("  " + line)
    sys.exit({"PASS": 0, "FAIL": 1}.get(v, 2))


def expected_lineh(n):
    """[(step, names)] the rows lineh() writes for N (pure)."""
    out = []
    for group in names(n):
        for step in ("solo1", "together", "solo2"):
            out += [(step, tuple(group))] if step == "together" else [(step, (x,)) for x in group]
    return out


def lineh_verdict(rows, n):
    seen, bad = {}, []
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or not all(k in r for k in ("step", "names", "sha256", "peak_running")) \
                or not isinstance(r["names"], list) or not isinstance(r["sha256"], list) or len(r["names"]) != len(r["sha256"]):
            bad.append(f"row {i + 1}: not a line-h record (step, names, sha256 of each name, peak_running)")
            continue
        key = (r["step"], tuple(r["names"]))
        seen[key] = seen.get(key, 0) + 1
    if bad:
        return "MALFORMED", bad
    want = expected_lineh(n)
    gaps = [f"{s} {'+'.join(k)}: missing" for s, k in want if seen.get((s, k), 0) == 0]
    gaps += [f"{s} {'+'.join(k)}: {seen[(s, k)]} times" for s, k in want if seen.get((s, k), 0) > 1]
    gaps += [f"{s} {'+'.join(k)}: not a row of line h at {n}" for s, k in seen if (s, k) not in want]
    dig, peaks = {}, []
    for r in rows:
        for name, h in zip(r["names"], r["sha256"]):
            dig.setdefault(name, {})[r["step"]] = h
        if r["step"] == "together":
            peaks.append(r["peak_running"])
    lines, fail, inc = [], False, False
    for name in sorted(dig):
        d = dig[name]
        if set(d) != {"solo1", "together", "solo2"}:
            lines.append(f"{name}: INCONCLUSIVE (a step is missing)")
            inc = True
        elif d["solo1"] != d["solo2"]:
            lines.append(f"{name}: INCONCLUSIVE (the chat alone does not repeat)")
            inc = True
        elif d["together"] != d["solo1"]:
            lines.append(f"{name}: FAIL (together differs from alone)")
            fail = True
        else:
            lines.append(f"{name}: equal")
    if len(peaks) != 2 or min(peaks) < n:
        lines.append(f"{n} requests were not seen running together in both groups (peaks {peaks})")
        inc = True
    if len(dig) != 2 * n:
        lines.append(f"{len(dig)} chats recorded, want {2 * n}")
        inc = True
    if gaps:
        lines += gaps
        inc = True
    return ("FAIL" if fail else "INCONCLUSIVE" if inc else "PASS"), lines


def lineh(a):
    os.makedirs(a.out)
    P = prompts(a.texts)
    model = json.load(urllib.request.urlopen(a.url + "/v1/models", timeout=10))["data"][0]["id"]
    for group in names(a.n):
        for step in ("solo1", "together", "solo2"):
            for cell in ([group] if step == "together" else [(n,) for n in group]):
                req, peak = together(a.url, model, P, cell, lambda n: 300, salted=True)
                row = {"step": step, "names": list(cell), "sha256": [r["sha256"] for r in req],
                       "out": [r["out"] for r in req], "peak_running": peak, "utc": time.strftime("%FT%TZ", time.gmtime())}
                with open(os.path.join(a.out, "lineh.jsonl"), "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"LINEH {step} {'+'.join(cell)}: out {row['out']}, peak running {peak}", flush=True)
    v, lines = lineh_verdict([json.loads(line) for line in open(os.path.join(a.out, "lineh.jsonl"))], a.n)
    print(f"LINE H at {a.n}: {v}")
    for line in lines:
        print("  " + line)
    sys.exit({"PASS": 0, "FAIL": 1, "MALFORMED": 5}.get(v, 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--texts", required=True)
    r.add_argument("--url", default="http://127.0.0.1:18043")
    r.add_argument("--passes", type=int, default=2)
    j = sub.add_parser("judge")
    j.add_argument("--n", type=int, required=True)
    j.add_argument("--dir", required=True)
    h = sub.add_parser("lineh")
    h.add_argument("--n", type=int, required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--texts", required=True)
    h.add_argument("--url", default="http://127.0.0.1:18043")
    m = sub.add_parser("summary")
    m.add_argument("runs", nargs="+")
    a = ap.parse_args()
    try:
        {"run": run, "judge": judge, "lineh": lineh, "summary": summary}[a.cmd](a)
    except SystemExit:
        raise
    except Exception as e:   # never exit 1 on an error: the driver reads 1 as FAIL (2026-11-17)
        import traceback
        traceback.print_exc()
        print(f"slots_cells: {a.cmd} stopped on an unexpected error ({type(e).__name__}): exit 5")
        sys.exit(5)
