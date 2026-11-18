#!/usr/bin/env python3
"""timebudget_parse.py: step 0 of the twelve-card kernel plan, the time budget per pipeline stage and per phase.

  timebudget_parse.py log    SERVER_LOG [--chunk 4092] [--ladder LADDER_OUT]
  timebudget_parse.py traces TRACE_DIR --windows WINDOWS --classmap CLASSMAP [--stages 3] [--tp 4]
                             [--layers 14,14,12] [--chunk 4092] [--skip-rounds 2]
  timebudget_parse.py all    --log SERVER_LOG --traces TRACE_DIR --windows WINDOWS --classmap CLASSMAP [...]

Standard library only; run it as `python3 -I timebudget_parse.py ...`. It reads, never writes.

What it reads (the facilities of the build, READ from its code):
  * the server log (stdout and stderr of the server and its twelve workers, each worker line prefixed
    "(Worker_PP<s>_TP<t> pid=<n>) "):
      - VLLM_V41_ATTN_TIMING=1: "[attn timing] tokens <rows> ms: <label>=<ms> ..." (models/deepseek_v41/model.py,
        _treport): CUDA events between marks of every layer of ONE prompt forward of more than 256 rows, summed per
        label over the stage's layers; printed by TP rank 0 of each stage only. One device synchronize per forward.
      - VLLM_SM70_DEBUG=mtp: "SM70 V2 MTP profile interval_avg_ms calls=.. interval_calls=.. tokens=.. drafts=..
        <key>=<ms> ..." (v1/worker/runtime/profiling.py, StepProfiler.report): last stage only, every TP rank,
        averages over an interval of verify steps.
      - VLLM_NVFP4_8R_PREFILL_TRACE=1: "nvfp4_8r prefill trace: calls=50 rows=.. staged_experts=.. seconds_staged=..
        seconds_iq=.. seconds_total=.." (fused_moe/experts/nvfp4_8r_native.py, PrefillTrace.add), every rank.
      - "Failed to start profiler" / "Failed to stop profiler" (profiler/wrapper.py): the torch profiler only WARNS.
  * the torch profiler's Kineto traces (profiler-config in the profile; one
    "dp0_pp<s>_tp<t>_dcp0_ep<e>_rank<r>.<ns>.pt.trace.json.gz" per worker per /start_profile../stop_profile cycle),
    with the per-step annotation "execute_context_<n>(<tok>)_generation_<n>(<tok>)" (v1/worker/gpu_worker.py,
    annotate_profile), CUPTI kernel records on the GPU clock.
  * WINDOWS: the arm's list of profiler cycles in order, one line each: "<cycle> <label> <requests> <ok|short>",
    label c1 | c3 | c4 | read. "short" = fewer requests ran at the window's end than at its start.
  * LADDER_OUT: ctx_ladder.py's output ("LADDER <tag>: prompt P tok, first token F s, read R tok/s, ...").

Every cell that cannot be filled prints INCONCLUSIVE with the reason; nothing defaults to zero.
Exit: 0 every wanted cell filled and every check PASS; 3 a cell INCONCLUSIVE or a check not PASS;
4 an input cannot be parsed (not a server log, a malformed timing line, an unreadable trace); 2 usage.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import json
import os
import re
import statistics
import sys

EXIT_OK, EXIT_USAGE, EXIT_INCONCLUSIVE, EXIT_UNPARSEABLE = 0, 2, 3, 4

# Proposed validity numbers (the arm owner fixes them; see REPORT.md section 3). Named so a change is one line.
PROPOSED_UNCLASSIFIED_MAX = 0.02    # unclassified kernel time / a stage's busy time, per cell (ktrace V4's tolerance)
PROPOSED_COVERAGE_C1_MIN = 0.85     # c1: sum over stages of busy / round period (0.915 MEASURED on two stages)
PROPOSED_TRACE_VS_CLIENT = 0.02     # c1: mean trace round period against the client's profiled round (ktrace V2)
PROPOSED_PROFILER_COST_MAX = 0.06   # c1: profiled round over unprofiled round of the same boot, minus one
PROPOSED_UNMATCHED_MAX = 0.10       # c1: rounds whose later stages show not exactly one decode step
PROPOSED_MIN_ROUNDS = 30            # decode rounds per cell
PROPOSED_MIN_CHUNKS = 3             # prompt chunks per cell
PROPOSED_READ_SHARE = 0.97          # T8: the untimed-trace read against the reference read (--ref-read)

PREFIX_RE = re.compile(r"^\(Worker(?:_PP(\d+))?(?:_TP(\d+))? pid=\d+\) ")
ATTN_TAG = "[attn timing]"
ATTN_RE = re.compile(r"\[attn timing\] tokens (\d+) ms: (.*)$")
ATTN_KV = re.compile(r"(\S+?)=(\d+)(?=\s|$)")
MTP_TAG = "SM70 V2 MTP profile interval_avg_ms"
MTP_RE = re.compile(r"SM70 V2 MTP profile interval_avg_ms calls=(\d+) interval_calls=(\d+) tokens=(\d+) drafts=(\d+) (.*)$")
MTP_KV = re.compile(r"(\w+)=(-?\d+\.\d+)(?=\s|$)")
NV_TAG = "nvfp4_8r prefill trace:"
NV_RE = re.compile(r"nvfp4_8r prefill trace: calls=(\d+) rows=(\d+) staged_experts=(\d+) seconds_staged=([\d.]+) "
                   r"seconds_iq=([\d.]+) seconds_total=([\d.]+)$")
PROF_FAIL_RE = re.compile(r"Failed to (start|stop) profiler")
LADDER_RE = re.compile(r"^LADDER (\S+): prompt (\d+) tok, first token ([\d.]+) s, read (\d+) tok/s, "
                       r"decode ([\d.]+|nan) tok/s, out (\d+)$")
STEP_RE = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$")
FILE_RE = re.compile(r"^dp\d+_pp(\d+)_tp(\d+)_dcp\d+_ep\d+_rank(\d+)\.(\d+)\.pt\.trace\.json(\.gz)?$")
WIN_RE = re.compile(r"^(\d+) (c1|c3|c4|c2|read|readL|readF|turnS|turnL|turnW|decL|decW|dec4L) (\d+) (ok|short)$")
PREFIX_TAG = "VLLM_V41_PREFIX_LOG request "
PFX_RE = re.compile(r"VLLM_V41_PREFIX_LOG request (\S+) slot (-?\d+) prompt (\d+) resumed (\d+) pool_hit (-?\d+) "
                       r"hit (-?\d+) decision (\S+) reason (\S+) .*?\brewind (-?\d+)\b")
RT_GROUPS = (("sync", re.compile(r"Synchronize")), ("free", re.compile(r"^cu(da)?Free|^cudaFreeAsync")),
             ("malloc", re.compile(r"^cu(da)?Malloc|^cuMemAlloc|^cuMemCreate|^cuMemMap")),
             ("memcpy_api", re.compile(r"^cu(da)?Memcpy|^cu(da)?Memset")),
             ("graph_launch", re.compile(r"^cu(da)?GraphLaunch")),
             ("kernel_launch", re.compile(r"^cu(da)?LaunchKernel|^cuLaunch")),
             ("event", re.compile(r"Event(Record|Query|Create|Destroy)")))

# attn-timing labels -> the plan's phases (two labels hold the TP sums: they are not timed apart there)
ATTN_PHASE = {"engram": "engram", "hc_pre": "hc_norm", "hc_mid": "hc_norm", "hc_post": "hc_norm",
              "a.proj": "dense (q/kv projections)", "a.window_insert": "attention", "a.compressor": "attention",
              "a.indexer": "index", "a.attend": "attention", "a.out_proj": "dense + TP sum (output projection)",
              "ffn(moe+shared)": "experts + shared + router + TP sum"}


class Unparseable(Exception):
    pass


def q(xs, p):
    """Nearest-rank percentile of a non-empty list (p in 0..100)."""
    s = sorted(xs)
    k = max(0, min(len(s) - 1, -(-len(s) * p // 100) - 1))
    return s[int(k)]


def stat(xs):
    return f"n={len(xs)} median={statistics.median(xs):.3f} p90={q(xs, 90):.3f}"


class Report:
    def __init__(self):
        self.inconclusive = []
        self.checks = []

    def inc(self, cell, why):
        self.inconclusive.append(cell)
        print(f"INCONCLUSIVE {cell}: {why}")

    def check(self, name, status, text):
        self.checks.append((name, status))
        print(f"CHECK {name} {status}: {text}")

    def exit_code(self):
        bad = [n for n, s in self.checks if s != "PASS"]
        print(f"VERDICT {'COMPLETE' if not self.inconclusive and not bad else 'INCONCLUSIVE'}: "
              f"{len(self.inconclusive)} cell(s) inconclusive, {len(bad)} check(s) not PASS"
              + (f" ({', '.join(bad)})" if bad else ""))
        return EXIT_OK if not self.inconclusive and not bad else EXIT_INCONCLUSIVE


# ---------------------------------------------------------------- server log

def read_log(path):
    """Every timing line of the log, parsed; raises Unparseable on a line that carries a tag but not its format."""
    out = {"attn": [], "mtp": [], "nv": [], "prefix": [], "prof_fail": 0, "workers": set(), "lines": 0}
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            out["lines"] += 1
            m = PREFIX_RE.match(line)
            stage = tp = None
            if m:
                stage = int(m.group(1)) if m.group(1) is not None else 0
                tp = int(m.group(2)) if m.group(2) is not None else 0
                out["workers"].add((stage, tp))
            if PROF_FAIL_RE.search(line):
                out["prof_fail"] += 1
            if PREFIX_TAG in line:
                a = PFX_RE.search(line)
                if not a:
                    raise Unparseable(f"{path}:{out['lines']}: a VLLM_V41_PREFIX_LOG line not in the engine's format")
                out["prefix"].append({"id": a.group(1), "slot": int(a.group(2)), "prompt": int(a.group(3)),
                                      "pool_hit": int(a.group(5)), "hit": int(a.group(6)), "decision": a.group(7),
                                      "reason": a.group(8), "rewind": int(a.group(9))})
                continue
            if ATTN_TAG in line:
                a = ATTN_RE.search(line)
                if not a or m is None:
                    raise Unparseable(f"{path}:{out['lines']}: an [attn timing] line not in the engine's format")
                kv = ATTN_KV.findall(a.group(2))
                if not kv or " ".join(f"{k}={v}" for k, v in kv) != a.group(2).strip():
                    raise Unparseable(f"{path}:{out['lines']}: [attn timing] segments not label=ms")
                out["attn"].append({"stage": stage, "tp": tp, "rows": int(a.group(1)),
                                    "ms": {k: float(v) for k, v in kv}})
            elif MTP_TAG in line:
                a = MTP_RE.search(line)
                if not a or m is None:
                    raise Unparseable(f"{path}:{out['lines']}: an MTP profile line not in the engine's format")
                out["mtp"].append({"stage": stage, "tp": tp, "interval_calls": int(a.group(2)),
                                   "tokens": int(a.group(3)), "drafts": int(a.group(4)),
                                   "ms": {k: float(v) for k, v in MTP_KV.findall(a.group(5))}})
            elif NV_TAG in line:
                a = NV_RE.search(line)
                if not a or m is None:
                    raise Unparseable(f"{path}:{out['lines']}: an nvfp4_8r prefill trace line not in the engine's format")
                out["nv"].append({"stage": stage, "tp": tp, "calls": int(a.group(1)), "rows": int(a.group(2)),
                                  "staged": int(a.group(3)), "s_staged": float(a.group(4)),
                                  "s_iq": float(a.group(5)), "s_total": float(a.group(6))})
    if not out["workers"] and not out["prefix"]:
        raise Unparseable(f"{path}: no worker-prefixed line: not a server log of this engine")
    return out


def read_ladder(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = LADDER_RE.match(line.strip())
            if m:
                rows.append({"tag": m.group(1), "prompt": int(m.group(2)), "first_s": float(m.group(3)),
                             "read_tps": int(m.group(4))})
    if not rows:
        raise Unparseable(f"{path}: no LADDER line")
    return rows


def report_log(lg, rep, n_stages, chunk, ladder, a=None):
    print(f"LOG workers seen {len(lg['workers'])}: "
          + " ".join(f"PP{s}TP{t}" for s, t in sorted(lg["workers"])))
    if lg["prof_fail"]:
        rep.check("profiler_start_stop", "FAIL", f"{lg['prof_fail']} 'Failed to start/stop profiler' line(s)")
    # ---- one prompt chunk per stage, from the attention timers
    # A forward that returns without its report (the skipped-prefill path, model.py _prefill_skipped) leaves its
    # marks for the next report, which then carries the wall time between the two forwards (MEASURED on the
    # eight-card log of 2026-11-05: a.out_proj=60131 ms in one 8,188-row line of the stage that skips). Such a line
    # is longer than three times any other stage's median line of the same row count.
    med_by = collections.defaultdict(dict)
    for s in range(n_stages):
        for rows in {r["rows"] for r in lg["attn"] if r["stage"] == s}:
            med_by[rows][s] = statistics.median(sum(r["ms"].values()) for r in lg["attn"]
                                                if r["stage"] == s and r["rows"] == rows)
    piled = 0
    for r in lg["attn"]:
        others = [v for s, v in med_by[r["rows"]].items() if s != r["stage"]]
        if others and sum(r["ms"].values()) > 3 * max(others):
            piled += 1
    if lg["attn"]:
        rep.check("attn_lines_whole", "PASS" if piled == 0 else "FAIL",
                  f"{piled} [attn timing] line(s) longer than 3x every other stage's median at the same row count "
                  f"(marks of an unreported forward carried into the next line)")
    spans = {}
    for s in range(n_stages):
        rows = [r for r in lg["attn"] if r["stage"] == s and r["rows"] == chunk]
        other = collections.Counter(r["rows"] for r in lg["attn"] if r["stage"] == s and r["rows"] != chunk)
        cell = f"prefill_chunk stage{s} (attn timing, rows {chunk})"
        if len(rows) < PROPOSED_MIN_CHUNKS:
            rep.inc(cell, f"{len(rows)} [attn timing] line(s) of {chunk} rows from stage {s} (need "
                          f"{PROPOSED_MIN_CHUNKS}); other row counts seen: {dict(other) or 'none'}")
            continue
        labels = sorted({k for r in rows for k in r["ms"]})
        sp = [sum(r["ms"].values()) for r in rows]
        spans[s] = statistics.median(sp)
        print(f"CELL {cell}: lines {len(rows)}; forward span ms {stat(sp)}")
        for k in sorted(labels, key=lambda k: -statistics.median([r["ms"].get(k, 0.0) for r in rows])):
            xs = [r["ms"][k] for r in rows if k in r["ms"]]
            if len(xs) != len(rows):
                rep.inc(f"{cell} {k}", f"label present in {len(xs)} of {len(rows)} lines")
                continue
            print(f"PHASE {cell} {k:18s} [{ATTN_PHASE.get(k, 'UNMAPPED LABEL')}] ms {stat(xs)}")
    # the wall per chunk from an untimed-trace read of the same boot
    if ladder and spans:
        rd = ladder[-1]
        wall_chunk = chunk / rd["read_tps"] * 1000.0
        mx = max(spans.values())
        print(f"COVERAGE prefill_chunk (log): slowest stage median span {mx:.1f} ms; wall per {chunk}-row chunk "
              f"{wall_chunk:.1f} ms from LADDER {rd['tag']} (prompt {rd['prompt']} tok at {rd['read_tps']} tok/s); "
              f"sum of the stages' spans {sum(spans.values()):.1f} ms; slowest/wall {mx / wall_chunk:.3f}")
        status = "PASS" if mx <= 1.02 * wall_chunk else "FAIL"
        rep.check("prefill_span_le_wall", status, f"a stage's timed forward ({mx:.1f} ms) cannot exceed the wall per "
                  f"chunk ({wall_chunk:.1f} ms) by more than 2 %")
    if ladder and a is not None:
        rd = ladder[0]
        rep.check("attention_timer_cost", "PASS" if rd["read_tps"] >= PROPOSED_READ_SHARE * a.ref_read else "FAIL",
                  f"untimed-trace read {rd['read_tps']} tok/s ({rd['prompt']} tokens, LADDER {rd['tag']}) against "
                  f"{a.ref_read:.0f} tok/s of the {a.ref_name}: {100 * rd['read_tps'] / a.ref_read:.1f} % "
                  f"(floor {100 * PROPOSED_READ_SHARE:.0f} %)")
    # ---- last-stage verify step split, if the boot had VLLM_SM70_DEBUG=mtp
    if lg["mtp"]:
        keys = sorted({k for r in lg["mtp"] for k in r["ms"]})
        for (s, t) in sorted({(r["stage"], r["tp"]) for r in lg["mtp"]}):
            rs = [r for r in lg["mtp"] if (r["stage"], r["tp"]) == (s, t)]
            print(f"CELL mtp_profile PP{s}TP{t}: intervals {len(rs)} (steps {sum(r['interval_calls'] for r in rs)})")
            for k in keys:
                xs = [r["ms"][k] for r in rs if k in r["ms"]]
                if xs:
                    print(f"PHASE mtp_profile PP{s}TP{t} {k:26s} interval-mean ms {stat(xs)}")
    else:
        print("NOTE mtp_profile: no 'SM70 V2 MTP profile' line (VLLM_SM70_DEBUG=mtp was not on): not a zero")
    if lg["prefix"]:
        d = collections.Counter(f"{r['decision']}/{r['reason']}" for r in lg["prefix"])
        hits = [r for r in lg["prefix"] if r["decision"] == "hit"]
        print(f"PREFIX admissions {len(lg['prefix'])}: decisions {dict(d)}")
        if hits:
            new = [r["prompt"] - r["hit"] for r in hits]
            print(f"PREFIX hits: prompt tokens {stat([r['prompt'] for r in hits])}; new rows (prompt - hit) {stat(new)}; "
                  f"rewind {stat([r['rewind'] for r in hits])}")
    else:
        print("NOTE prefix_log: no VLLM_V41_PREFIX_LOG line (one slot, or the switch off): not a zero")
    if lg["nv"]:
        for (s, t) in sorted({(r["stage"], r["tp"]) for r in lg["nv"]}):
            rs = [r for r in lg["nv"] if (r["stage"], r["tp"]) == (s, t)]
            calls = sum(r["calls"] for r in rs)
            print(f"CELL nvfp4_prefill_trace PP{s}TP{t}: windows {len(rs)} calls {calls} rows {sum(r['rows'] for r in rs)} "
                  f"staged experts {sum(r['staged'] for r in rs)}; ms per call: total "
                  f"{1000 * sum(r['s_total'] for r in rs) / calls:.3f}, staged {1000 * sum(r['s_staged'] for r in rs) / calls:.3f}, "
                  f"iq {1000 * sum(r['s_iq'] for r in rs) / calls:.3f} (synchronized marks)")
    else:
        print("NOTE nvfp4_prefill_trace: no line (VLLM_NVFP4_8R_PREFILL_TRACE=1 was not on): not a zero")


# ---------------------------------------------------------------- traces

class ClassMap:
    def __init__(self, path):
        with open(path) as f:
            d = json.load(f)
        self.name = d["name"]
        self.anchor = re.compile(d["anchor"])
        self.dmv = re.compile(d["dmv"])
        self.graph_launch = re.compile(d["graph_launch"])
        self.phases = list(d["phases"])
        self.rules = [(re.compile(rx), c) for rx, c in d["classes"]]
        for _, c in self.rules:
            if c not in self.phases and c != "dense_or_drafter":
                raise ValueError(f"{path}: class {c!r} is not a phase")

    def by_name(self, n):
        for rx, c in self.rules:
            if rx.search(n):
                return c
        return "unclassified"


def kname(raw):
    return raw[5:] if raw.startswith("void ") else raw


def anchors_of(cm, ks):
    """Attention anchors in a time-sorted kernel list; anchors with no dmv between them count once (one layer)."""
    n, seen_dmv = 0, True
    for k in ks:
        if cm.dmv.search(k[2]):
            seen_dmv = True
        elif cm.anchor.search(k[2]):
            if seen_dmv:
                n += 1
            seen_dmv = False
    return n


def union_us(iv):
    tot, cs, ce = 0.0, None, None
    for s, e in sorted(iv):
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        tot += ce - cs
    return tot


def digest_trace(path, cm, is_last_stage):
    """One rank's trace of one cycle -> a list of step digests (aggregates only; the trace is dropped)."""
    try:
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt") as f:
            ev = json.load(f)["traceEvents"]
    except (OSError, ValueError, KeyError) as e:
        raise Unparseable(f"{path}: {type(e).__name__}: {e}")
    rt = {}
    api = []  # (ts, dur, group) of every runtime / driver call
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            c = e.get("args", {}).get("correlation")
            if c is not None:
                rt[c] = (e["name"], float(e["ts"]))
            g = next((g for g, rx in RT_GROUPS if rx.search(e.get("name", ""))), "other_api")
            api.append((float(e["ts"]), float(e.get("dur", 0.0)), g))
    api.sort()
    api_ts = [x[0] for x in api]
    kernels = []  # (ts, end, name, launch_ts, corr, is_graph)
    for e in ev:
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X":
            c = e.get("args", {}).get("correlation")
            r = rt.get(c)
            ts = float(e["ts"])
            kernels.append((ts, ts + float(e.get("dur", 0.0)), kname(e["name"]), r[1] if r else None, c,
                            bool(r and cm.graph_launch.search(r[0])), int(e.get("args", {}).get("bytes", 0) or 0)))
    steps = []
    for e in ev:
        if e.get("cat") == "user_annotation":
            m = STEP_RE.match(e.get("name", ""))
            if m:
                steps.append((float(e["ts"]), float(e["ts"]) + float(e["dur"]), int(m.group(1)), int(m.group(2)),
                              int(m.group(3)), int(m.group(4))))
    del ev
    steps.sort()
    launched = sorted((k for k in kernels if k[3] is not None), key=lambda k: k[3])
    lts = [k[3] for k in launched]
    out = []
    # A step owns every kernel launched from its annotation's start until the next step's annotation starts: the
    # annotation wraps execute_model only, and the sampler, the lm head and the drafter are launched after it
    # (sample_tokens), still before the next step (MEASURED on the 2026-11-14 traces: with the annotation span alone
    # the last stage showed no drafter at all).
    # Steps with no request ("execute_context_0(0)_generation_0(0)", MEASURED between every two decode steps on the
    # 2026-11-14 traces) own nothing: a real step owns every launch until the next real step's annotation.
    real = [st for st in steps if st[2] or st[4]]
    for i, (a, b, ctx_n, ctx_t, gen_n, gen_t) in enumerate(real):
        nxt = real[i + 1][0] if i + 1 < len(real) else float("inf")
        ks = sorted(launched[bisect.bisect_left(lts, a):bisect.bisect_left(lts, nxt)])
        if not ks:
            continue
        kind = "decode" if ctx_n == 0 and gen_n >= 1 else ("prefill" if gen_n == 0 else "mixed")
        # the target forward: the graph launch holding the most attention anchors
        groups = collections.defaultdict(list)
        for k in ks:
            if k[5]:
                groups[k[4]].append(k)
        tg, tg_n = None, 0
        for c, g in groups.items():
            n = anchors_of(cm, g)
            if n > tg_n:
                tg, tg_n = c, n
        tg_end = max(k[1] for k in groups[tg]) if tg is not None else None
        by = collections.Counter()
        unc = collections.Counter()
        sums = []
        comp_iv = []
        for k in ks:
            c = cm.by_name(k[2])
            in_tg = tg is not None and k[4] == tg
            after = (kind == "decode" and is_last_stage and tg_end is not None and not in_tg and k[0] >= tg_end)
            if after:
                if c == "coll_gather" or cm.dmv.search(k[2]):
                    c = "head"
                elif c in ("sampling", "bookkeeping", "memcpy", "handoff"):
                    pass
                elif c == "unclassified" and not k[5]:
                    pass
                else:
                    c = "drafter"
            elif c == "dense_or_drafter":
                c = "dense"
            if c == "coll_gather" and not k[5] and not after:
                # an eager gather outside the forward: the sampled tokens coming back from the last stage; its kernel
                # spins until that stage sends (MEASURED 2026-11-14 traces: 14 ms of a 25.8 ms round on stage 0)
                c = "handoff"
            dur = k[1] - k[0]
            by[c] += dur
            if c == "unclassified":
                unc[k[2][:100]] += dur
            if c != "handoff":
                comp_iv.append((k[0], k[1]))
            if c == "coll_tp_sum" and kind == "prefill":
                sums.append((k[0], dur))
        if not comp_iv:
            continue
        rtg = collections.Counter()
        for ts_, du_, g in api[bisect.bisect_left(api_ts, a):bisect.bisect_left(api_ts, nxt)]:
            rtg[g] += du_
        mc = collections.Counter()
        for k in ks:
            if k[2].startswith("Memcpy"):
                kind_ = k[2].split()[1] if len(k[2].split()) > 1 else "?"
                mc[f"{kind_}_us"] += k[1] - k[0]
                mc[f"{kind_}_bytes"] += k[6]
        out.append({"kind": kind, "ctx_tok": ctx_t, "gen_n": gen_n, "gen_tok": gen_t,
                    "cpu_start": a, "cpu_dur": b - a, "cpu_to_next": (nxt - a) if nxt != float("inf") else None,
                    "api": dict(rtg), "memcpy": dict(mc), "gpu_end_all": max(k[1] for k in ks),
                    "start": min(s for s, _ in comp_iv), "end": max(e for _, e in comp_iv),
                    "busy": union_us(comp_iv), "by": dict(by), "unc": dict(unc), "anchors": tg_n, "sums": sums})
    return out


def load_cycles(trace_dir, world):
    files = collections.defaultdict(list)
    names = os.listdir(trace_dir) if os.path.isdir(trace_dir) else []
    for n in names:
        m = FILE_RE.match(n)
        if m:
            files[int(m.group(3))].append((int(m.group(4)), int(m.group(1)), int(m.group(2)), os.path.join(trace_dir, n)))
    if sorted(files) != list(range(world)):
        return None, f"trace files for ranks {sorted(files)} in {trace_dir}, need 0..{world - 1}"
    counts = {r: len(v) for r, v in files.items()}
    if len(set(counts.values())) != 1:
        return None, f"unequal trace-file counts per rank {counts}"
    n = next(iter(counts.values()))
    return [{r: sorted(files[r])[i] for r in files} for i in range(n)], None


def read_windows(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            m = WIN_RE.match(line.strip())
            if not m:
                raise Unparseable(f"{path}: window line {line.strip()!r} is not '<cycle> <label> <requests> <ok|short>'")
            rows.append((int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)))
    if [r[0] for r in rows] != list(range(1, len(rows) + 1)):
        raise Unparseable(f"{path}: cycles are not numbered 1..n in order")
    return rows


def phase_lines(cell, steps, phases, rep):
    """Per phase over steps (ms): median and p90 of the step sums; unclassified share of busy."""
    busy = [st["busy"] / 1000 for st in steps]
    span = [(st["end"] - st["start"]) / 1000 for st in steps]
    print(f"CELL {cell}: steps {len(steps)}; GPU span ms {stat(span)}; busy ms {stat(busy)}")
    for p in phases:
        xs = [st["by"].get(p, 0.0) / 1000 for st in steps]
        if any(xs):
            print(f"PHASE {cell} {p:13s} ms {stat(xs)}")
    unc = sum(st["by"].get("unclassified", 0.0) for st in steps)
    tot = sum(st["busy"] for st in steps)
    share = unc / tot if tot else 1.0
    names = collections.Counter()
    for st in steps:
        names.update(st["unc"])
    rep.check(f"unclassified {cell}", "PASS" if share <= PROPOSED_UNCLASSIFIED_MAX else "FAIL",
              f"{100 * share:.2f} % of busy (ceiling {100 * PROPOSED_UNCLASSIFIED_MAX:.0f} %)"
              + ("; top: " + "; ".join(f"{n} {v / 1000:.1f} ms" for n, v in names.most_common(4)) if names else ""))
    return statistics.median(busy)


def report_traces(a, rep):
    cm = ClassMap(a.classmap)
    world = a.stages * a.tp
    layers = [int(x) for x in a.layers.split(",")]
    if len(layers) != a.stages:
        raise SystemExit("--layers needs one count per stage")
    wins = read_windows(a.windows)
    cycles, why = load_cycles(a.traces, world)
    print(f"TRACES {a.traces}: classmap {cm.name}; {len(wins)} window line(s)")
    if cycles is None:
        for _, label, _, _ in wins:
            rep.inc(f"{label} (all stages)", why)
        return
    if len(cycles) != len(wins):
        rep.check("cycles", "FAIL", f"{len(cycles)} profiler cycle(s) per rank, the windows file lists {len(wins)}")
        return
    rep.check("cycles", "PASS", f"{len(cycles)} cycle(s), {world} of {world} workers wrote one trace file in each")
    # digest every (cycle, rank); stage and TP from the file name
    dig = {}
    for (ci, label, nreq, ok), cyc in zip(wins, cycles):
        for r, (_, s, t, p) in sorted(cyc.items()):
            dig[(ci, s, t)] = digest_trace(p, cm, s == a.stages - 1)
            print(f"READ cycle {ci} {label} PP{s}TP{t}: {len(dig[(ci, s, t)])} step(s)", flush=True)
    by_label = collections.OrderedDict()
    for ci, label, nreq, ok in wins:
        by_label.setdefault(label, []).append((ci, nreq, ok))
    turns = {}
    if a.turns:
        with open(a.turns) as f:
            for line in f:
                r = json.loads(line)
                if "metrics_delta" in r:
                    turns[r["cycle"]] = r
    prefix = {}
    if getattr(a, "log", None) and a.cmd == "all":
        # the engine's request id is the API's id plus "-<8 hex>" (MEASURED, window t1: chatcmpl-9688985b99c729aa in
        # the reply, chatcmpl-9688985b99c729aa-affe1bda in the prefix line): keyed by both
        for r in read_log(a.log)["prefix"]:
            prefix[r["id"]] = r
            m = re.match(r"^(.*)-[0-9a-f]{8}$", r["id"])
            if m:
                prefix.setdefault(m.group(1), r)
    for label, cl in by_label.items():
        bad = [ci for ci, _, ok in cl if ok != "ok"]
        if bad:
            rep.inc(f"{label} cycles {bad}", "fewer requests ran at the cycle's close than it was for: left out")
            cl = [x for x in cl if x[2] == "ok"]
            if not cl:
                continue
        if label in ("read", "readL", "readF"):
            report_prefill(a, cm, dig, cl, layers, rep, label)
            continue
        if label in ("decL", "decW"):
            continue   # decode cells of their own (below)
        if label in ("turnS", "turnL", "turnW"):
            for nreq in sorted({n for _, n, _ in cl}):
                report_turns(a, cm, dig, label, [x for x in cl if x[1] == nreq], turns, prefix, rep)
            continue
        report_decode(a, cm, dig, label, cl, layers, rep)
    # decode at a grown context: the decode steps of the turn cycles (after each turn's prompt step) and dec4L
    # L = a chat of ~128,000 tokens, W = ~250,000; the digit = requests running. Since turn_probe's second version the
    # alone decode cells are the decL / decW cycles (300 tokens out); the turns answer 48 tokens (within the ring's
    # reach), so their decode rows are few. The first version's turn cycles (200 out) still give L1 / W1 when no
    # decL / decW cycle exists.
    dec_labels = {lab for _, lab, _, _ in wins}
    l1 = ("decL",) if "decL" in dec_labels else ("turnS", "turnL")
    w1 = ("decW",) if "decW" in dec_labels else ("turnW",)
    for name, labs, nreq in (("L1", l1, 1), ("W1", w1, 1), ("L2", ("turnS", "turnL"), 2)):
        cl = [(ci, n, ok) for ci, lab, n, ok in wins if lab in labs and n == nreq and ok == "ok"]
        if cl:
            report_decode(a, cm, dig, name, cl, layers, rep, skip=1)


def report_decode(a, cm, dig, label, cl, layers, rep, skip=None):
    nreq = cl[0][1]
    skip_rounds = a.skip_rounds if skip is None else skip
    per_stage = {s: [] for s in range(a.stages)}
    periods, cover, unmatched, nrounds = [], [], 0, 0
    struct_bad = 0
    gen = collections.Counter()
    for ci, _, _ in cl:
        st0 = [x for x in dig[(ci, 0, 0)] if x["kind"] == "decode"]
        gen.update(x["gen_n"] for x in st0)
        st0 = st0[skip_rounds:]
        for s in range(a.stages):
            ds = [x for x in dig[(ci, s, 0)] if x["kind"] == "decode"][skip_rounds:]
            per_stage[s].extend(ds[:-1] if s == 0 else ds)
            struct_bad += sum(1 for x in ds if x["anchors"] != layers[s])
        for i in range(len(st0) - 1):
            t0, t1 = st0[i]["start"], st0[i + 1]["start"]
            periods.append((t1 - t0) / 1000)
            nrounds += 1
            if nreq == 1:
                busy = st0[i]["busy"]
                good = True
                for s in range(1, a.stages):
                    c = [x for x in dig[(ci, s, 0)] if x["kind"] == "decode" and t0 <= x["start"] < t1]
                    if len(c) != 1:
                        good = False
                        break
                    busy += c[0]["busy"]
                if good:
                    cover.append(busy / (t1 - t0))
                else:
                    unmatched += 1
    print(f"WINDOW {label}: cycles {[c for c, _, _ in cl]}, {nreq} request(s) running; decode steps by requests per "
          f"step on PP0TP0: {dict(gen)}")
    if nrounds < PROPOSED_MIN_ROUNDS:
        rep.inc(f"decode_{label} (all stages)", f"{nrounds} round(s) on PP0TP0 after skipping {skip_rounds} per "
                                                f"cycle (need {PROPOSED_MIN_ROUNDS})")
        return
    print(f"CELL decode_{label} round period ms (PP0TP0 step start to step start) {stat(periods)}")
    busy_med = {}
    for s in range(a.stages):
        if not per_stage[s]:
            rep.inc(f"decode_{label} stage{s}", "no decode step on this stage's TP rank 0")
            continue
        busy_med[s] = phase_lines(f"decode_{label} stage{s}", per_stage[s], cm.phases, rep)
        sk = tp_skew(a, dig, cl, s, "decode")
        if sk is not None:
            print(f"TPSKEW decode_{label} stage{s}: busy max-min over the four TP ranks ms {stat(sk)}")
    rep.check(f"structure decode_{label}", "PASS" if struct_bad == 0 else "FAIL",
              f"{struct_bad} decode step(s) whose target graph does not hold the stage's layer count of attention anchors")
    if label == "c1" and a.runs_json:
        with open(a.runs_json) as f:
            rj = json.load(f)
        prof = [r["round_ms"] for r in rj["profiled"]]
        unprof = [r["round_ms"] for r in rj["unprofiled"]]
        tm = statistics.fmean(periods)
        cm_ = statistics.fmean(prof)
        rep.check(f"trace_vs_client decode_{label}", "PASS" if abs(tm - cm_) <= PROPOSED_TRACE_VS_CLIENT * cm_ else "FAIL",
                  f"mean trace round {tm:.3f} ms, client's profiled rounds {', '.join(f'{x:.2f}' for x in prof)} ms "
                  f"(mean {cm_:.3f}; tolerance {100 * PROPOSED_TRACE_VS_CLIENT:.0f} %)")
        cost = statistics.median(prof) / statistics.median(unprof) - 1
        rep.check(f"profiler_cost decode_{label}", "PASS" if cost <= PROPOSED_PROFILER_COST_MAX else "FAIL",
                  f"profiled round median {statistics.median(prof):.3f} ms against unprofiled {statistics.median(unprof):.3f} "
                  f"ms of the same boot: +{100 * cost:.2f} % (ceiling {100 * PROPOSED_PROFILER_COST_MAX:.0f} %)")
        shas = {r["tok_sha"] for r in rj["profiled"] + rj["unprofiled"]}
        rep.check(f"same_tokens decode_{label}", "PASS" if len(shas) == 1 else "FAIL",
                  f"{len(shas)} distinct token digest(s) over the profiled and unprofiled requests")
    elif label == "c1":
        rep.inc(f"profiler_cost decode_{label}", "no --runs-json: the client times of the same boot are not known")
    if nreq == 1:
        share = unmatched / nrounds
        rep.check(f"rounds_matched decode_{label}", "PASS" if share <= PROPOSED_UNMATCHED_MAX else "FAIL",
                  f"{unmatched} of {nrounds} round(s) without exactly one decode step on each later stage")
        if cover:
            mc = statistics.median(cover)
            print(f"COVERAGE decode_{label}: sum over stages of busy / round period, median {mc:.3f} "
                  f"(p10 {q(cover, 10):.3f}); the rest is launch gaps, host time and hand-off transfer")
            rep.check(f"coverage decode_{label}", "PASS" if mc >= PROPOSED_COVERAGE_C1_MIN else "FAIL",
                      f"median {mc:.3f} (floor {PROPOSED_COVERAGE_C1_MIN})")
        else:
            rep.inc(f"coverage decode_{label}", "no round could be matched across the stages")
    elif busy_med:
        pm = statistics.median(periods)
        s_max = max(busy_med, key=busy_med.get)
        print(f"COVERAGE decode_{label}: busiest stage {s_max} median busy {busy_med[s_max]:.3f} ms of a {pm:.3f} ms "
              f"period ({busy_med[s_max] / pm:.3f}); per stage "
              + ", ".join(f"{s}: {v / pm:.3f}" for s, v in sorted(busy_med.items())))


def tp_skew(a, dig, cl, s, kind, ctx_tok=None):
    out = []
    for ci, _, _ in cl:
        lists = []
        for t in range(a.tp):
            xs = [x for x in dig[(ci, s, t)] if x["kind"] == kind and (ctx_tok is None or x["ctx_tok"] == ctx_tok)]
            lists.append(xs)
        if len({len(x) for x in lists}) != 1:
            return None
        for i in range(len(lists[0])):
            b = [lists[t][i]["busy"] / 1000 for t in range(a.tp)]
            out.append(max(b) - min(b))
    return out or None


def ran_layers(x):
    """A prompt step that ran the stage's layers: expert kernels or TP sums (a skipped step has neither)."""
    return bool(x["by"].get("experts") or x["by"].get("coll_tp_sum"))


def report_prefill(a, cm, dig, cl, layers, rep, label="read"):
    print(f"WINDOW {label}: cycles {[c for c, _, _ in cl]}; prompt chunks of {a.chunk} rows")
    seen = collections.Counter()
    skipped = collections.Counter()
    busy_med = {}
    starts0 = []
    for s in range(a.stages):
        steps = []
        for ci, _, _ in cl:
            # the cycle's first prompt step carries the profiler's start (MEASURED: 217-293 ms of host time for a
            # 19-row step); left out of the cell
            xs = sorted((x for x in dig[(ci, s, 0)] if x["kind"] == "prefill"), key=lambda x: x["start"])[1:]
            seen.update(f"PP{s}:{x['ctx_tok']}" for x in xs)
            # a chunk the last stage skips (VLLM_V41_PREFILL_SKIP_ROWS: no layer of that stage runs) launches no
            # expert kernel (MEASURED, window t1: such a step still runs the drafter's _linear and a ring insert, so
            # attention kernels alone do not tell it); it is counted apart, not averaged in
            skipped[s] += sum(1 for x in xs if x["ctx_tok"] == a.chunk and not ran_layers(x))
            steps.extend(x for x in xs if x["ctx_tok"] == a.chunk and ran_layers(x))
            if s == 0:
                starts0.append([x["start"] for x in xs if x["ctx_tok"] == a.chunk])
        cell = f"prefill_chunk_{label} stage{s} (trace, rows {a.chunk})"
        if len(steps) < PROPOSED_MIN_CHUNKS:
            rep.inc(cell, f"{len(steps)} prefill step(s) of {a.chunk} rows (need {PROPOSED_MIN_CHUNKS}); prefill "
                          f"steps seen by rows: {dict(seen)}")
            continue
        print(f"NOTE {cell}: {skipped[s]} chunk(s) of {a.chunk} rows ran no layer on this stage (skipped: no expert kernel, no TP sum) and are "
              f"not in the cell")
        busy_med[s] = phase_lines(cell, steps, cm.phases, rep)
        report_sums(a, dig, cl, s, layers[s], rep, label)
    gaps = [(b - x) / 1000 for st in starts0 for x, b in zip(st, st[1:])]
    if busy_med and gaps:
        pm = statistics.median(gaps)
        print(f"COVERAGE prefill_chunk (trace): chunk interval on PP0TP0 median {pm:.1f} ms; busy per stage "
              + ", ".join(f"{s}: {v:.1f} ms ({v / pm:.3f})" for s, v in sorted(busy_med.items())))
    elif busy_med:
        rep.inc("prefill_chunk interval", "fewer than two chunks on PP0TP0 in one cycle")


def ms(us):
    return us / 1000.0


WARMUP_ROWS_MAX = 64   # turn_probe's warm-up ("Say OK.", one token out) is a prompt step of a few rows


def merge_steps(ps):
    """One turn's prompt steps as one: rows, host time, API and memcpy sums, busy and phases added; first start, last end."""
    m = {"ctx_tok": sum(x["ctx_tok"] for x in ps), "steps": len(ps), "cpu_start": ps[0]["cpu_start"],
         "cpu_dur": sum(x["cpu_dur"] for x in ps), "start": ps[0]["start"], "end": ps[-1]["end"],
         "gpu_end_all": ps[-1]["gpu_end_all"], "busy": sum(x["busy"] for x in ps), "parts": list(ps)}
    for key in ("by", "api", "memcpy"):
        c = collections.Counter()
        for x in ps:
            c.update(x[key])
        m[key] = dict(c)
    return m


def report_turns(a, cm, dig, label, cl, turns, prefix, rep):
    """ONE TURN ON A GROWN CHAT, from arrival to first token, per cycle and then as medians over the cycles:
      client       seconds to the first streamed token (turn_probe)
      metrics      /metrics deltas of the one request: TTFT (API arrival -> first token), queue (queued -> scheduled,
                   the admission and prefix decision inside it), prefill (scheduled -> first token)
      front        TTFT - queue - prefill: chat template, tokenization, IPC (a residual: no timer splits it)
      prefix       the scheduler's decision line (hit, new rows, rewind)
      stage s      its prompt step: rows; host time inside execute_model (annotation span) and the CUDA API time in
                   it (synchronize, free = the allocator release, memcpy calls); GPU busy by phase; memcpy by kind
      chain        PP0's prompt step start (host) -> the last stage's last kernel before its next step (GPU): the
                   engine's own execution of the turn across the three stages, sampling and drafter included"""
    nreq = cl[0][1]
    cell = f"turn_{label}x{nreq}"
    rows = []
    for ci, _, _ in cl:
        per = {}
        for s in range(a.stages):
            # The turn's prompt steps: every prompt step of the cycle except turn_probe's one-token warm-up, which
            # an alone cycle sends first (it takes the profiler's slow first step; MEASURED, see turn_probe.py). A
            # turn over 4,092 rows (the ~4,200 turn plus its template and rewind) takes two steps: they are summed.
            ps = sorted((x for x in dig[(ci, s, 0)] if x["kind"] in ("prefill", "mixed")), key=lambda x: x["start"])
            if nreq == 1 and len(ps) >= 2 and ps[0]["ctx_tok"] <= WARMUP_ROWS_MAX:
                ps = ps[1:]
            if not ps or any(x["ctx_tok"] <= WARMUP_ROWS_MAX for x in ps[:-1]):
                rep.inc(f"{cell} cycle {ci} stage{s}", f"prompt steps of {[x['ctx_tok'] for x in ps]} rows: not one turn "
                                                       f"beside a warm-up of at most {WARMUP_ROWS_MAX} rows")
                per = None
                break
            per[s] = merge_steps(ps)
        if per is None:
            continue
        t = turns.get(ci)
        md = t["metrics_delta"] if t else None
        req = t["requests_measured"][0] if t else None
        pf = prefix.get(req["id"]) if req else None
        chain = (per[a.stages - 1]["gpu_end_all"] - per[0]["cpu_start"]) / 1000
        line = f"TURN {cell} cycle {ci}:"
        pr_ms = None
        if req:
            line += f" prompt {req['prompt']} tok, client first token {req['ttft_s']:.3f} s;"
        if md:
            if all(md[h]["count"] == 1 for h in ("time_to_first_token_seconds", "request_queue_time_seconds",
                                                  "request_prefill_time_seconds")):
                tt, qu, pr = (md[h]["sum_s"] for h in ("time_to_first_token_seconds", "request_queue_time_seconds",
                                                       "request_prefill_time_seconds"))
                pr_ms = pr * 1000
                line += (f" engine TTFT {tt * 1000:.1f} ms = front {1000 * (tt - qu - pr):.1f} + queue {qu * 1000:.1f}"
                         f" + scheduled-to-first-token {pr * 1000:.1f};")
            else:
                rep.inc(f"{cell} cycle {ci} metrics", "another request finished inside the cycle: the histogram deltas"
                                                      " are not this request's alone")
        else:
            rep.inc(f"{cell} cycle {ci} metrics", "no turns.jsonl record for this cycle")
        if pf:
            line += f" prefix {pf['decision']}/{pf['reason']} hit {pf['hit']} new rows {pf['prompt'] - pf['hit']} rewind {pf['rewind']};"
        elif req and a.cmd == "all":
            rep.inc(f"{cell} cycle {ci} prefix", f"no VLLM_V41_PREFIX_LOG line for request {req['id']}")
        line += f" engine chain {chain:.1f} ms"
        if pr_ms is not None:
            # what the trace's chain does not hold of scheduled-to-first-token: scheduler-to-worker hand-over, the
            # output's way back to the API server, and any host time before PP0's step annotation (MEASURED residual)
            line += f"; residual (scheduled-to-first-token - chain) {pr_ms - chain:.1f} ms"
        print(line)
        for s in range(a.stages):
            x = per[s]
            if x["steps"] > 1:
                # a turn over 4,092 rows: each prompt step apart (the pair the next optimization changes), then the sum
                for j, y in enumerate(x["parts"]):
                    ty = sorted(((k, v) for k, v in y["by"].items() if k != "handoff"), key=lambda kv: -kv[1])[:6]
                    print(f"TURNSTEP {cell} cycle {ci} stage{s} step {j + 1} of {x['steps']}: rows {y['ctx_tok']}; host in "
                          f"execute_model {ms(y['cpu_dur']):.2f} ms (API: " + ", ".join(
                              f"{g} {ms(v):.2f}" for g, v in sorted(y["api"].items(), key=lambda kv: -kv[1]))
                          + f"); GPU span {ms(y['end'] - y['start']):.2f} busy {ms(y['busy']):.2f} ms ("
                          + ", ".join(f"{k} {ms(v):.2f}" for k, v in ty) + ")")
            top = sorted(((k, v) for k, v in x["by"].items() if k != "handoff"), key=lambda kv: -kv[1])[:6]
            print(f"TURNSTAGE {cell} cycle {ci} stage{s}: rows {x['ctx_tok']} in {x['steps']} step(s); host in execute_model {ms(x['cpu_dur']):.2f} ms "
                  f"(API: " + ", ".join(f"{g} {ms(v):.2f}" for g, v in sorted(x["api"].items(), key=lambda kv: -kv[1])) +
                  f"); GPU span {ms(x['end'] - x['start']):.2f} busy {ms(x['busy']):.2f} ms ("
                  + ", ".join(f"{k} {ms(v):.2f}" for k, v in top) + f"); hand-off {ms(x['by'].get('handoff', 0.0)):.2f}; "
                  f"memcpy " + (", ".join(f"{k} {v / 1e6:.1f} MB" if k.endswith('bytes') else f"{k} {ms(v):.2f} ms"
                                         for k, v in sorted(x["memcpy"].items())) or "none"))
        rows.append({"chain": chain, "per": per, "md": md})
    if not rows:
        rep.inc(cell, "no cycle could be read")
        return
    print(f"CELL {cell}: cycles {len(rows)}; engine chain ms {stat([r['chain'] for r in rows])}")
    for s in range(a.stages):
        steps = sorted({r["per"][s]["steps"] for r in rows})
        print(f"PHASE {cell} stage{s} prompt steps per turn {steps}; host_in_step ms {stat([ms(r['per'][s]['cpu_dur']) for r in rows])}; "
              f"gpu_busy ms {stat([ms(r['per'][s]['busy']) for r in rows])}")


def report_sums(a, dig, cl, s, nl, rep, label="read"):
    """The large TP sums of a prompt forward, timed apart: per chunk, the coll_tp_sum kernels of each TP rank in
    launch order; the sum j's arrival skew = latest start - earliest start over the four ranks, its transfer
    estimate = the shortest of the four durations (the last rank to arrive waits for no one)."""
    want = 2 * nl
    cnt, dur, skew, xfer = collections.Counter(), [], [], []
    for ci, _, _ in cl:
        per_t = [[x for x in sorted((y for y in dig[(ci, s, t)] if y["kind"] == "prefill"), key=lambda y: y["start"])[1:]
                  if x["ctx_tok"] == a.chunk and ran_layers(x)] for t in range(a.tp)]
        if len({len(x) for x in per_t}) != 1:
            rep.inc(f"tp_sums_{label} stage{s}", f"cycle {ci}: unequal chunk counts over the TP ranks")
            return
        for i in range(len(per_t[0])):
            ss = [per_t[t][i]["sums"] for t in range(a.tp)]
            cnt[len(ss[0])] += 1
            if len({len(x) for x in ss}) != 1:
                rep.inc(f"tp_sums_{label} stage{s}", f"cycle {ci} chunk {i}: unequal sum counts over the TP ranks")
                return
            dur.append(sum(d for _, d in ss[0]) / 1000)
            for j in range(len(ss[0])):
                st = [ss[t][j][0] for t in range(a.tp)]
                skew.append((max(st) - min(st)) / 1000)
                xfer.append(min(ss[t][j][1] for t in range(a.tp)) / 1000)
    print(f"SUMS {label} stage{s}: TP-sum kernels per chunk {dict(cnt)} (two per layer would be {want}); per chunk total ms "
          f"{stat(dur)}; per sum: transfer estimate ms {stat(xfer)}, arrival skew ms {stat(skew)}")
    # the embedding's, the indexer's and Engram's sums are of other sizes and cannot be told apart without shapes
    # (record_shapes is off): more than two per layer is expected, fewer means a sum was not seen
    rep.check(f"tp_sums_count {label} stage{s}", "PASS" if cnt and min(cnt) >= want else "FAIL",
              f"sum kernels per chunk {dict(cnt)}, at least two per layer = {want}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--stages", type=int, default=3)
        p.add_argument("--tp", type=int, default=4)
        p.add_argument("--chunk", type=int, default=4092)

    def refs(p):
        # the standing build's outside figures the timed boot is compared with: tonight's 200 W ones by default
        # (team lead 2026-11-17, the serving arm P3g1-S): one request 129.7 fresh / 179.2 repeated tok/s, read 3,733 tok/s
        p.add_argument("--ref-read", type=float, default=3733.0, help="tok/s of the standing build's prompt read")
        p.add_argument("--ref-one-fresh", type=float, default=129.7)
        p.add_argument("--ref-one-repeated", type=float, default=179.2)
        p.add_argument("--ref-name", default="standing build at 200 W, 2026-11-17 (P3g1-S)")

    lp = sub.add_parser("log")
    lp.add_argument("log")
    lp.add_argument("--ladder")
    common(lp)
    refs(lp)
    tp = sub.add_parser("traces")
    tp.add_argument("traces")
    for p in (tp, sub.add_parser("all")):
        p.add_argument("--windows", required=True)
        p.add_argument("--classmap", required=True)
        p.add_argument("--layers", default="14,14,12")
        p.add_argument("--skip-rounds", type=int, default=2)
        p.add_argument("--runs-json", help="ktrace_driver.py's runs.json of the same boot (client round times)")
        p.add_argument("--turns", help="turn_probe.py's turns.jsonl of the same boot (client and /metrics times)")
        common(p)
    allp = sub.choices["all"]
    allp.add_argument("--log", required=True)
    allp.add_argument("--traces", required=True)
    allp.add_argument("--ladder")
    refs(allp)
    a = ap.parse_args(argv)
    rep = Report()
    try:
        if a.cmd in ("log", "all"):
            lg = read_log(a.log)
            print(f"REFERENCE {a.ref_name}: read {a.ref_read:.0f} tok/s, one request {a.ref_one_fresh} fresh / "
                  f"{a.ref_one_repeated} repeated tok/s")
            report_log(lg, rep, a.stages, a.chunk, read_ladder(a.ladder) if a.ladder else None, a)
        if a.cmd in ("traces", "all"):
            report_traces(a, rep)
    except Unparseable as e:
        print(f"UNPARSEABLE {e}")
        return EXIT_UNPARSEABLE
    return rep.exit_code()


if __name__ == "__main__":
    sys.exit(main())
