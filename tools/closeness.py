"""closeness.py DIR REF ARM [REF ARM ...]: per prompt, from two quality_t3_wide receipts (token ids and logprobs only):
first differing token index, REF's logprob gap at that index between its own token and ARM's token (">=x" when ARM's
token is outside REF's top list), and mean / max |logprob difference| of the chosen tokens before the divergence.

STRICT RECEIPTS (2026-11-17, after the independent review's finding A2/C3: an answer cut short of its reference
counted as identical, because only the shorter answer was compared). Before any figure is computed, every pair of
receipts a mode reads must be COMPLETE (complete_pair):
  - the same set of prompt keys in both, no key twice in either; when the reference is a long capture (keys
    long<N>) its keys are exactly the seven sizes of tools/long_capture.py (LONG_SET);
  - the same prompt for each key, as the receipts record it (prompt_sha256 in the wide set, prompt_tokens in the
    long set; a field one receipt has and the other lacks, or neither recording either, is incomplete);
  - finite, aligned score arrays: logprobs, top_ids and top_logprobs of the answer's length, each top list paired
    with its logprobs, every number finite;
  - the finish reason and the completion's token count (finish_reason / finish, completion_tokens) are compared where
    BOTH receipts record them. tools/long_capture.py records both since this change; older receipts (the accepted
    capture of window o2 among them) and the wide set's (quality_t3_wide) do not: a REFERENCE without them is
    compared without them and the line says "finish reason not recorded in the reference"; an ARM without them
    where its reference has them is incomplete.
An incomplete pair makes the line INCONCLUSIVE (exit 3, the code the driver maps to 34), never a pass and never a
figure. The words of an INCONCLUSIVE line never include JUNK, WORSE, HOLDS or CLEAN (the driver's case patterns).
IDENTICAL for one prompt: equal token-id sequences of EQUAL LENGTH (and the same finish reason where recorded). An
answer shorter or longer than the other is a difference at position min(len), reported as "length differs (r, a)".
The closeness lines keep their definitions (the rules are on record): their first divergence is still the first
differing position within the shorter answer; each prompt's line says whether the comparison was CENSORED by an
answer's end (no differing token before min(len)) or ended at a real divergence."""
import json, math, statistics, sys

LONG_SET = frozenset(f"long{n}" for n in (6000, 10000, 12000, 14000, 17000, 20000, 24000))   # tools/long_capture.py SIZES
IDENTITY = ("prompt_sha256", "prompt_tokens")
FINISH = ("finish_reason", "finish")
COUNT = ("completion_tokens",)
NOT_IN_REASONS = ("JUNK", "WORSE", "HOLDS", "CLEAN")


def items_of(d, name):
    """The receipt's prompt records as a LIST (a key that occurs twice is kept twice, for complete_pair to see)."""
    doc = json.load(open(f"{d}/quality-{name}.json"))
    items = doc["prompts"] if isinstance(doc, dict) and "prompts" in doc else doc.get("items", doc) if isinstance(doc, dict) else doc
    if isinstance(items, dict):
        items = list(items.values())
    return items


def load(d, name):
    return {it["id"]: it for it in items_of(d, name)}


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _record_gaps(it):
    """Reasons a prompt record cannot be scored: missing or misaligned or non-finite arrays."""
    out = []
    t = it.get("token_ids")
    if not isinstance(t, list) or not all(isinstance(v, int) for v in t):
        return ["token_ids missing or not integers"]
    for f in ("logprobs", "top_ids", "top_logprobs"):
        if not isinstance(it.get(f), list):
            out.append(f"{f} missing")
        elif len(it[f]) != len(t):
            out.append(f"{f} has {len(it[f])} entries for {len(t)} tokens")
    if out:
        return out
    if not all(_finite(v) for v in it["logprobs"]):
        out.append("a logprob is not finite")
    for ids, lps in zip(it["top_ids"], it["top_logprobs"]):
        if not isinstance(ids, list) or not isinstance(lps, list) or len(ids) != len(lps) or not ids:
            out.append("a top list is empty or not paired with its logprobs")
            break
        if not all(_finite(v) for v in lps):
            out.append("a top logprob is not finite")
            break
    return out


def complete_pair(d, ref, arm, d_arm=None):
    """Reasons (empty = complete) the receipts REF (in d) and ARM (in d_arm, default d) cannot be compared prompt for
    prompt. Pure but for reading the two files."""
    reasons = []
    try:
        ri, ai = items_of(d, ref), items_of(d_arm or d, arm)
    except (OSError, ValueError, AttributeError, TypeError) as e:
        return [f"a receipt cannot be read ({type(e).__name__})"]
    for name, items in ((ref, ri), (arm, ai)):
        if not isinstance(items, list) or not all(isinstance(it, dict) and "id" in it for it in items):
            return [f"{name}: prompt records without an id"]
        seen = [it["id"] for it in items]
        dup = sorted({k for k in seen if seen.count(k) > 1})
        if dup:
            reasons.append(f"{name}: key(s) occurring twice: {dup[:3]}")
    rk, ak = {it["id"] for it in ri}, {it["id"] for it in ai}
    if any(str(k).startswith("long") for k in rk) and rk != LONG_SET:
        reasons.append(f"{ref}: the long-prompt keys are not the seven sizes (missing {sorted(LONG_SET - rk)[:3]}, "
                       f"other {sorted(rk - LONG_SET)[:3]})")
    if ak - rk:
        reasons.append(f"{arm}: key(s) the reference lacks: {sorted(ak - rk)[:3]}")
    if rk - ak:
        reasons.append(f"{arm}: missing key(s): {sorted(rk - ak)[:3]}")
    R, A = {it["id"]: it for it in ri}, {it["id"]: it for it in ai}
    for k in sorted(rk & ak, key=str):
        r, a = R[k], A[k]
        both = [f for f in IDENTITY if f in r and f in a]
        one = [f for f in IDENTITY if (f in r) != (f in a)]
        if one or not both:
            reasons.append(f"{k}: the prompt's identity is not recorded in both receipts ({', '.join(one) or 'neither'})")
        elif any(r[f] != a[f] for f in both):
            reasons.append(f"{k}: another prompt ({', '.join(f for f in both if r[f] != a[f])} differ)")
        for name, it in ((ref, r), (arm, a)):
            for g in _record_gaps(it):
                reasons.append(f"{k} in {name}: {g}")
        for fields, what in ((FINISH, "finish reason"), (COUNT, "completion token count")):
            if any(f in r for f in fields) and not any(f in a for f in fields):
                reasons.append(f"{k}: the reference records a {what}, {arm} does not")
    for x in reasons:
        assert not any(w in x for w in NOT_IN_REASONS), x
    return reasons


def _finish_of(it):
    return next((it[f] for f in FINISH if f in it), None)


def _count_of(it):
    return next((it[f] for f in COUNT if f in it), None)


def _ends_differ(r, a):
    """The finish reasons or completion counts that BOTH record and that differ, as words; empty when none (pure)."""
    out = []
    for get, what in ((_finish_of, "finish reason"), (_count_of, "completion tokens")):
        x, y = get(r), get(a)
        if x is not None and y is not None and x != y:
            out.append(f"{what} differs ({x}, {y})")
    return out


def pair_notes(d, ref, arm, d_arm=None):
    """What the comparison leaves out because the REFERENCE does not record it (printed, not a gap)."""
    try:
        R = items_of(d, ref)
    except (OSError, ValueError, AttributeError, TypeError):
        return []
    out = []
    if R and not any(f in it for it in R for f in FINISH):
        out.append(f"{ref}: finish reason not recorded in the reference")
    if R and not any(f in it for it in R for f in COUNT):
        out.append(f"{ref}: completion token count not recorded in the reference")
    return out


def _end(r, a, first):
    """How the comparison of one prompt ended (pure)."""
    lr, la = len(r["token_ids"]), len(a["token_ids"])
    if first < min(lr, la):
        return "diverged"
    if lr != la:
        return f"CENSORED by an answer's end: length differs ({lr}, {la})"
    if _ends_differ(r, a):
        return "same tokens, " + "; ".join(_ends_differ(r, a))
    return "identical (equal length)"


def is_identical(r, a):
    return r["token_ids"] == a["token_ids"] and not _ends_differ(r, a)


def _inconclusive(head, pairs):
    """Print each pair's reasons and the machine line HEAD: INCONCLUSIVE (...). Returns the reasons (empty: complete)."""
    reasons = []
    for args in pairs:
        for x in pair_notes(*args):
            print(f"  NOTE {x}")
        reasons += [f"{args[1]}/{args[2]}: {x}" for x in complete_pair(*args)]
    if reasons:
        for x in reasons:
            print(f"  INCOMPLETE {x}")
        print(f"{head}: INCONCLUSIVE ({len(reasons)} gap(s) in the receipts; first: {reasons[0]})")
    return reasons

def compare(d, ref, arm):
    """The report of one pair; returns False (and prints SUMMARY ...: INCONCLUSIVE) when the receipts are incomplete."""
    print(f"== {arm} against {ref}: prompt, tokens compared, first divergence, {ref}'s gap there (nat), mean and max |dlogprob| before it")
    if _inconclusive(f"  SUMMARY {arm} vs {ref}", [(d, ref, arm)]):
        return False
    R, A = load(d, ref), load(d, arm)
    rows, ends = [], {}
    for pid in R:
        r, a = R[pid], A[pid]
        n = min(len(r["token_ids"]), len(a["token_ids"]))
        first = next((i for i in range(n) if r["token_ids"][i] != a["token_ids"][i]), n)
        dl = [abs(r["logprobs"][i] - a["logprobs"][i]) for i in range(first)]
        gap = None
        if first < n:
            ids, lps = r["top_ids"][first], r["top_logprobs"][first]
            own = r["logprobs"][first]
            gap = (own - lps[ids.index(a["token_ids"][first])], "") if a["token_ids"][first] in ids else (own - min(lps), ">=")
        rows.append((pid, n, first, gap, statistics.fmean(dl) if dl else None, max(dl) if dl else None))
        ends[pid] = (_end(r, a, first), is_identical(r, a))
    for pid, n, first, gap, mean, mx in rows:
        g = ("none (identical)" if ends[pid][1] else "none (no differing token)") if gap is None else f"{gap[1]}{gap[0]:.3f}"
        print(f"  {str(pid)[:14]:14} {n:5d} {first:5d}  gap {g:24}  mean {('%.4f' % mean) if mean is not None else '   -  '}  max {('%.3f' % mx) if mx is not None else '  -  '}  end: {ends[pid][0]}")
    means = [m for *_, m, _ in rows if m is not None]
    gaps = [g[0] for *_, g, _, _ in rows if g is not None]
    ident = sum(1 for pid, *_ in rows if ends[pid][1])
    early = sum(1 for _, n, f, *_ in rows if f <= 3 and f < n)
    print(f"  SUMMARY {arm} vs {ref}: identical {ident} of {len(rows)}; diverging at token 0-3: {early}; median first divergence "
          f"{statistics.median(f for _, _, f, *_ in rows):.0f}; median of the mean |dlogprob| {statistics.median(means) if means else float('nan'):.4f}; "
          f"prompts with mean <= 0.01: {sum(m <= 0.01 for m in means)}; largest gap at a divergence {max(gaps) if gaps else float('nan'):.3f}; gaps over 0.9 nat: {sum(g > 0.9 for g in gaps)}")
    return True

def verdict(d, ref, arm):
    """One machine line. JUNK = the fault arm X showed on 2026-11-17 (first decode round after the prompt wrong):
    on at least 10 of 12 prompts the first divergence is at token 0-3 AND the reference's gap there is over 0.9 nat.
    CLEAN = at most 2 prompts show that. Anything else UNCLEAR. Fixed 2026-11-17 01:4xZ before the diagnostic arms.
    INCONCLUSIVE: the receipts are incomplete (complete_pair)."""
    if _inconclusive(f"VERDICT {arm} vs {ref}", [(d, ref, arm)]):
        return "INCONCLUSIVE"
    R, A = load(d, ref), load(d, arm)
    junk = ident = 0
    for pid in R:
        r, a = R[pid], A[pid]
        n = min(len(r["token_ids"]), len(a["token_ids"]))
        first = next((i for i in range(n) if r["token_ids"][i] != a["token_ids"][i]), n)
        print(f"  END {pid}: {_end(r, a, first)}")
        if first == n:
            ident += is_identical(r, a)
            continue
        ids, lps = r["top_ids"][first], r["top_logprobs"][first]
        own = r["logprobs"][first]
        gap = own - lps[ids.index(a["token_ids"][first])] if a["token_ids"][first] in ids else own - min(lps)
        junk += first <= 3 and gap > 0.9
    word = "JUNK" if junk >= 10 else "CLEAN" if junk <= 2 else "UNCLEAR"
    print(f"VERDICT {arm} vs {ref}: {word} (early wrong-token divergences {junk} of {len(R)}, identical {ident})")
    return word


def keeps(d, ref, arm, min_first, max_dlp):
    """One machine line for a change that claims to be the same computation: the arm's closeness to REF must not be
    worse than figures measured before the change. HOLDS = the median first divergence is at least min_first tokens
    and the median over prompts of the mean |dlogprob| before the divergence is at most max_dlp; else WORSE.
    INCONCLUSIVE: the receipts are incomplete (complete_pair)."""
    if _inconclusive(f"KEEPS {arm} vs {ref}", [(d, ref, arm)]):
        return "INCONCLUSIVE"
    R, A = load(d, ref), load(d, arm)
    firsts, means = [], []
    for pid in R:
        r, a = R[pid], A[pid]
        n = min(len(r["token_ids"]), len(a["token_ids"]))
        first = next((i for i in range(n) if r["token_ids"][i] != a["token_ids"][i]), n)
        print(f"  END {pid}: {_end(r, a, first)}")
        firsts.append(first)
        if first:
            means.append(statistics.fmean(abs(r["logprobs"][i] - a["logprobs"][i]) for i in range(first)))
    mf = statistics.median(firsts)
    md = statistics.median(means) if means else float("nan")
    word = "HOLDS" if mf >= min_first and md <= max_dlp else "WORSE"
    print(f"KEEPS {arm} vs {ref}: {word} (median first divergence {mf:.0f}, need at least {min_first:g}; "
          f"median mean |dlogprob| {md:.4f}, need at most {max_dlp:g})")
    return word


def _medians(d, ref, arm):
    R, A = load(d, ref), load(d, arm)
    firsts, means = [], []
    for pid in R:
        r, a = R[pid], A[pid]
        n = min(len(r["token_ids"]), len(a["token_ids"]))
        first = next((i for i in range(n) if r["token_ids"][i] != a["token_ids"][i]), n)
        print(f"  END {arm} {pid}: {_end(r, a, first)}")
        firsts.append(first)
        if first:
            means.append(statistics.fmean(abs(r["logprobs"][i] - a["logprobs"][i]) for i in range(first)))
    return statistics.median(firsts), (statistics.median(means) if means else float("nan"))


def nearer(d, ref, arm, other):
    """One machine line: is ARM nearer REF than OTHER is? HOLDS = ARM's median first divergence from REF is at least
    twice OTHER's AND ARM's median mean |dlogprob| is at most half OTHER's; else NOT SHOWN. (What rule line 3 asks
    of the all-NVFP4 build: nearer the all-native reference than the default; window w5: 208 tokens and 0.0090
    against the default's 7 tokens and 0.0656.) INCONCLUSIVE: the receipts are incomplete (complete_pair)."""
    if _inconclusive(f"NEARER {arm} than {other} to {ref}", [(d, ref, arm), (d, ref, other)]):
        return "INCONCLUSIVE"
    fa, da = _medians(d, ref, arm)
    fo, do = _medians(d, ref, other)
    word = "HOLDS" if fa >= 2 * fo and da <= 0.5 * do else "NOT SHOWN"
    print(f"NEARER {arm} than {other} to {ref}: {word} (median first divergence {fa:.0f} against {fo:.0f}; "
          f"median mean |dlogprob| {da:.4f} against {do:.4f})")
    return word


def _first_and_mean(r, a, label=None):
    n = min(len(r["token_ids"]), len(a["token_ids"]))
    first = next((i for i in range(n) if r["token_ids"][i] != a["token_ids"][i]), n)
    if label:
        print(f"  END {label}: {_end(r, a, first)}")
    mean = statistics.fmean(abs(r["logprobs"][i] - a["logprobs"][i]) for i in range(first)) if first else None
    return n, first, mean


def scaled_word(arm_rows, base_rows):
    """(word, arm median first, arm median mean, base median first, base median mean) from per-prompt (tokens compared,
    first divergence, mean |dlogprob| or None) of ARM against REF and of BASE against REF on the same prompts. (Pure.)
    CANNOT DECIDE: no prompt, or BASE is identical to REF on every prompt (the comparator did not move), or a median
    mean is undefined. HOLDS: ARM's median first divergence is at least half BASE's and ARM's median mean |dlogprob| at
    most twice BASE's. Else WORSE."""
    if not arm_rows or len(arm_rows) != len(base_rows):
        return "CANNOT DECIDE", None, None, None, None
    fa, fb = statistics.median(f for _, f, _ in arm_rows), statistics.median(f for _, f, _ in base_rows)
    ma, mb = [m for *_, m in arm_rows if m is not None], [m for *_, m in base_rows if m is not None]
    da, db = (statistics.median(ma) if ma else None), (statistics.median(mb) if mb else None)
    if all(f == n for n, f, _ in base_rows) or da is None or db is None:
        return "CANNOT DECIDE", fa, da, fb, db
    return ("HOLDS" if fa >= 0.5 * fb and da <= 2 * db else "WORSE"), fa, da, fb, db


def scaled(d, ref, arm, base, min_prompt):
    """One machine line for a change of how a prompt is cut (window o2, PLAN Amendment 12): does ARM differ from REF by
    no more than BASE, a harmless change of forward shape, differs from REF? Only prompts whose REF prompt_tokens
    exceed min_prompt count (prompts every arm cuts). INCONCLUSIVE: the receipts are incomplete (complete_pair)."""
    if _inconclusive(f"SCALED {arm} vs {ref} against {base} vs {ref}", [(d, ref, arm), (d, ref, base)]):
        return "INCONCLUSIVE"
    R, A, B = load(d, ref), load(d, arm), load(d, base)
    ids = [pid for pid in R if R[pid]["prompt_tokens"] > min_prompt]
    word, fa, da, fb, db = scaled_word([_first_and_mean(R[i], A[i], f"{arm} {i}") for i in ids], [_first_and_mean(R[i], B[i], f"{base} {i}") for i in ids])
    f = lambda x, spec: "undefined" if x is None else format(x, spec)
    print(f"SCALED {arm} vs {ref} against {base} vs {ref}: {word} ({len(ids)} prompts over {min_prompt} tokens; median first "
          f"divergence {f(fa, '.0f')} against {f(fb, '.0f')}, need at least half; median mean |dlogprob| {f(da, '.4f')} "
          f"against {f(db, '.4f')}, need at most twice)")
    return word


def scaled_by(d, ref, arm, ydir, yref, ybase, min_prompt):
    """Line c2's rule (scaled_word, unchanged) with the yardstick taken from ANOTHER directory: ARM against REF in D,
    judged against YBASE against YREF in YDIR (arm G, owner 2026-11-17: the yardstick is window o2's 64-row shift of
    the cut, F2s against F2 in o2/ql). The prompts are those whose REF prompt_tokens exceed min_prompt; YREF must
    hold the same prompts with the same prompt_tokens, else CANNOT DECIDE (the two pairs would not be measured on
    the same prompts). INCONCLUSIVE: the receipts are incomplete (complete_pair)."""
    if _inconclusive(f"SCALED {arm} vs {ref} against {ybase} vs {yref} (yardstick {ydir})", [(d, ref, arm), (ydir, yref, ybase)]):
        return "INCONCLUSIVE"
    R, A = load(d, ref), load(d, arm)
    YR, YB = load(ydir, yref), load(ydir, ybase)
    ids = [pid for pid in R if R[pid]["prompt_tokens"] > min_prompt]
    yids = [pid for pid in YR if YR[pid]["prompt_tokens"] > min_prompt]
    same = sorted(ids) == sorted(yids) and all(R[i]["prompt_tokens"] == YR[i]["prompt_tokens"] for i in ids)
    if same:
        word, fa, da, fb, db = scaled_word([_first_and_mean(R[i], A[i], f"{arm} {i}") for i in ids], [_first_and_mean(YR[i], YB[i], f"{ybase} {i}") for i in ids])
    else:
        word, fa, da, fb, db = "CANNOT DECIDE", None, None, None, None
    f = lambda x, spec: "undefined" if x is None else format(x, spec)
    print(f"SCALED {arm} vs {ref} against {ybase} vs {yref} (yardstick {ydir}): {word} ({len(ids)} prompts over {min_prompt} "
          f"tokens{'' if same else ', NOT the yardstick prompts'}; median first divergence {f(fa, '.0f')} against {f(fb, '.0f')}, "
          f"need at least half; median mean |dlogprob| {f(da, '.4f')} against {f(db, '.4f')}, need at most twice)")
    return word


if __name__ == "__main__":
    d = sys.argv[1]
    if sys.argv[2] == "--scaled-by":
        sys.exit({"HOLDS": 0, "WORSE": 1, "CANNOT DECIDE": 3, "INCONCLUSIVE": 3}[scaled_by(d, *sys.argv[3:8], int(sys.argv[8]))])
    if sys.argv[2] == "--scaled":
        sys.exit({"HOLDS": 0, "WORSE": 1, "CANNOT DECIDE": 3, "INCONCLUSIVE": 3}[scaled(d, sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]))])
    if sys.argv[2] == "--nearer":
        sys.exit({"HOLDS": 0, "NOT SHOWN": 1, "INCONCLUSIVE": 3}[nearer(d, sys.argv[3], sys.argv[4], sys.argv[5])])
    if sys.argv[2] == "--verdict":
        sys.exit({"CLEAN": 0, "JUNK": 1, "UNCLEAR": 3, "INCONCLUSIVE": 3}[verdict(d, sys.argv[3], sys.argv[4])])
    if sys.argv[2] == "--keeps":
        sys.exit({"HOLDS": 0, "WORSE": 1, "INCONCLUSIVE": 3}[keeps(d, sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6]))])
    complete = [compare(d, sys.argv[i], sys.argv[i + 1]) for i in range(2, len(sys.argv), 2)]
    sys.exit(0 if all(complete) else 3)
