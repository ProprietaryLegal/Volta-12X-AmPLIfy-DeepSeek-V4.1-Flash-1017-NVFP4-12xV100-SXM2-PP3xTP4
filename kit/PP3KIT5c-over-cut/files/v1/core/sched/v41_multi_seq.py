# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_MULTI_SEQ: serve up to four DeepSeek-V4.1 requests at once.

The V4.1 package keeps per-request state on the device outside the vLLM block
pool: the window rings, the ratio-2 compressor states, the Engram hash
history, the pending commit buffers, the first stage's commit protocol and the
packed long-context KV store (host arena, hot ring, GPU index keys). All of it
is sized for one request, so the model refuses ``max_num_seqs != 1``.

With the knob on, every running request owns one *slot*: a private copy of
that state. A step still runs exactly one request (M <= 6 verify rows, the
fast single-request kernels and their numerics), so nothing in the model's
forward changes. With pipeline parallelism (PP=2) the engine's batch queue
already keeps two steps in flight; two running requests then alternate, one
in each stage, instead of one request leaving a stage idle on every other
step. Three or four requests time-slice the same two stages.

Pieces, all keyed off this module's decision:

- scheduler (``V41SchedulerMultiSeq``): at most one request per step, chosen
  least-recently-scheduled among the running requests that can run, with a
  turn for waiting requests; a slot per running request, released when its
  blocks are freed (finish, abort, preemption), sent to the workers as
  ``SchedulerOutput.v41_slots``; the previous-step record keeps every running
  request the runner holds, so a request's token history is not resent after
  another request's step; one session-prefix record per slot; the
  prefill-beside-decoders policy (``prefill_policy``): decode first, a full
  prefill step owes every decoding request ``VLLM_V41_MULTI_SEQ_DECODE_TURNS``
  steps (pro rata for shorter ones), after up to
  ``VLLM_V41_MULTI_SEQ_PREFILL_BURST`` prefill steps in a row, and
  ``VLLM_V41_MULTI_SEQ_PREFILL_CHUNK`` optionally caps a prefill step's tokens
  while a request is decoding.
- runner (``vllm/v1/worker/v41_multi_seq_runner.py``): binds the step's slot
  before the forward, parks an unscheduled request's token row instead of
  rebuilding it from Python lists, replays one CUDA graph set per slot, and
  hands each step's drafts back with its output.
- model (``vllm/models/deepseek_v41/slots.py``): the slot copies and the
  binding (dictionary swaps, only when the slot changes).

Every request type is served (KCAMP1106 K4d, the concurrency default): a
request with structured output (response_format json/json_schema, guided
json/regex/choice/grammar, required or named tool_choice) is admitted
EXCLUSIVE: only when no request is resident, and nothing is admitted while
it is resident or waiting (the running requests drain, so it cannot starve).
It then runs the one-request step path, exactly as single mode serves it; the
mode returns to grouping when it finishes. Every other request is GROUPED.

With the knob off, nothing here runs and the one-request path is unchanged.

``VLLM_V41_PREFIX_LOG=1`` (unset or 0: off) logs, from the scheduler process,
one INFO line per admitted request: its slot, prompt length, the block pool's
prefix hit, the hit taken, hit or cold and a reason from ``PrefixReason``,
with every slot's processed length and the length of its record that matches
the prompt (numbers only: no token ids, no text, no hashes). It reads the
decision; it changes none (prefix-log package, DESIGN.md).

``VLLM_V41_SNAPSHOT_HIT=1`` (unset or 0: off, today's decisions exactly) adds
one hit. Today a hit must lie within ``VLLM_V41_RING_REACH`` rows of the
slot's last processed position, because the 256-row window rings hold only
the newest rows; a conversation whose last reply does not come back token for
token is read again from position 0. With the switch on every rank copies its
slot's rings to host memory right after the forward that ends a request's
prompt, and a later request that contains that whole prompt may start at the
prompt's end: the rings are copied back first. ``RingTrack`` holds, per slot,
the positions whose ring rows are exact (live rings and snapshot); the
scheduler decides every admission from it and each worker runs the same
arithmetic on the forwards it executes and raises if the two ever disagree
(replay-hit package, DESIGN.md).
"""

from __future__ import annotations

import enum
import os
from collections import deque
from copy import copy
from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, NamedTuple

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

KNOB = "VLLM_V41_MULTI_SEQ"
# Slots are lists sized by the slot count on both sides (V41SlotTable here, V41Slots and SlotOwners on the
# worker); nothing else is sized by this number. Four until 2026-11-17; eight for the twelve-card build's
# concurrency arms (each slot costs a stage 183 to 272 MiB of state plus about 160 MiB of graphs per card).
MAX_SLOTS = 8
# Slot k's name in the admission log line's field names (slot_a_free ...): lower-case letters and underscores only.
SLOT_LETTERS = "abcdefgh"
if len(SLOT_LETTERS) != MAX_SLOTS:
    raise RuntimeError(f"VLLM_V41_MULTI_SEQ: {MAX_SLOTS} slots need {MAX_SLOTS} slot letters")
# Slot k > 0 replays CUDA graphs keyed by graph_variant SLOT_VARIANT_BASE - k.
# Negative variants get their own graph memory pool (cuda_graph.py); slot 0
# keeps the default variant, so its graphs are the ones captured today.
SLOT_VARIANT_BASE = -1000
V41_MODEL_TYPE = "deepseek_v41_text"
DECODE_TURNS_KNOB = "VLLM_V41_MULTI_SEQ_DECODE_TURNS"
PREFILL_BURST_KNOB = "VLLM_V41_MULTI_SEQ_PREFILL_BURST"
PREFILL_CHUNK_KNOB = "VLLM_V41_MULTI_SEQ_PREFILL_CHUNK"
SINGLE_PREFILL_BUDGET_KNOB = "VLLM_V41_MULTI_SEQ_SINGLE_PREFILL_BUDGET"

logger = init_logger(__name__)


def multi_seq_requested() -> bool:
    """Whether the knob is set to 1. Any value other than unset, 0 or 1 is an
    error, so a typo cannot leave the operator believing the path ran."""
    value = (envs.VLLM_V41_MULTI_SEQ or "0").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{KNOB} must be 0 or 1, got {value!r}")
    if value == "0" and prefix_log_requested():
        raise ValueError(
            f"{PREFIX_LOG_KNOB}=1 needs {KNOB}=1 (it reports the per-slot "
            f"session records); unset it or turn {KNOB} on"
        )
    snapshot = snapshot_hit_requested()  # a value other than 0 or 1 raises here
    if value == "0" and snapshot:
        raise ValueError(
            f"{SNAPSHOT_HIT_KNOB}=1 needs {KNOB}=1 (the snapshot belongs to a "
            f"slot); unset it or turn {KNOB} on"
        )
    return value == "1"


# --- VLLM_V41_PREFIX_LOG: the cached-prefix decision of every admission ------
PREFIX_LOG_KNOB = "VLLM_V41_PREFIX_LOG"
# Session records overwritten by another sequence, or dropped on release, kept
# only to name the reason of a later cold read. Never read by a decision.
PREFIX_LOG_HISTORY = 4


def prefix_log_requested() -> bool:
    """Unset or 0: off; 1: on; anything else is an error."""
    value = os.environ.get(PREFIX_LOG_KNOB, "").strip()
    if value in ("", "0"):
        return False
    if value != "1":
        raise ValueError(f"{PREFIX_LOG_KNOB} must be 0 (off) or 1, got {value!r}")
    return True


# --- VLLM_V41_SNAPSHOT_HIT: a hit at the end of a conversation's last prompt --
SNAPSHOT_HIT_KNOB = "VLLM_V41_SNAPSHOT_HIT"
# attention_ops.WINDOW_RING: each ring layer keeps the row of position q in
# ring row q % 256, so a forward that ends at e leaves the rows of [e - 256, e).
RING_ROWS = 256
# attention_ops._attention_addresses: the row at position q attends q - 127 .. q.
# A forward that starts at s takes the rows of [s - 127, s) from the ring.
RING_WINDOW = 127
# prefill_skip.py (the cone): a prefill forward that ends at least
# VLLM_V41_PREFILL_SKIP_ROWS rows before the prompt's end may skip layers 21-39
# and zero their rings; the ring of layer L then holds exact rows again from
# 127 * (L - 21) rows after the skipped forward's end: 2286 for layer 39.
SKIP_CONE = RING_WINDOW * 18


def snapshot_hit_requested() -> bool:
    """Unset or 0: off; 1: on; anything else is an error."""
    value = os.environ.get(SNAPSHOT_HIT_KNOB, "").strip()
    if value in ("", "0"):
        return False
    if value != "1":
        raise ValueError(f"{SNAPSHOT_HIT_KNOB} must be 0 (off) or 1, got {value!r}")
    return True


class RingPath(enum.Enum):
    """How a request that starts at a position finds its window rows."""

    FRESH = "fresh"  # position 0: no row before it
    LIVE = "live"  # the slot's rings still hold the 127 rows before it
    RESTORE = "restore"  # the slot's host snapshot holds them: copy it back
    STALE = "stale"  # neither does: the request may not start there


class RingTrack:
    """One slot's window rings as positions: which rows are exact.

    ``[lo, end)``: the positions whose rows the live rings hold, written at
    that position by the slot's current sequence (rows at or past a request's
    accepted length are rejected drafts; the token comparison of the session
    record excludes them). ``[snap_lo, snap_end)``: the same for the host
    snapshot, taken right after the forward that ended at ``snap_end``, a
    request's prompt length; ``snap_end`` -1: none.

    The scheduler keeps one per slot and feeds it every admission and every
    scheduled forward; each worker keeps its own and feeds it the forwards it
    executes. Pure integers: no tokens, no tensors.

    ``skip_rows``: VLLM_V41_PREFILL_SKIP_ROWS. A forward that the skip rule
    may skip (it ends ``skip_rows`` or more before the prompt's end) leaves no
    exact row before ``its end + SKIP_CONE``, whether or not a worker skipped.
    """

    __slots__ = ("skip_rows", "lo", "end", "snap_lo", "snap_end")

    def __init__(self, skip_rows: int = 0) -> None:
        if skip_rows < 0 or 0 < skip_rows < SKIP_CONE + RING_ROWS:
            # the rows at a prompt's end must be exact again
            raise ValueError(
                f"{SNAPSHOT_HIT_KNOB}: skip rows {skip_rows} (0 or at least "
                f"{SKIP_CONE + RING_ROWS})"
            )
        self.skip_rows = skip_rows
        self.lo = 0
        self.end = 0
        self.snap_lo = 0
        self.snap_end = -1

    @staticmethod
    def _holds(lo: int, end: int, at: int) -> bool:
        # the rows of [at - 127, at) (from 0) are all in [lo, end)
        return 0 < at <= end and lo <= max(0, at - RING_WINDOW)

    def live_clean(self, at: int) -> bool:
        return self._holds(self.lo, self.end, at)

    def snap_clean(self, at: int) -> bool:
        return self.snap_end >= 0 and self._holds(self.snap_lo, self.snap_end, at)

    def path(self, at: int) -> RingPath:
        """The live rings first, then the snapshot: the same order everywhere."""
        if at == 0:
            return RingPath.FRESH
        if self.live_clean(at):
            return RingPath.LIVE
        if self.snap_clean(at):
            return RingPath.RESTORE
        return RingPath.STALE

    def start(self, at: int) -> RingPath:
        """A request starts in the slot at ``at`` (0: a new sequence)."""
        path = self.path(at)
        if path is RingPath.STALE:
            raise RuntimeError(
                f"{SNAPSHOT_HIT_KNOB}: a request starts at {at}, but the slot's "
                f"rings hold [{self.lo}, {self.end}) and its snapshot "
                f"[{self.snap_lo}, {self.snap_end})"
            )
        if path is RingPath.FRESH:
            self.lo = 0
            self.snap_end = -1
        elif path is RingPath.LIVE:
            # The request may leave the snapshot's prompt from ``at`` on, and
            # would then rewrite packed rows the snapshot's prompt owns.
            if at < self.snap_end:
                self.snap_end = -1
        else:
            self.lo = self.snap_lo
        self.end = at
        return path

    def forward(self, start: int, rows: int, prompt_len: int) -> bool:
        """A forward of ``rows`` rows from ``start``; True: it ended the
        request's prompt, and the rings as they now stand are the snapshot."""
        if rows <= 0 or not 0 <= start <= self.end:
            raise RuntimeError(
                f"{SNAPSHOT_HIT_KNOB}: a forward of {rows} rows from {start}, but "
                f"the slot's rings end at {self.end}"
            )
        self.end = start + rows
        self.lo = max(self.lo, self.end - RING_ROWS)
        if self.skip_rows > 0 and self.end <= prompt_len - self.skip_rows:
            self.lo = max(self.lo, self.end + SKIP_CONE)
        if self.end != prompt_len:
            return False
        self.snap_lo, self.snap_end = self.lo, self.end
        return True


class SnapshotVerdict(enum.Enum):
    """What one free slot's snapshot gives a request (snapshot_verdict)."""

    HIT = "hit"
    NONE = "none"  # no snapshot, or the pool hit leaves this conversation first
    PARTIAL_PROMPT = "partial_prompt"  # the request does not contain the snapshot's whole prompt
    POOL_SHORT = "pool_short"  # the pool's hit ends more than ring_reach before the prompt's end
    RING_STALE = "ring_stale"  # the snapshot does not hold the 127 rows before the hit


def snapshot_verdict(
    common: int,
    pool_hit: int,
    grain: int,
    ring_reach: int,
    snap_lo: int,
    snap_end: int,
) -> tuple[SnapshotVerdict, int]:
    """The snapshot hit of one slot (pure). ``common``: how many of the
    request's first tokens equal the snapshot's prompt; ``snap_end``: that
    prompt's length (-1: no snapshot). The hit is the pool hit cut to the
    prompt's end and floored to the grain; 0 unless the verdict is HIT."""
    if snap_end < 0:
        return SnapshotVerdict.NONE, 0
    if common < snap_end:
        if common >= max(pool_hit, grain):
            return SnapshotVerdict.PARTIAL_PROMPT, 0
        return SnapshotVerdict.NONE, 0
    hit = min(pool_hit, snap_end) // grain * grain
    if hit <= 0 or hit < snap_end - ring_reach:
        return SnapshotVerdict.POOL_SHORT, 0
    if not RingTrack._holds(snap_lo, snap_end, hit):
        return SnapshotVerdict.RING_STALE, 0
    return SnapshotVerdict.HIT, hit


class PrefixReason(enum.Enum):
    """Why an admitted request starts where it does. Each value names the
    predicate of KVCacheManager.get_computed_blocks or V41SlotTable.session_hit
    that decided it; checked in this order (classify_prefix)."""

    HIT = "hit"  # a hit was taken (session_hit accepted a free slot's record)
    NO_LOOKUP = "no_lookup"  # admitted with computed tokens, no local lookup
    CACHE_READ_OFF = "cache_read_off"  # caching off or skip_reading_prefix_cache
    ODD_HIT = "odd_hit"  # session_hit: hit % 2
    SLOT_BUSY = "slot_busy"  # the record that continues the prompt is in a slot that is not free
    UNEXPLAINED = "unexplained"  # a free slot passes both predicates, yet no hit: must not occur
    PAST_PROCESSED = "past_processed"  # pool hit > the record's processed length
    EVICTED = "evicted"  # the record reaches processed - ring_reach, the pool hit does not
    MISMATCH_BEYOND_REWIND = "mismatch_beyond_rewind"  # prompt leaves the record (after a block or more) too early: the hit in reach of the rings would pass processed - ring_reach
    DISPLACED = "displaced"  # the continuing record was overwritten by another sequence's release
    RECORD_DROPPED = "record_dropped"  # the continuing record was dropped on release (session_record_ok)
    NO_RECORD = "no_record"  # no slot holds a record and none was overwritten or dropped
    NO_SESSION = "no_session"  # no record, current or recent, continues this prompt
    # VLLM_V41_SNAPSHOT_HIT=1 only (never produced with the switch off):
    SNAPSHOT = "snapshot"  # a hit was taken at the end of a free slot's snapshot prompt (snapshot_verdict HIT)
    PARTIAL_PROMPT = "partial_prompt"  # a free slot holds this conversation's last prompt, the request only a leading part of it
    RING_STALE = "ring_stale"  # the record or snapshot continues the prompt, its rings no longer hold the rows before the hit
    SNAPSHOT_POOL_SHORT = "snapshot_pool_short"  # the request contains a free slot's snapshot prompt, the pool cannot serve the hit at its end


class PrefixNote(NamedTuple):
    """get_computed_blocks' result for one lookup, with the slot table as the
    session guard saw it (free flag and record of every slot, recent history)."""

    pool_hit: int | None  # None: the lookup was skipped (CACHE_READ_OFF)
    hit: int
    grain: int  # hit granularity (lcm of the KV cache groups' block sizes)
    slots: tuple[tuple[bool, tuple[list[int], int] | None], ...]
    history: tuple[tuple[str, tuple[list[int], int]], ...]
    # VLLM_V41_SNAPSHOT_HIT=1 only: every slot's ring state, the slot whose
    # snapshot the lookup took (-1: none) and whether the pool then served it.
    rings: tuple[RingNote, ...] = ()
    snapshot_slot: int = -1
    unserved: bool = False


class RingNote(NamedTuple):
    """One slot's RingTrack and prompts as the session guard saw them."""

    live_lo: int
    live_end: int
    snap_lo: int
    snap_end: int
    snap_tokens: list[int] | None  # the snapshot's prompt
    record_prompt: int | None  # prompt length of the slot's record


class SnapView(NamedTuple):
    ring_lo: int  # the live rings hold rows from here
    snap_prompt: int | None  # the snapshot's prompt length
    snap_lo: int | None  # the snapshot holds rows from here
    snap_matched: int | None  # tokens of the snapshot's prompt equal to the request's first tokens


class SlotView(NamedTuple):
    free: bool
    processed: int | None
    matched: int | None  # tokens of the record equal to the prompt's first tokens


class PrefixDecision(NamedTuple):
    slot: int
    prompt: int
    resumed: bool
    pool_hit: int | None
    hit: int
    reason: PrefixReason
    ref: str | None  # "s<k>" (slot) or "h<i>" (history) the reason refers to
    processed: int | None
    matched: int | None
    rewind: int | None  # processed - hit (taken) or processed - pool hit (cold)
    slots: tuple[SlotView, ...]
    snaps: tuple[SnapView, ...] = ()  # VLLM_V41_SNAPSHOT_HIT=1 only


def common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix (list compares, then a bisection
    that compares each element at most a few times)."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n  # a[:lo] == b[:lo]; a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def classify_prefix(
    tokens: Sequence[int],
    slot: int,
    resumed: bool,
    note: PrefixNote | None,
    ring_reach: int,
) -> PrefixDecision:
    """The reason of one admission from its lookup note (pure: data in, data out)."""
    n = len(tokens)
    if note is None:
        return PrefixDecision(
            slot, n, resumed, None, 0, PrefixReason.NO_LOOKUP, None, None, None, None, ()
        )
    views = []
    matched: list[int | None] = []
    for free, record in note.slots:
        if record is None:
            views.append(SlotView(free, None, None))
            matched.append(None)
            continue
        c = common_prefix(tokens, record[0])
        views.append(SlotView(free, record[1], c))
        matched.append(c)
    views_t = tuple(views)
    # VLLM_V41_SNAPSHOT_HIT=1: each slot's snapshot against this prompt.
    rings = note.rings
    snap_common: list[int | None] = [
        None if ring.snap_tokens is None else common_prefix(tokens, ring.snap_tokens)
        for ring in rings
    ]
    snaps_t = tuple(
        SnapView(
            ring.live_lo,
            ring.snap_end if ring.snap_end >= 0 else None,
            ring.snap_lo if ring.snap_end >= 0 else None,
            snap_common[k],
        )
        for k, ring in enumerate(rings)
    )

    def done(reason, ref=None, processed=None, c=None, rewind=None):
        return PrefixDecision(
            slot, n, resumed, note.pool_hit, note.hit, reason, ref, processed, c,
            rewind, views_t, snaps_t,
        )

    raw = note.pool_hit
    if note.hit > 0:
        reason = PrefixReason.SNAPSHOT if note.snapshot_slot >= 0 else PrefixReason.HIT
        record = note.slots[slot][1]
        if record is None:
            return done(reason, f"s{slot}" if note.snapshot_slot >= 0 else None)
        return done(reason, f"s{slot}", record[1], matched[slot],
                    record[1] - note.hit)
    if raw is None:
        return done(PrefixReason.CACHE_READ_OFF)
    if raw % 2:
        return done(PrefixReason.ODD_HIT)
    grain = note.grain

    def reach(c: int, processed: int) -> bool:
        # the record continues the prompt far enough for some hit in reach
        return (min(c, n - 1, processed) // grain * grain) >= processed - ring_reach

    def would_hit(c: int, processed: int) -> bool:
        # session_hit's predicates at the pool hit
        return processed - ring_reach <= raw <= processed and c >= raw

    found = [
        (k, free, record[1], matched[k])
        for k, (free, record) in enumerate(note.slots)
        if record is not None
    ]
    by_slot = {k: (p, c) for k, _, p, c in found}

    def snap_done(reason, k):
        p, c = by_slot.get(k, (None, None))
        return done(reason, f"s{k}", p, c, None if p is None else p - raw)

    if rings and note.unserved:
        # the lookup took a snapshot hit and the pool then did not serve it
        return snap_done(PrefixReason.SNAPSHOT_POOL_SHORT, note.snapshot_slot)

    def busy_snapshot(k: int, c: int, p: int) -> bool:
        # the busy slot's record finished its prompt, the request contains that
        # prompt and the pool hit reaches its end: a snapshot hit, were it free
        prompt = rings[k].record_prompt if rings else None
        return (
            prompt is not None
            and p >= prompt
            and snapshot_verdict(min(c, prompt), raw, grain, ring_reach, 0, prompt)[0]
            is SnapshotVerdict.HIT
        )

    for k, free, p, c in found:
        if not free and (
            (raw > 0 and would_hit(c, p)) or reach(c, p) or busy_snapshot(k, c, p)
        ):
            return done(PrefixReason.SLOT_BUSY, f"s{k}", p, c, p - raw)
    for k, free, p, c in found:
        if free and raw > 0 and would_hit(c, p):
            if rings and not RingTrack._holds(rings[k].live_lo, rings[k].live_end, raw):
                return done(PrefixReason.RING_STALE, f"s{k}", p, c, p - raw)
            return done(PrefixReason.UNEXPLAINED, f"s{k}", p, c, p - raw)
    for k, free, p, c in found:
        if free and reach(c, p) and raw > p:
            return done(PrefixReason.PAST_PROCESSED, f"s{k}", p, c, p - raw)
    for k, free, p, c in found:
        if free and reach(c, p) and raw < p - ring_reach:
            return done(PrefixReason.EVICTED, f"s{k}", p, c, p - raw)
    if rings:
        for k, ring in enumerate(rings):
            if not note.slots[k][0] or snap_common[k] is None:
                continue
            verdict, _ = snapshot_verdict(
                snap_common[k], raw, grain, ring_reach, ring.snap_lo, ring.snap_end
            )
            if verdict is SnapshotVerdict.HIT:
                return snap_done(PrefixReason.UNEXPLAINED, k)
            if verdict is SnapshotVerdict.PARTIAL_PROMPT:
                return snap_done(PrefixReason.PARTIAL_PROMPT, k)
            if verdict is SnapshotVerdict.POOL_SHORT:
                return snap_done(PrefixReason.SNAPSHOT_POOL_SHORT, k)
            if verdict is SnapshotVerdict.RING_STALE:
                return snap_done(PrefixReason.RING_STALE, k)
        for k, free, p, c in found:
            # a record that remembers its prompt, without a snapshot
            prompt = rings[k].record_prompt
            if (
                free
                and prompt is not None
                and c < prompt
                and c >= max(raw, grain)
                and not reach(c, p)
            ):
                return done(PrefixReason.PARTIAL_PROMPT, f"s{k}", p, c, p - raw)
    for k, free, p, c in found:
        if free and c >= max(raw, grain) and not reach(c, p):
            return done(PrefixReason.MISMATCH_BEYOND_REWIND, f"s{k}", p, c, p - raw)
    for i, (kind, (h_tokens, p)) in enumerate(note.history):
        c = common_prefix(tokens, h_tokens)
        if reach(c, p) or (raw > 0 and c >= raw):
            reason = (
                PrefixReason.RECORD_DROPPED if kind == "dropped" else PrefixReason.DISPLACED
            )
            return done(reason, f"h{i}", p, c, p - raw)
    if not found:
        return done(PrefixReason.NO_RECORD)
    return done(PrefixReason.NO_SESSION)


def format_prefix(req_id: str, d: PrefixDecision) -> str:
    """The log line: lower-case field names, integers (-1: none), the decision and reason words, the request id."""

    def num(x):
        return -1 if x is None else x

    ref_slot = int(d.ref[1:]) if d.ref and d.ref[0] == "s" else -1
    ref_history = int(d.ref[1:]) if d.ref and d.ref[0] == "h" else -1
    # slot 0, 1, 2 ... -> slot_a, slot_b, slot_c ...: field names stay lower-case letters and underscores
    slots = "".join(
        f" slot_{SLOT_LETTERS[k]}_free {int(v.free)} slot_{SLOT_LETTERS[k]}_processed {num(v.processed)}"
        f" slot_{SLOT_LETTERS[k]}_matched {num(v.matched)}"
        for k, v in enumerate(d.slots)
    )
    # VLLM_V41_SNAPSHOT_HIT=1: four more integers per slot, after every older field
    snaps = "".join(
        f" slot_{SLOT_LETTERS[k]}_ring_lo {v.ring_lo} slot_{SLOT_LETTERS[k]}_snap_prompt {num(v.snap_prompt)}"
        f" slot_{SLOT_LETTERS[k]}_snap_lo {num(v.snap_lo)} slot_{SLOT_LETTERS[k]}_snap_matched {num(v.snap_matched)}"
        for k, v in enumerate(d.snaps)
    )
    return (
        f"{PREFIX_LOG_KNOB} request {req_id} slot {d.slot} prompt {d.prompt} "
        f"resumed {int(d.resumed)} pool_hit {num(d.pool_hit)} hit {d.hit} "
        f"decision {'hit' if d.hit > 0 else 'cold'} reason {d.reason.value} "
        f"ref_slot {ref_slot} ref_history {ref_history} processed {num(d.processed)} "
        f"matched {num(d.matched)} rewind {num(d.rewind)}{slots}{snaps}"
    )


def is_v41_model(vllm_config: VllmConfig) -> bool:
    model_config = vllm_config.model_config
    if model_config is None:
        return False
    text_config = getattr(model_config, "hf_text_config", None)
    return getattr(text_config, "model_type", None) == V41_MODEL_TYPE


def _unsupported(vllm_config: VllmConfig) -> list[str]:
    """Every configuration this path does not cover, in words."""
    parallel_config = vllm_config.parallel_config
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config
    speculative_config = vllm_config.speculative_config
    problems = []
    max_num_seqs = scheduler_config.max_num_seqs
    if not 1 <= max_num_seqs <= MAX_SLOTS:
        problems.append(f"max_num_seqs={max_num_seqs} (supported: 1..{MAX_SLOTS})")
    if scheduler_config.async_scheduling:
        problems.append(
            "async scheduling (its PP path rebuilds requests from the history "
            "and broadcasts drafts by batch row)"
        )
    if parallel_config.pipeline_parallel_size not in (1, 2, 3):
        problems.append(
            f"pipeline_parallel_size={parallel_config.pipeline_parallel_size}"
        )
    if not vllm_config.use_v2_model_runner:
        problems.append("this concurrency port requires Model Runner V2")
    if speculative_config is None or speculative_config.method != "dspark":
        problems.append("MRV2 concurrency requires DSpark")
    elif speculative_config.num_speculative_tokens != 5:
        problems.append("MRV2 concurrency requires five DSpark drafts")
    if model_config is not None and model_config.runner_type != "generate":
        problems.append("MRV2 concurrency requires text generation")
    if vllm_config.compilation_config.cudagraph_mode.name not in ("FULL", "NONE"):
        problems.append("MRV2 concurrency supports FULL or eager graphs")
    if vllm_config.compilation_config.pass_config.enable_sp:
        problems.append("sequence parallelism")
    if parallel_config.distributed_executor_backend == "external_launcher":
        problems.append("the external_launcher backend")
    if vllm_config.kv_transfer_config is not None:
        problems.append("a KV connector")
    if vllm_config.ec_transfer_config is not None:
        problems.append("an encoder-cache connector")
    if vllm_config.lora_config is not None:
        problems.append("LoRA")
    if model_config is not None and getattr(model_config, "enable_prompt_embeds", 0):
        problems.append("prompt embeddings")
    if speculative_config is not None:
        if speculative_config.use_ngram_gpu():
            problems.append("ngram_gpu drafting (GPU token tensors per batch row)")
        if speculative_config.use_dflash_ddtree():
            problems.append("dflash_ddtree (tree payloads per batch row)")
    if int(envs.VLLM_V41_TRUNC_CUTOFF or 0):
        problems.append(
            "VLLM_V41_TRUNC_CUTOFF (decided on the previous step's batch, which "
            "can be another request)"
        )
    return problems


def _snapshot_unsupported(vllm_config: VllmConfig) -> list[str]:
    """Every configuration VLLM_V41_SNAPSHOT_HIT=1 does not cover, in words."""
    from vllm.models.deepseek_v41 import multi_rows

    problems = []
    if envs.VLLM_V41_SESSION_PREFIX != "1":
        problems.append("VLLM_V41_SESSION_PREFIX off (no session records)")
    if envs.VLLM_V41_WINDOW_RING != "1":
        problems.append("VLLM_V41_WINDOW_RING off (no window rings to copy)")
    if multi_rows.ENABLED:
        problems.append("VLLM_V41_MULTI_ROWS (several requests in one forward)")
    if envs.VLLM_V41_PACKED_KV != "1":
        problems.append("VLLM_V41_PACKED_KV off (compressed rows outside the slot's store)")
    model_config = vllm_config.model_config
    hot = int(envs.VLLM_V41_KV_HOT_POSITIONS)
    if model_config is not None and hot < model_config.max_model_len:
        problems.append(
            f"VLLM_V41_KV_HOT_POSITIONS={hot} below max_model_len="
            f"{model_config.max_model_len} (the hot ring wraps: rows past a far "
            "rewind would sit in slots the readers treat as current)"
        )
    if vllm_config.cache_config.partial_prefix_cache:
        problems.append("the partial prefix cache (hits off the block grain)")
    drafter = getattr(vllm_config, "v41_drafter", None)
    if drafter is not None and type(drafter).__name__ not in ("NotApplicable", "Local"):
        problems.append(
            f"the remote drafter backend {type(drafter).__name__} (it keeps "
            "256 context rows per slot and no snapshot of them)"
        )
    return problems


def multi_seq_decision(vllm_config: VllmConfig) -> tuple[bool, str]:
    """Return ``(enabled, reason)``; the same answer in every process.

    Knob off: ``(False, "")``. Knob on for a model other than V4.1: ``(False,
    reason)``, the knob is inert. Knob on for V4.1: ``(True, reason)``, or a
    ``ValueError`` naming every unsupported setting (fail closed: never fall
    back silently to a path the operator did not ask for).
    """
    if not multi_seq_requested():
        return False, ""
    if not is_v41_model(vllm_config):
        return False, "the model is not DeepSeek-V4.1"
    problems = _unsupported(vllm_config)
    if problems:
        raise ValueError(
            f"{KNOB}=1 does not support: "
            + "; ".join(problems)
            + f". Unset {KNOB} or change the configuration."
        )
    if snapshot_hit_requested():
        problems = _snapshot_unsupported(vllm_config)
        if problems:
            raise ValueError(
                f"{SNAPSHOT_HIT_KNOB}=1 does not support: "
                + "; ".join(problems)
                + f". Unset {SNAPSHOT_HIT_KNOB} or change the configuration."
            )
    pp_size = vllm_config.parallel_config.pipeline_parallel_size
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    overlap = (
        "PP=1: requests time-slice one stage (no overlap)"
        if pp_size == 1
        else f"{pp_size} pipeline stages overlap {pp_size} requests"
    )
    return True, (
        f"{max_num_seqs} slot(s), one request per step (M<=6 single-request "
        f"kernels), {overlap}"
    )


class DraftingCapacity(NamedTuple):
    configured_requests: int
    forward_requests: int
    single_prefill_budget: bool
    decode_requests: int


def drafting_request_capacity(vllm_config: VllmConfig) -> DraftingCapacity:
    """Requests that can add draft slots to one target forward.

    Reuses CONCBATCH1104 bf6f71c5f7. Prefill always runs one request;
    MULTI_ROWS decode can run several. Its bounded decode geometry must also
    fit the unchanged allocation before the one-request prefill reserve is
    valid. Resident capacity and token allocation are unchanged.
    """
    value = envs.VLLM_V41_MULTI_SEQ_SINGLE_PREFILL_BUDGET.strip()
    if value not in ("0", "1"):
        raise ValueError(f"{SINGLE_PREFILL_BUDGET_KNOB} must be 0 or 1, got {value!r}")
    configured = vllm_config.scheduler_config.max_num_seqs
    if value == "0":
        return DraftingCapacity(configured, configured, False, configured)
    target = vllm_config
    if vllm_config.is_speculative_draft:
        # DSpark constructs derived VllmConfigs with its draft model/TP
        # geometry but the target-owned scheduler. Revalidate the explicit
        # target owner rather than treating the draft as another target model.
        speculative = vllm_config.speculative_config
        if (
            speculative is None
            or speculative.target_model_config is None
            or speculative.target_parallel_config is None
        ):
            raise ValueError(
                f"{SINGLE_PREFILL_BUDGET_KNOB}=1 requires the draft's target config"
            )
        target = copy(vllm_config)
        target.model_config = speculative.target_model_config
        target.parallel_config = speculative.target_parallel_config
        target.is_speculative_draft = False
    enabled, reason = multi_seq_decision(target)
    if not enabled:
        raise ValueError(
            f"{SINGLE_PREFILL_BUDGET_KNOB}=1 requires the admitted "
            f"{KNOB}=1 adapter; {reason or 'the adapter is disabled'}"
        )
    from vllm.models.deepseek_v41 import multi_rows

    decode_requests = group_size(configured, multi_rows.ENABLED)
    # DSpark5: each grouped decoder has <=6 verify rows and can add four
    # drafting slots. Unlike prefill, a group never approaches the 8192-row
    # allocation. Refuse a small allocation that cannot cover that bound.
    assert target.speculative_config is not None  # admission requires DSpark5
    extra = target.speculative_config.max_num_new_target_slots_for_drafting
    decode_slots = decode_requests * (multi_rows.REQUEST_ROWS + extra)
    if target.scheduler_config.max_num_batched_tokens < decode_slots:
        raise ValueError(
            f"{SINGLE_PREFILL_BUDGET_KNOB}=1 needs {decode_slots} token slots "
            f"for {decode_requests} grouped decoder(s)"
        )
    return DraftingCapacity(configured, 1, True, decode_requests)


def _int_knob(name: str, value: str | None, default: str, minimum: int) -> int:
    text = (value or default).strip()
    try:
        number = int(text)
    except ValueError:
        number = minimum - 1
    if number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {text!r}")
    return number


class PrefillPolicy(NamedTuple):
    """A prefill beside decoding requests (``prefill_policy``)."""

    decode_turns: int = 1
    prefill_burst: int = 1
    prefill_chunk: int = 0


def prefill_policy(chunked_prefill: bool = True) -> PrefillPolicy:
    """The prefill-beside-decoders policy from the environment.

    With one request per step, a prefill step of a long prompt holds a
    pipeline stage for seconds, and a decoding request's next round waits
    behind it. Decode first: a full prefill step gives every decoding
    request ``decode_turns`` steps before the next prefill step (a shorter
    step pro rata; 1 = alternate). Up to ``prefill_burst`` prefill steps
    may run back to back first, the later ones a running request's next
    chunks (with PP2, 2 overlaps two chunks across the stages; the debt is
    the sum). ``prefill_chunk`` > 0 caps a prefill step's tokens
    while another request is decoding; 0 keeps the full token budget,
    because on SM70 every prefill step pays a large fixed cost (1Cat #1295:
    small chunks beside decoders lowered prefill speed and decode rounds
    alike).
    """
    policy = PrefillPolicy(
        _int_knob(DECODE_TURNS_KNOB, envs.VLLM_V41_MULTI_SEQ_DECODE_TURNS, "16", 1),
        _int_knob(PREFILL_BURST_KNOB, envs.VLLM_V41_MULTI_SEQ_PREFILL_BURST, "2", 1),
        _int_knob(PREFILL_CHUNK_KNOB, envs.VLLM_V41_MULTI_SEQ_PREFILL_CHUNK, "0", 0),
    )
    if policy.prefill_chunk and not chunked_prefill:
        # Without chunked prefill a prompt above the cap could not be admitted
        # while another request decodes.
        raise ValueError(f"{PREFILL_CHUNK_KNOB} > 0 needs chunked prefill")
    return policy


def slot_variant(slot: int) -> int:
    """CUDA graph variant of a slot (0 keeps today's graphs)."""
    if not 0 <= slot < MAX_SLOTS:
        raise ValueError(f"{KNOB}: slot {slot} outside 0..{MAX_SLOTS - 1}")
    return 0 if slot == 0 else SLOT_VARIANT_BASE - slot


def _session_tokens_match(request: Request, tokens: Sequence[int], hit: int) -> bool:
    return list(request.all_token_ids)[:hit] == list(tokens[:hit])


class V41SlotTable:
    """Which running request owns which slot, plus one session record per slot.

    A slot's device state continues the last sequence that ran in it, so a
    same-session prefix hit is admissible only in the slot that ran that
    session, and only while the slot is free.
    """

    def __init__(
        self,
        num_slots: int,
        ring_reach: int,
        record_sessions: bool,
        log_prefix: bool = False,
        snapshot_hits: bool = False,
        skip_rows: int = 0,
    ):
        if not 1 <= num_slots <= MAX_SLOTS:
            raise ValueError(f"{KNOB}: {num_slots} slots (supported 1..{MAX_SLOTS})")
        # VLLM_V41_SNAPSHOT_HIT: off, the table decides and records as before.
        self.snapshot_hits = snapshot_hits
        # Per slot: the positions its rings and its snapshot hold.
        self.rings: list[RingTrack] = [RingTrack(skip_rows) for _ in range(num_slots)]
        # Per slot: the prompt its snapshot ends (None with rings[k].snap_end -1).
        self.snap_tokens: list[list[int] | None] = [None] * num_slots
        # Per slot: the prompt length of records[k] (None with it).
        self.record_prompt: list[int | None] = [None] * num_slots
        # Running requests by id (their scheduled forwards move the rings).
        self.requests: dict[str, Request] = {}
        # Request id -> the free slot whose snapshot its lookup took.
        self.snap_matched: dict[str, int] = {}
        # Requests admitted by a snapshot hit, until released.
        self.snapshot_admitted: set[str] = set()
        # VLLM_V41_PREFIX_LOG: the lookup notes and the overwritten/dropped
        # records behind the log line; read by no decision.
        self.log_prefix = log_prefix
        self.notes: dict[str, PrefixNote] = {}
        self.history: deque[tuple[str, tuple[list[int], int]]] = deque(
            maxlen=PREFIX_LOG_HISTORY
        )
        self.num_slots = num_slots
        self.ring_reach = ring_reach
        self.record_sessions = record_sessions
        self.owner: list[str | None] = [None] * num_slots
        self.by_req: dict[str, int] = {}
        # Per slot: (all token ids, num computed tokens) of the last request
        # freed from it, like KVCacheManager._v41_last for one request.
        self.records: list[tuple[list[int], int] | None] = [None] * num_slots
        # Request id -> the free slot its admitted session hit continues.
        self.matched: dict[str, int] = {}
        # Per slot: when it was last freed (a count of releases). A fresh
        # prefill overwrites the record that has waited longest, so a chat
        # that has ended gives up its slot before one still taking turns.
        self.freed_at: list[int] = [0] * num_slots
        self.releases = 0

    def slot_of(self, req_id: str) -> int | None:
        return self.by_req.get(req_id)

    def free_slots(self) -> list[int]:
        return [k for k, owner in enumerate(self.owner) if owner is None]

    def session_hit(self, request: Request, hit: int) -> bool:
        """KVCacheManager._v41_session_hit, per slot. Remembers the slot."""
        self.matched.pop(request.request_id, None)
        self.snap_matched.pop(request.request_id, None)
        if hit % 2:
            return False
        for slot in self.free_slots():
            record = self.records[slot]
            if record is None:
                continue
            tokens, processed = record
            if not processed - self.ring_reach <= hit <= processed:
                continue
            if self.snapshot_hits and not self.rings[slot].live_clean(hit):
                # A request that ended before the previous one's end left rows
                # of the previous one in the ring rows before this hit.
                continue
            if _session_tokens_match(request, tokens, hit):
                self.matched[request.request_id] = slot
                return True
        return False

    def snapshot_hit(self, request: Request, pool_hit: int, grain: int) -> int:
        """VLLM_V41_SNAPSHOT_HIT, after session_hit refused the pool hit: the
        hit at the end of a free slot's snapshot prompt, or 0. Remembers the slot."""
        if not self.snapshot_hits:
            return 0
        tokens = list(request.all_token_ids)
        for slot in self.free_slots():
            prompt = self.snap_tokens[slot]
            if prompt is None:
                continue
            ring = self.rings[slot]
            verdict, hit = snapshot_verdict(
                common_prefix(tokens, prompt),
                pool_hit,
                grain,
                self.ring_reach,
                ring.snap_lo,
                ring.snap_end,
            )
            if verdict is SnapshotVerdict.HIT:
                self.matched[request.request_id] = slot
                self.snap_matched[request.request_id] = slot
                return hit
        return 0

    def snapshot_unserved(self, request: Request) -> int:
        """The pool did not serve the snapshot hit: the request reads cold.
        Returns the slot whose snapshot was taken (for the log)."""
        self.matched.pop(request.request_id, None)
        return self.snap_matched.pop(request.request_id)

    def forward(self, req_id: str, rows: int) -> None:
        """VLLM_V41_SNAPSHOT_HIT: a step of ``rows`` rows was scheduled for a
        running request, from its computed position (end_step runs before the
        scheduler advances it)."""
        request = self.requests[req_id]
        slot = self.by_req[req_id]
        if self.rings[slot].forward(
            request.num_computed_tokens, rows, request.num_prompt_tokens
        ):
            self.snap_tokens[slot] = list(
                request.all_token_ids[: request.num_prompt_tokens]
            )

    def assign(self, request: Request) -> int:
        """Give an admitted (new or resumed) request its slot."""
        req_id = request.request_id
        if req_id in self.by_req:
            raise RuntimeError(f"{KNOB}: request {req_id} already owns a slot")
        free = self.free_slots()
        if not free:
            raise RuntimeError(
                f"{KNOB}: no free slot for request {req_id} "
                f"({self.num_slots} slots, owners {self.owner})"
            )
        matched = self.matched.pop(req_id, None)
        if request.num_computed_tokens > 0:
            # A prefix hit continues one slot's device state; any other slot
            # would read another sequence's rings and packed rows.
            if matched is None or matched not in free:
                raise RuntimeError(
                    f"{KNOB}: request {req_id} starts at "
                    f"{request.num_computed_tokens} cached tokens without a "
                    "free slot that holds that session"
                )
            slot = matched
        else:
            # Fresh prefill: prefer a slot without a session record worth
            # keeping (an unused slot), else the free slot freed longest ago.
            # The lowest free slot instead made every new chat evict slot 0
            # while ended chats' records pinned the others (STALL-1107: on
            # the overnight order, one request at a time, 16% of prompt
            # tokens hit with three slots against 95% this way).
            unused = [k for k in free if self.records[k] is None]
            slot = (
                unused[0]
                if unused
                else min(free, key=lambda k: self.freed_at[k])
            )
        self.owner[slot] = req_id
        self.by_req[req_id] = slot
        if self.snapshot_hits:
            self.rings[slot].start(request.num_computed_tokens)
            if self.rings[slot].snap_end < 0:
                self.snap_tokens[slot] = None
            self.requests[req_id] = request
            if self.snap_matched.pop(req_id, None) is not None:
                self.snapshot_admitted.add(req_id)
        return slot

    def release(self, request: Request, *, record_session: bool = True) -> None:
        """Blocks freed (finish, abort, preemption): record and free the slot."""
        req_id = request.request_id
        self.matched.pop(req_id, None)
        self.notes.pop(req_id, None)  # VLLM_V41_PREFIX_LOG (empty when off)
        slot = self.by_req.pop(req_id, None)
        if slot is None:
            return
        self.owner[slot] = None
        self.releases += 1
        self.freed_at[slot] = self.releases
        if self.snapshot_hits:
            self.requests.pop(req_id, None)
            self.snap_matched.pop(req_id, None)
            if (
                req_id in self.snapshot_admitted
                and request.num_computed_tokens < request.num_prompt_tokens
            ):
                # Freed while reading the rows after its snapshot hit: no
                # ordinary hit may continue it (the snapshot itself stands:
                # the request wrote nothing before its hit).
                record_session = False
            self.snapshot_admitted.discard(req_id)
        if self.log_prefix:
            self._note_release(slot, request, record_session)
        if self.record_sessions and record_session:
            self.records[slot] = (
                list(request.all_token_ids),
                request.num_computed_tokens,
            )
            if self.snapshot_hits:
                self.record_prompt[slot] = request.num_prompt_tokens
        else:
            self.records[slot] = None
            if self.snapshot_hits:
                self.record_prompt[slot] = None

    # -- VLLM_V41_PREFIX_LOG (reads the table, changes no decision) ----------
    def _note_release(self, slot: int, request: Request, record_session: bool) -> None:
        old = self.records[slot]
        tokens = list(request.all_token_ids)
        if old is not None and (
            not (self.record_sessions and record_session)
            or common_prefix(tokens, old[0]) < old[1] - self.ring_reach
        ):
            self.history.appendleft(("overwritten", old))
        if self.record_sessions and not record_session:
            self.history.appendleft(("dropped", (tokens, request.num_computed_tokens)))

    def note(
        self,
        request: Request,
        pool_hit: int | None,
        hit: int,
        grain: int,
        snapshot_slot: int = -1,
        unserved: bool = False,
    ) -> None:
        """get_computed_blocks' result; the latest one before admission counts."""
        self.notes[request.request_id] = PrefixNote(
            pool_hit,
            hit,
            grain,
            tuple(
                (owner is None, record)
                for owner, record in zip(self.owner, self.records)
            ),
            tuple(self.history),
            tuple(
                RingNote(
                    ring.lo, ring.end, ring.snap_lo, ring.snap_end, prompt, record_prompt
                )
                for ring, prompt, record_prompt in zip(
                    self.rings, self.snap_tokens, self.record_prompt
                )
            )
            if self.snapshot_hits
            else (),
            snapshot_slot,
            unserved,
        )

    def log_admission(self, request: Request, slot: int) -> None:
        decision = classify_prefix(
            list(request.all_token_ids),
            slot,
            request.num_preemptions > 0,
            self.notes.pop(request.request_id, None),
            self.ring_reach,
        )
        logger.info("%s", format_prefix(request.request_id, decision))


_WAITING = "waiting"


class Admission(enum.Enum):
    """How MULTI_SEQ admits a request (K4d)."""

    GROUPED = "grouped"  # shares the server with the other resident requests
    EXCLUSIVE = "exclusive"  # structured output: served as the only resident request


def admission(request: Request) -> Admission:
    return Admission.EXCLUSIVE if request.use_structured_output else Admission.GROUPED


def may_admit(
    candidate: Admission, resident: Sequence[Admission], exclusive_waiting: bool
) -> bool:
    """Whether a waiting request may be admitted now. Nothing joins an
    EXCLUSIVE request; an EXCLUSIVE request enters only an empty server; while
    one waits, no GROUPED request is admitted (the server drains for it)."""
    if Admission.EXCLUSIVE in resident:
        return False
    if candidate is Admission.EXCLUSIVE:
        return not resident
    return not exclusive_waiting


def group_size(num_slots: int, multi_rows: bool) -> int:
    """Decoding requests per step: min(slots, 4) with VLLM_V41_MULTI_ROWS=1
    (KCAMP1106 K1, multi_rows.MAX_REQUESTS), else today's one."""
    return min(num_slots, 4) if multi_rows else 1


class V41SchedulerMultiSeq:
    """Scheduler side: the step policy, slots and the held record."""

    def __init__(
        self,
        num_slots: int,
        ring_reach: int,
        record_sessions: bool,
        policy: PrefillPolicy | None = None,
        token_budget: int = 8192,
        group_decode: int = 1,
        pipeline_depth: int = 1,
    ):
        policy = policy or PrefillPolicy()
        if (
            policy.decode_turns < 1
            or policy.prefill_burst < 1
            or policy.prefill_chunk < 0
            or token_budget < 1
        ):
            raise ValueError(f"{KNOB}: {policy}, token_budget {token_budget}")
        if not 1 <= group_decode <= 4:
            raise ValueError(f"{KNOB}: group_decode {group_decode} (1..4)")
        if not 1 <= pipeline_depth <= 4:
            raise ValueError(f"{KNOB}: pipeline_depth {pipeline_depth} (1..4)")
        skip_rows = 0
        if snapshot_hit_requested():
            # The workers' value (model.py: _v41_skip_rows), from the same
            # environment. Imported here: the model package is not needed off.
            from vllm.models.deepseek_v41 import prefill_skip

            skip_rows = prefill_skip.skip_rows()
        self.slots = V41SlotTable(
            num_slots,
            ring_reach,
            record_sessions,
            prefix_log_requested(),
            snapshot_hit_requested(),
            skip_rows,
        )
        if self.slots.snapshot_hits:
            if not record_sessions or group_decode != 1:
                raise ValueError(
                    f"{SNAPSHOT_HIT_KNOB}=1 needs session records and one "
                    f"request per step (records {record_sessions}, group "
                    f"{group_decode})"
                )
            logger.info(
                "%s=1 active (scheduler): a request that contains a free slot's "
                "whole last prompt may start at that prompt's end (the hit is "
                "the pool hit cut to it, floored to the grain, at most %d rows "
                "before it); every admission is decided from the slot's ring "
                "positions (%d ring rows, %d rows read before a forward, "
                "prefill skip rows %d).",
                SNAPSHOT_HIT_KNOB,
                ring_reach,
                RING_ROWS,
                RING_WINDOW,
                skip_rows,
            )
        if self.slots.log_prefix:
            logger.info(
                "%s=1 active (scheduler): one line per admitted request with its "
                "slot, the pool hit, the hit taken and the reason (numbers only); "
                "ring reach %d, session records %s.",
                PREFIX_LOG_KNOB,
                ring_reach,
                "on" if record_sessions else "off",
            )
        self.policy = policy
        self.token_budget = token_budget
        # VLLM_V41_MULTI_ROWS (KCAMP1106 K1): up to group_decode decoding
        # requests share one step (one forward, several requests' rows);
        # 1 = one request per step. Prefill steps always run one request.
        self.group_decode = group_decode
        # KCAMP1106 K1 C4: the engine's batch queue holds pipeline_depth steps
        # (the PP size) and a decoding request has at most one step in flight.
        # Up to pipeline_depth single-request steps already share the stages,
        # so decoders group only when there are more of them than that
        # (c4_cadence_sim.py, PP2 at the measured conc2 overlap cost: 2
        # decoders 156.9 tok/s alternating vs 141.1 in one group; 3 decoders
        # 159.0 in one group vs 156.9 alternating, 144.4 in groups of 2 + 1).
        self.pipeline_depth = pipeline_depth
        self.decoding_now = 0
        # This step's requests when several decode together (LRU order),
        # else () and ``turn`` names the one request.
        self.group: tuple[str, ...] = ()
        self.last_step: dict[str, int] = {}
        self.turn: str | None = None  # a running request id, _WAITING or None
        self.last_turn_running = False
        self.waiting_backoff = False
        # Prefill beside decoders: the decode steps each decoding request is
        # owed (a full prefill step adds decode_turns, each of its own steps
        # pays one), the prefill steps in a row since the last decode step,
        # and this step's choice.
        self.owed: dict[str, Fraction] = {}
        self.prefill_run = 0
        self.step_decoders: list[str] = []
        self.prefill_turn = False
        self.holding_prefill = False
        self.token_cap: int | None = None
        # The requests of the latest step that scheduled any. A request runs
        # every pipeline_depth-th step at the earliest, so while one request
        # decodes alone its step is the latest that scheduled any until its
        # next one (with several, each one's step is in flight in turn).
        self.in_flight: frozenset[str] = frozenset()
        self.held_empty_steps = 0
        # K4d admission (Admission): the resident requests' admissions and
        # whether an EXCLUSIVE request waits, as of this step's begin_step.
        self.resident: tuple[Admission, ...] = ()
        self.exclusive_waiting = False

    # -- the step policy -------------------------------------------------
    @staticmethod
    def can_run(request: Request, current_step: int) -> bool:
        """A running request with work and no step of it still in flight."""
        if current_step < request.next_decode_eligible_step:
            return False
        return (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
            > 0
        )

    @staticmethod
    def is_decoding(request: Request) -> bool:
        """Its prompt (or resumed history) is scheduled: at most the last
        sampled token is left (an in-flight verify step leaves fewer)."""
        return request.num_tokens - request.num_computed_tokens <= 1

    def _owes(self, request: Request) -> bool:
        return self.owed.get(request.request_id, 0) >= 1

    def _prefill_held(
        self,
        running: list[Request],
        candidates: list[Request],
        admit: bool,
        current_step: int,
    ) -> bool:
        """Decode first: a prefill step waits while a decoding request that
        can run, or whose step is in flight, is owed a step."""
        if not admit and all(self.is_decoding(r) for r in candidates):
            return False  # no prefill step wants to run
        owing = [
            r
            for r in running
            if self.is_decoding(r)
            and self._owes(r)
            and (self.can_run(r, current_step) or r.request_id in self.in_flight)
        ]
        if not owing:
            return False
        if self.held_empty_steps >= max(2, self.pipeline_depth):
            # A request runs every pipeline_depth-th step (the PP sampled-token
            # ring), so an owed decoder alone leaves pipeline_depth - 1 held
            # steps that schedule nothing between two of its own: one at two
            # stages, two at three. That many and one more: the request that
            # holds the prefill is neither runnable nor in flight. Never stall.
            raise RuntimeError(
                f"{KNOB}: prefill debt remains with no runnable or in-flight request"
            )
        return True

    def begin_step(
        self,
        running: Iterable[Request],
        has_waiting: bool,
        can_admit: bool,
        current_step: int,
        exclusive_waiting: bool = False,
    ) -> None:
        """Choose this step's turn: one running request or the waiting queue.

        Waiting requests get every other turn (and any turn when no running
        request can run); a waiting turn that schedules nothing hands the
        next turn to the running requests, so a full KV pool cannot stall
        the requests that would free it. Among running requests, the least
        recently scheduled goes first.

        Decode first: a prefill step (a waiting request's admission or a
        running request's prefill chunk) of t tokens, against a full step of
        T tokens (the token budget, or the chunk cap while it applies), owes
        every decoding request ``decode_turns * t / T`` steps (fractions carry
        over). A running request's prefill may first take ``prefill_burst``
        steps in a row. Then the next prefill step waits: the decoding
        requests that are owed steps go first, then the other decoding
        requests, and the step stays empty while the only one owed is in
        flight.
        """
        running = list(running)
        candidates = [r for r in running if self.can_run(r, current_step)]
        self.resident = tuple(admission(r) for r in running)
        self.exclusive_waiting = exclusive_waiting
        # No admission turn when no waiting request may enter (K4d): an
        # EXCLUSIVE request is resident, or one waits for the server to drain.
        admissible = Admission.EXCLUSIVE not in self.resident and not (
            exclusive_waiting and running
        )
        admit = has_waiting and can_admit and admissible and not self.waiting_backoff
        self.waiting_backoff = False
        self.prefill_turn = False
        self.holding_prefill = False
        self.group = ()
        prefilling = [r for r in candidates if not self.is_decoding(r)]
        self.decoding_now = sum(1 for r in running if self.is_decoding(r))
        if prefilling and 0 < self.prefill_run < self.policy.prefill_burst:
            self._choose_running(prefilling)  # the burst goes on
            self.prefill_turn = True
        else:
            self.holding_prefill = self._prefill_held(
                running, candidates, admit, current_step
            )
            if self.holding_prefill:
                decoding = [r for r in candidates if self.is_decoding(r)]
                owed = [r for r in decoding if self._owes(r)]
                self._choose_running(owed or decoding)
                # owed decoders first, then the others, LRU within each
                self._group([*self._lru(owed), *self._lru(decoding)])
            elif admit and (self.last_turn_running or not candidates):
                self.turn = _WAITING
                self.last_turn_running = False
                self.prefill_turn = True
            else:
                self._choose_running(candidates)
                self.prefill_turn = any(r.request_id == self.turn for r in prefilling)
                if not self.prefill_turn:
                    self._group(
                        self._lru([r for r in candidates if self.is_decoding(r)])
                    )
        self.step_decoders = (
            [
                r.request_id
                for r in running
                if self.is_decoding(r) and r.request_id != self.turn
            ]
            if self.prefill_turn
            else []
        )
        self.token_cap = (
            self.policy.prefill_chunk
            if self.step_decoders and self.policy.prefill_chunk > 0
            else None
        )

    def _lru(self, requests: list[Request]) -> list[Request]:
        return sorted(requests, key=lambda r: self.last_step.get(r.request_id, -1))

    def _group(self, decoders: list[Request]) -> None:
        """A decode step: up to group_decode decoders (deduplicated, in the
        given order) share it when there are at least two and more decoding
        requests than pipeline_depth (else they take turns in the stages)."""
        ids = list(dict.fromkeys(r.request_id for r in decoders))
        if (
            self.group_decode > 1
            and len(ids) > 1
            and self.decoding_now > self.pipeline_depth
        ):
            self.group = tuple(ids[: self.group_decode])
            self.turn = self.group[0]
            self.last_turn_running = True

    def _choose_running(self, candidates: list[Request]) -> None:
        if candidates:
            chosen = min(candidates, key=lambda r: self.last_step.get(r.request_id, -1))
            self.turn = chosen.request_id
            self.last_turn_running = True
        else:
            self.turn = None

    def step_token_budget(self, token_budget: int) -> int:
        """This step's token budget: capped for a prefill beside decoders."""
        if self.token_cap is None:
            return token_budget
        return min(token_budget, self.token_cap)

    def skip_running(self, request: Request) -> bool:
        if self.group:
            return request.request_id not in self.group
        return self.turn != request.request_id

    def skip_waiting(self, num_scheduled: int) -> bool:
        return self.turn != _WAITING or num_scheduled > 0

    def defer_waiting(self, request: Request) -> bool:
        """This waiting request may not be admitted now (may_admit)."""
        return not may_admit(admission(request), self.resident, self.exclusive_waiting)

    def end_step(
        self,
        scheduled_req_ids: Iterable[str],
        new_or_resumed: Iterable[Request],
        current_step: int,
        num_tokens: int = 0,
    ) -> dict[str, int]:
        """After a step of ``num_tokens`` scheduled tokens: slots for newly
        admitted requests, the LRU stamps, the decode debt and the step's
        ``v41_slots`` (request id -> slot, every scheduled one)."""
        scheduled = list(scheduled_req_ids)
        if self.group:
            foreign = [r for r in scheduled if r not in self.group]
            if foreign:
                raise RuntimeError(
                    f"{KNOB}: the scheduler scheduled {foreign} outside the "
                    f"step's decode group {self.group}"
                )
        elif len(scheduled) > 1:
            raise RuntimeError(
                f"{KNOB}: the scheduler scheduled {len(scheduled)} requests in "
                "one step; V4.1 runs one request per step"
            )
        for request in new_or_resumed:
            slot = self.slots.assign(request)
            if self.slots.log_prefix:
                self.slots.log_admission(request, slot)
        if self.slots.snapshot_hits:
            for req_id in scheduled:  # one request (checked above)
                self.slots.forward(req_id, num_tokens)
        if self.turn == _WAITING and not scheduled:
            self.waiting_backoff = True
        if scheduled:
            self.in_flight = frozenset(scheduled)
            self.held_empty_steps = 0
            if self.prefill_turn:
                self.prefill_run += 1
                full = self.token_cap or self.token_budget
                debt = Fraction(self.policy.decode_turns * num_tokens, full)
                for decoder in self.step_decoders:
                    self.owed[decoder] = self.owed.get(decoder, 0) + debt
            else:
                self.prefill_run = 0
                for req_id in scheduled:
                    if req_id in self.owed:
                        self.owed[req_id] = max(Fraction(0), self.owed[req_id] - 1)
        elif self.holding_prefill:
            self.held_empty_steps += 1
        else:
            self.held_empty_steps = 0
        slots = {}
        for req_id in scheduled:
            slot = self.slots.slot_of(req_id)
            if slot is None:
                raise RuntimeError(f"{KNOB}: scheduled request {req_id} has no slot")
            slots[req_id] = slot
            self.last_step[req_id] = current_step
        return slots

    def forget(self, req_id: str) -> None:
        self.last_step.pop(req_id, None)
        self.owed.pop(req_id, None)

    @staticmethod
    def held_record(
        previous: set[str], scheduled: Iterable[str], running_ids: set[str]
    ) -> None:
        """The runner holds every running request it has seen (in its batch
        or parked); a preempted or finished request leaves the record, so a
        resumed request is never treated as held. Updates ``previous``."""
        previous.update(scheduled)
        previous.intersection_update(running_ids)
