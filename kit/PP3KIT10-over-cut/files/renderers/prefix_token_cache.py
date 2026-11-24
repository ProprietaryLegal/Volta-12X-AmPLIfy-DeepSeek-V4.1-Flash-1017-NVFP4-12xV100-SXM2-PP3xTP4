# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_PREFIX_TOKEN_CACHE: tokenize only the new part of a prompt that continues an earlier one.

A grown chat's next request renders to the previous prompt's text with its end changed (in the build's own use nearly every
request begins with the whole previous message list). Today the API server tokenizes the whole text again: about
2.4 us per token on one host core (the build's own tokenizer: 304 ms for a 129,597-token prompt), 90 % of what
a turn waits for before the scheduler sees it.

THE RULE. A cut is usable only at an added (special) token S whose COMPLETE string lies inside the common prefix of
the cached text and the new text: old[:c+len(S)] == new[:c+len(S)], S at offset c in both. Then
ids(new) == ids(old)[:t] + encode(new[c:]), t the index of S's id in the cached ids. Why, for this tokenizer
(model tokenizer.json, checked when the cache is built; any failing check refuses the switch at the first use):
  1. The tokenizer extracts added tokens from the raw text first (leftmost-longest match), then normalizes and
     pre-tokenizes each piece between them independently; ids are the concatenation of the pieces' ids.
  2. No added token has lstrip, rstrip or single_word set: a match depends only on the token's own characters (no
     whitespace absorbed, no look at the neighbours). The normalizer is empty: matching sees the raw text.
  3. No added token occurs inside another at an offset > 0, and none is a prefix of another. So with S complete at c in
     both texts, both match S at c, no match starting before c reaches past c, and matching before c sees identical
     text: the pieces before c, hence their ids, are the same in both; the full encode of new splits at c.
  4. Each piece is pre-tokenized alone (ByteLevel without a prefix space; no state crosses a split), so
     encode(new[c:]) equals the tail of encode(new) from S on.
The earlier rule (front.patch, front2.patch) took a cut whose START lay in the common prefix; the new text need not
have S there ('test<think>one' then 'testing'): wrong ids, sometimes of equal length (found by an independent review, item A5).

Values: unset or 0 = off (today's path); 1 = on; verify = on, and every served request is checked against a full
tokenization with the call's own arguments (the full one is returned; a DIFFERENT result is evicted, so it never
becomes a parent), one log line per request (ids compared, ids reused: 0 = a miss), and a summary line every
SUMMARY_EVERY requests and at exit (hits, misses, reused ids, differences).
Served only: calls that add no special tokens and whose other arguments are truncation / max_length (the joined ids
must not be longer than max_length, else today's call runs). A MISS is exactly today's call (the caller's own
encode with its own arguments), then the result is stored. One cache per tokenizer object.
Entries: a hit whose common prefix reaches the entry's last added token (a chat's next turn) REPLACES that entry;
other hits (a shared opening, an earlier branch, an edit) and misses are added beside; the oldest of ENTRIES goes.
Reuse under MIN_REUSE tokens is a miss. Ids are kept as array('i'). The lock covers lookup and store only: the
tokenizer runs outside it.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
from array import array
from collections import OrderedDict
from collections.abc import Callable, Sequence

from vllm.logger import init_logger

logger = init_logger(__name__)

KNOB = "VLLM_V41_PREFIX_TOKEN_CACHE"
ENTRIES = 8          # conversation chains kept: four slots' chats and the side conversations beside them
MIN_REUSE = 1024     # tokens: a shorter shared part (every prompt opens with the same few tokens) is a miss
SUMMARY_EVERY = 100  # requests between two summary lines


def mode_from_env(raw: str | None) -> str:
    value = (raw or "0").strip()
    if value not in ("0", "1", "verify"):
        raise ValueError(f"{KNOB} must be 0, 1 or verify, got {value!r}")
    return value


def common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix of two strings (slice compares and a bisection: pure)."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def served_max_length(kwargs: dict) -> int | None | bool:
    """The max_length to honour if these encode kwargs are served (no special tokens added; nothing but truncation /
    max_length besides); False if they are not."""
    if kwargs.get("add_special_tokens", True) or set(kwargs) - {"add_special_tokens", "truncation", "max_length"}:
        return False
    if kwargs.get("truncation"):
        return kwargs.get("max_length") if kwargs.get("max_length") is not None else False
    return None


def check_tokenizer_json(spec: dict) -> list[str]:
    """The reasons this tokenizer breaks the rule's premises (points 2 and 3 of the module text); [] if none."""
    bad = []
    added = spec.get("added_tokens", [])
    for a in added:
        for flag in ("lstrip", "rstrip", "single_word"):
            if a.get(flag):
                bad.append(f"added token {a['content']!r} has {flag}")
    norm = spec.get("normalizer")
    if norm is not None and not (norm.get("type") == "Sequence" and not norm.get("normalizers")):
        if any(a.get("normalized") for a in added):
            bad.append(f"normalizer {norm.get('type')} applies to normalized added tokens")
    texts = [a["content"] for a in added]
    for t in texts:
        for s in texts:
            if s != t and (s in t[1:] or t.startswith(s)):
                bad.append(f"added token {s!r} occurs inside or at the start of {t!r}")
    return bad


def _trie_regex(words: Sequence[str]) -> str:
    """One regex matching any of the words, built as a trie (2.4 ms against 7.1 ms for a plain alternation of the
    1,283 added tokens over 500,000 characters, one host core). With no word a prefix of another (check_tokenizer_json),
    a match is the unique word at that position, as the tokenizer's leftmost-longest matcher finds it."""
    root: dict = {}
    for w in words:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node[""] = {}

    def rx(node: dict) -> str:
        alts = [re.escape(ch) + rx(sub) for ch, sub in sorted(node.items()) if ch]
        if not alts:
            return ""
        body = alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"
        return "(?:" + body + ")?" if "" in node else body
    return rx(root)


class _Entry:
    __slots__ = ("text", "ids", "cuts")

    def __init__(self, text: str, ids: array, cuts: list[tuple[int, int, int]]):
        self.text, self.ids, self.cuts = text, ids, cuts   # cuts: (char start, char end, token index)


class PrefixTokenCache:
    def __init__(self, cut_tokens: Sequence[str], cut_ids: Sequence[int], entries: int = ENTRIES):
        if not cut_tokens:
            raise ValueError(f"{KNOB}: the tokenizer has no added tokens to cut at")
        self._pattern = re.compile(_trie_regex(cut_tokens))
        self._cut_ids = frozenset(cut_ids)
        self._entries: OrderedDict[int, _Entry] = OrderedDict()
        self._max = entries
        self._next = 0
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "tokens_reused": 0, "tokens_encoded": 0, "evictions": 0,
                      "different": 0}

    def _cuts(self, text: str, ids: Sequence[int], c0: int = 0, t0: int = 0) -> list[tuple[int, int, int]] | None:
        """(char start, char end, token index) of every cut token from c0 / t0 on, paired in order; None if they do
        not pair up (then the text is not cached)."""
        spans = [(m.start(), m.end()) for m in self._pattern.finditer(text, c0)]
        cid = self._cut_ids
        toks = [i for i in range(t0, len(ids)) if ids[i] in cid]
        if len(spans) != len(toks):
            return None
        return [(a, b, i) for (a, b), i in zip(spans, toks)]

    def _best(self, text: str):
        """(key, entry, char cut, token cut, continues) of the cut reusing the most tokens, or None. A cut is usable
        only if its COMPLETE added token lies in the common prefix (char end <= common prefix)."""
        best = None
        for key, e in self._entries.items():
            n = common_prefix_len(text, e.text)
            lo, hi = 0, len(e.cuts)
            while lo < hi:   # the last cut whose END is within the common prefix (ends increase)
                mid = (lo + hi) // 2
                if e.cuts[mid][1] <= n:
                    lo = mid + 1
                else:
                    hi = mid
            if lo == 0:
                continue
            c, _, t = e.cuts[lo - 1]
            if t < MIN_REUSE:
                continue
            if best is None or t > best[3]:
                best = (key, e, c, t, n >= e.cuts[-1][0])
        return best

    def _add(self, text: str, ids: list[int], cuts) -> int | None:
        if cuts is None:
            return None
        key = self._next
        self._entries[key] = _Entry(text, array("i", ids), cuts)
        self._next += 1
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)
            self.stats["evictions"] += 1
        return key

    def tokenize(self, text: str, encode_piece: Callable[[str], list[int]], full_call: Callable[[], list[int]],
                 max_length: int | None) -> tuple[list[int], int, int | None]:
        """(ids, tokens reused, stored entry key). A hit joins the cached prefix and encode_piece(rest); a miss, or a
        hit whose joined ids are longer than max_length, returns full_call(), today's call, and stores its result."""
        with self._lock:
            found = self._best(text)
            if found is not None:
                key, e, c, t, continues = found
                head = e.ids[:t].tolist()
                base_cuts = [x for x in e.cuts if x[0] < c]
        if found is not None:
            ids = head + list(encode_piece(text[c:]))
            if max_length is None or len(ids) <= max_length:
                more = self._cuts(text, ids, c, t)
                with self._lock:
                    if continues and key in self._entries:
                        del self._entries[key]   # this chain's next prompt replaces the entry it continued
                    new = self._add(text, ids, None if more is None else base_cuts + more)
                    self.stats["hits"] += 1
                    self.stats["tokens_reused"] += t
                    self.stats["tokens_encoded"] += len(ids) - t
                return ids, t, new
        ids = list(full_call())
        cuts = self._cuts(text, ids)
        with self._lock:
            new = self._add(text, ids, cuts)
            self.stats["misses"] += 1
            self.stats["tokens_encoded"] += len(ids)
        return ids, 0, new

    def drop(self, key: int | None) -> None:
        """Remove a stored entry (verify: a result that differed must never become a parent)."""
        with self._lock:
            if key is not None:
                self._entries.pop(key, None)
            self.stats["different"] += 1

    def held_bytes(self) -> int:
        with self._lock:
            return sum(e.ids.itemsize * len(e.ids) + len(e.text.encode("utf-8")) for e in self._entries.values())

    def summary(self) -> str:
        with self._lock:
            s = dict(self.stats)
        return (f"{KNOB} summary: hits {s['hits']}, misses {s['misses']}, ids reused {s['tokens_reused']}, ids "
                f"tokenized {s['tokens_encoded']}, evictions {s['evictions']}, different {s['different']}")


_CACHES: dict[int, PrefixTokenCache] = {}
_CACHES_LOCK = threading.Lock()


def cache_for(tokenizer) -> PrefixTokenCache:
    with _CACHES_LOCK:
        key = id(tokenizer)
        if key not in _CACHES:
            spec = json.loads(tokenizer.backend_tokenizer.to_str())
            bad = check_tokenizer_json(spec)
            if bad:
                raise RuntimeError(f"{KNOB}: this tokenizer breaks the cut rule's premises: {'; '.join(bad[:5])}")
            added = [a["content"] for a in spec["added_tokens"]]
            cache = PrefixTokenCache(added, tokenizer.convert_tokens_to_ids(added))
            _CACHES[key] = cache
            atexit.register(lambda c=cache: logger.info("%s", c.summary()))
        return _CACHES[key]


MODE = mode_from_env(os.environ.get(KNOB))
