# PP3KIT9Z-over-cut

Kit 9X with the conversation store's per-step statistics taken from counts its index keeps, instead of a walk over every
entry and chunk on every scheduler step. No switch: the statistics are the same dictionary as kit 9X's. Served from
2026-11-23 on the twenty lines; superseded by kit 10 in window aw6 (2026-11-24 12:28:30Z).

**Why.** On 2026-11-22 the server ran about seven times slower than its 2026-11-19 figures: as the store's conversation
tier filled to its 400 GiB disk budget, a lone request's decode round went from about 26 ms to 165 ms and the engine
core's own turn from about 9 ms to 55 ms (MEASURED, the engine's loop counters). With the tier moved aside the round was
24 ms again (window fx1, [docs/benchmarks.md](../../docs/benchmarks.md#speed)). The scheduler asks the store for its
statistics on every step, and kit 9X built a full listing of the index only to count entries by kind and tier: 58.3 ms a
call on that tier's real index, against 0.0029 ms for kit 9Z's kept counts (MEASURED on a CPU). Only
`stats()` changed; the store's listing route answers as before.

**Files.** `MANIFEST` has 52 rows and is complete. The kit's two changed files are both store modules
(`v1/core/sched/v41_store.py`, `v1/core/sched/v41_store_files.py`) and are withheld with the other six store modules and
`apply.sh`, as in kit 9X (`README.md` of `PP3KIT9X-over-cut`). So is `tests/v41_stats/test_store_stats_real.py`: it
builds the index from the real conversation tier's entry manifests. Kit 9X's `tests/v41_idx` and `tests/v41_usage` run on
9Z as on 9X.

**Published copy.** As kit 9X: reworded comment lines, their sha256 recomputed in `MANIFEST`.
