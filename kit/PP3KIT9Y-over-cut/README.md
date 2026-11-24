# PP3KIT9Y-over-cut

Kit 9X plus the partial-prefix hit, behind `VLLM_V41_PARTIAL_PREFIX` (unset or `0`: kit 9X's behaviour; `1`: the hit;
any other value, or `1` without `VLLM_V41_SNAPSHOT_HIT=1`, refuses the boot). Built and CPU-tested on 2026-11-19; it
has not run on the cards and does not serve.

**What it changes.** Kit 9X serves a slot's saved state only to a request that contains the slot's whole last prompt; a
request that is a leading part of it reads cold although the prefix pool matched most of it, because the slot's rings
hold the rows before the prompt's end, not before the hit. With the switch on, each worker copies a slot's rings to host
memory after every prompt forward that ends at a multiple of 4,096 rows short of the prompt's end (with the prefill skip
on, at least the skip's rows before it); a later request that leaves the slot's prompt starts at the largest such
checkpoint at or below the shared length that leaves at least the skip's rows to read. The admission row then says
`decision hit reason partial_checkpoint`. Five files change: the scheduler's slot table and ring track
(`v1/core/sched/v41_multi_seq.py`), the pool re-lookup of a hit below the pool hit (`v1/core/kv_cache_manager.py`), the
worker's copy and restore (`v1/worker/gpu/model_states/deepseek_v41.py`, `models/deepseek_v41/slots.py`), and the
store's fingerprint exclusion of the switch (`v1/core/sched/v41_store_files.py`, withheld below), so a boot with it
opens the record set the boot before it saved. ESTIMATE from measured ring shapes: 39.0 MiB of host memory per
checkpoint per slot over the twelve workers. Only a card can check the worker's copy and restore.

**Files.** `MANIFEST` has 52 rows and is complete. Withheld: the store's eight modules and `apply.sh`, as in kit 9X
(`README.md` of `PP3KIT9X-over-cut`); `tests/v41_partial/test_store_fingerprint.py`, a test of a store module; and
`tests/v41_partial/test_partial_prefix.py`, whose cases are built on the shape of a request of the build's own use.

**Published copy.** As kit 9X: reworded comment lines, their sha256 recomputed in `MANIFEST`.
