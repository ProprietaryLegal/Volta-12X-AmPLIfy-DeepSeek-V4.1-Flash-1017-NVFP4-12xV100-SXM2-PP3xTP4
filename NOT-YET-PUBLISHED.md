# Not yet published

The releases so far (the fifth on 2026-11-24) leave out the files below. Each names the host's layout, private working
records or figures that are not ours to publish, and cleaning it properly takes more than a release allowed. They will
follow in later releases where they can be made public.

- `documents/plans/PLAN.md`: the build plan with every window's acceptance rule, prediction and result
  ("Amendment N" in the documents names its entries). Every result is in `docs/measurements.md` and
  `docs/history.md`; the rules themselves are not yet cleaned.
- `documents/reviews/KERNEL-INVENTORY-20261117.md`: the 79-row inventory of kernels and switches; most rows cite
  private working records.
- `window/run_window_pp3.sh` and every `window/chain_*.sh` (the chains of windows w1 to q1 and, since the second
  release, the store's chains of st5 to st7, the C2 chain of window c2a, the C3 chains of c3a and c3b, the C7 chain of c7a,
  the bench chains, since the fourth release the B4 chain of window b4a and the B3 chain of window b3a, and since the
  fifth the all-on chain of x2, the cached-count and partial-prefix chains, the requalification chain of z9a, aw6 and aw7
  and the one-change-per-boot chain of aw1 to aw5, with their rules and window specifications), `window/serving_lib.sh`, `window/same_answers.py`, `window/nccl_probe.py`: the window driver and its chains; they
  name the host's paths, the eight-card build's launcher and its device layout.
- `tools/closing_bench.py` and `tools/prefix_bench.py` with their judges and tests: the bench and the prefix-sharing
  measure of 2026-11-18 and 2026-11-19. They read the host's serving record, kit and adopted digest between groups and
  carry the prompt corpus's paths; their arithmetic is described in `docs/measurements.md` ("Closing bench,
  2026-11-19, second").
- `tools/hits_closeness.py` and its test: the measure of a cache hit against its own cold read (windows b4a and b3a);
  it reads the bench's prompt corpus in place on the host.
- `tools/run_allreduce_probe.sh` and the probe it wraps (window b3a's step 0, the exact sum's cost on the board): bound
  to the host's device layout.
- The C7 re-cut tools and the pack manifest they write: they read the private expert packs on the host and write
  288.8 GB of re-cut packs; host-bound. The small library's source is published under
  `kit/PP3KIT9U-over-cut/lib/v41qpn576/` and builds from pinned engine sources.
- Kit `PP3KIT9P-over-cut`, the first cut of the lean hand-off (window c3a: did not boot, a hidden size assumed); kit 9Q
  is its correction and is published.
- `run/swap_arm12.sh`: the launcher; it derives the device order from the host's PCI addresses.
- `tools/build_tree.sh`, `tools/update_tree_*.sh`, `tools/precheck_import.sh`, `tools/make_p18042_profile.sh`,
  `tools/after_power_trip.sh`, `tools/run_allreduce_probe.sh`, `tools/run_gemm_split_probe.sh`: installers and
  host-side wrappers bound to the host's layout.
- `tools/make_*profile*.py`, `tools/make_pp3_tier_map.py`: profile and tier-map makers; they read a private routing
  tally.
- `tools/mem_rehearsal.py`, `tools/turn_probe.py`, `tools/pipe_probe.py`, `tools/long_capture.py`,
  `tools/restore_pair.py`, `tools/gen_mtp_profile_lines.py`, `tools/cold_copy_probe.py`, `tools/rc_probe.py`, `tools/rc_cells_probe.py` and their tests: the memory rehearsal, the turn probes
  and the long-prompt capture; their constants and test cuts are not yet cleaned. `tools/closeness.py` still names
  `long_capture.py`'s seven long prompt sizes.
- The tests that read real window receipts in place on the host (`test_arm_g`, `test_default42`,
  `test_serve_checked`, `test_strict_judges*`, `test_timebudget_parse`, `test_turn_probe_prompts`,
  `test_tier_lines`, `test_eff_rows`, `test_mem_rehearsal`, `test_slots_cells`, `test_arm_p`, the `sim_*`
  dry runs) and the `Makefile` of the working repository.
- The conversation store's own modules, in kits 9X, 9Y, 9Z, 10 and 11 (eight rows each:
  `entrypoints/serve/v41_store/__init__.py` and `api_router.py`, `models/deepseek_v41/kv_store.py`, `store_dump.py`,
  `store_io.py` and `store_worker.py`, `v1/core/sched/v41_store.py` and `v41_store_files.py`), and its tests
  (`tests/v41_partial/test_store_fingerprint.py` in 9Y, `tests/v41_stats/test_store_stats_real.py` in 9Z, 10 and 11).
  The store serves since window x2 and its hooks in the published scheduler, worker and model files are in the kits; its
  modules stay private because they came from a separate private working repository and its tests read the store's
  real index and records in place. Each kit's `MANIFEST` keeps the withheld rows with their sha256, and each kit's
  README lists them. Kit 9Z's only two changed files are store modules, so its folder holds its `MANIFEST`, README and
  the files it shares with 9X.
- `apply.sh` of kits 9X, 9Y, 9Z, 10 and 11: its default staging argument names a folder on the host. `rollback.sh` and
  the `MANIFEST` are published; earlier kits' `apply.sh` shows the form.
- The compiled libraries the kits install: `v41qpn576.so` with its sha256 file (9X to 11, built from the published
  source under `lib/v41qpn576/`) and `v41mhc.so` (10 and 11, built from `lib/v41mhc/v41mhc.cu`). Their rows stay in the
  `MANIFEST`s; this repository holds no compiled libraries.
- Kit 9Y's `tests/v41_partial/test_partial_prefix.py`: its cases replay the shape of a real request.
- Tools of the fifth release's windows: `tools/amp_spec.py`, `tools/amp_measure.py`, `tools/amp_quality_regrade.py`
  and `tests/test_amp_quality.py` (the window specification parser, the measuring arm and the quality regrade, part of
  the withheld driver); `tools/allon_reads.py` and `tools/allon_witness.py` (x2's read and witness tools; host paths
  and a withheld probe); `tools/h1_cached_usage.py` (the cached-count check; it imports the withheld bench and hit
  measure); `tools/h3_partial_prefix.py` and its test (kit 9Y's window tools, never run); `tools/serving_tag_middleware.py`,
  `tools/update_tree_serving_tag.sh` and their test (the serving-tag header installed into the withheld launcher);
  `tools/l4_window_receipt.py` and its test (the drafter-tail kit, closed and unpublished); and the tests that read the
  campaign's receipts in place (`test_k10_window`, `test_k11_window`, `test_allon`, the `check_amp_*` and `sim_*` dry
  runs). The receipt counters of kits 10 and 11 are published with a test on aw7's real boot lines.
- The closed levers' kits and tools: the drafter's prompt tail (closed in aw5), the 4-bit drafter query pack and its
  converter (`tools/convert_v41dq4.py`, `tools/v41dq4_format.py`; closed on the bench card), the group-wave expert
  prefill port, the tensor-core prompt index scorer, the batched prompt top-k, the speculative tree's counter, and the
  first mHC kernels that failed the keep line. Their figures are in [README.md](README.md) and `docs/measurements.md`.
- `engine-patches/`: the first kit's change as patches; superseded by the kits under `kit/`.
- The copied Volta and kernel research: published separately in the research hub,
  [volta-v100-research](https://github.com/ProprietaryLegal/volta-v100-research).
