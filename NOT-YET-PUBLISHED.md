# Not yet published

This first release leaves out the files below. Each names the host's layout, private working records or figures
that are not ours to publish, and cleaning it properly takes more than this release allowed. They will follow in
later releases where they can be made public.

- `documents/plans/PLAN.md`: the build plan with every window's acceptance rule, prediction and result
  ("Amendment N" in the documents names its entries). Every result is in `docs/measurements.md` and
  `docs/history.md`; the rules themselves are not yet cleaned.
- `documents/reviews/KERNEL-INVENTORY-20261117.md`: the 79-row inventory of kernels and switches; most rows cite
  private working records.
- `window/run_window_pp3.sh` and every `window/chain_*.sh`, `window/serving_lib.sh`, `window/same_answers.py`: the
  window driver and its chains; they name the host's paths, the eight-card build's launcher and its device layout.
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
- The conversation store's kit and tools (a separate private working repository).
- `engine-patches/`: the first kit's change as patches; superseded by the kits under `kit/`.
- The copied Volta and kernel research: published separately in the research hub,
  [volta-v100-research](https://github.com/ProprietaryLegal/volta-v100-research).
