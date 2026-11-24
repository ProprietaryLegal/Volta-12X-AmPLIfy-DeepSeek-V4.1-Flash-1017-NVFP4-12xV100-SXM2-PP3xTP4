# PP3KIT11-over-cut

Kit 10 plus an early control header on the pipeline hand-off, on by default (`VLLM_V41_PP_CONTROL_AHEAD=0` is kit 10).
Requalified in window aw7 and serving since 2026-11-24 15:57:15Z.

**What it changes.** With the lean hand-off on, a stage now sends a small control header (schedule, width, sequence)
over Gloo ahead of the NCCL payload, and the receiver checks it before the payload, instead of after it. Four engine files change: the switch parser in `envs.py`, the hand-off in
`v1/worker/gpu/v41_pp_handoff.py` (with one witness line on the first transfer), its wiring in `v1/worker/gpu_worker.py`,
and the store's fingerprint exclusions, so kit 11 opens the record set kit 10 saved. The lean hand-off is required: a
worker that has `VLLM_V41_PP_LEAN_HANDOFF=0` must also have `VLLM_V41_PP_CONTROL_AHEAD=0`, or it refuses to boot.
`VLLM_V41_PP_HOSTGAP_TIMING` (default 0) is a diagnostic. With the switch at 0 the real worker entry point emitted
transport payloads identical to kit 10's on all twelve ranks (CPU, Gloo).

**Speed.** ESTIMATE 0.20 to 0.30 ms saved per decode round of about 25.1 ms if both header delays are exposed (0.8 to 1.2
%), under the 1.8 % adoption band; installed on the owner's decision on identity and safety. Window aw7 measured every
decode cell within 0.25 % of window aw6 on kit 10 (one boot each): the change is below what the cells resolve.

**Files.** `MANIFEST` has 55 rows and is complete. Withheld: the mHC library `v41mhc.so` (kit 10's bytes, built from
`lib/v41mhc/`), the store's eight modules, the packed experts' compiled library and `apply.sh`, as in kits 10 and 9X.
`tests/v41_hostgap/test_pp_control_ahead.py` runs the hand-off on CPU ranks over Gloo.

**Published copy.** As kit 9X: reworded comment lines, their sha256 recomputed in `MANIFEST`.
