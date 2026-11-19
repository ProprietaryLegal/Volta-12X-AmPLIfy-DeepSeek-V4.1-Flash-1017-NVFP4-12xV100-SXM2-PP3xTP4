# Engine changes of this build

The build serves from its own copy of the engine of our eight-card build, Volta Max uPLIft (commit `f20d155`: the
baseline `a7b0ebf` plus that build's draft-cut kit). It descends from 1Cat-vLLM
[#1277](https://github.com/1CatAI/1Cat-vLLM/pull/1277) through the PP2 x TP4 port of Volta sPLIt and the native
expert path of Volta Max uPLIft. Every change for twelve cards reaches that tree as a **kit**: a folder with
`MANIFEST` (path, sha256 of the file it replaces, sha256 of the new file), `apply.sh SRC VENV BACKUP`,
`rollback.sh` and `files/`. A kit is applied to both the engine's source tree and the venv's site-packages,
verified file by file, and can be taken out byte for byte.

The kit files are whole engine files. Their base is the eight-card engine, which is not published as a tree; the
`MANIFEST`'s base sha256 identifies each file it replaces. In this published copy, comment lines that named a
private working repository, host, folder or reviewer were reworded (kit 8: the scheduler and model-runner files;
kits 9L, 9M and 9N: those two, the token-cache renderer and the multi-sequence scheduler file, and in 9N the two
model files, the test, the library source and `apply.sh`; kits 9Q, 9S, 9T, 9R and 9U: the same files, and in 9R and
9U two expert files whose comments named a plan item by a private name; kit 9V: the same six files as 9S, its four new files unchanged), and the `MANIFEST` new-file sha256 of each reworded kit
file is recomputed (a header line in each `MANIFEST` says so), so the digests differ from the serving tree's by those
comment bytes only; the code is unchanged.

| Kit (`kit/`) | Engine commit | What it is | State |
|---|---|---|---|
| `PP3KIT9S-over-cut` | 9733024 + C2 | Kit 9N's 17 files plus `models/deepseek_v41/prefill_dense_blas.py` (new; `dense_tm.py` and `model.py` changed): the launch option `VLLM_V41_WO_A_BATCHED_4K` (default off), the attention output projection of a 4,092-row prompt chunk in one batched launch (item C2 of the kernel plan) | **Serving** since 2026-11-19 03:55:51Z with ten adopted lines (window c2a), since 13:13:21Z with the eleven below (window b4a); the way back of windows c3b, c7a and b3a |
| `PP3KIT9V-over-cut` | 9S + B3 | Kit 9S plus four files, Python only: `distributed/device_communicators/{collective_provider.py, cuda_communicator.py, custom_all_reduce.py}` and `envs.py`: the launch option `VLLM_V41_TP_EXACT_SUM` (default off), the four-card sum of a prompt read done as the engine's one-stage custom all-reduce in 8 MiB tiles, fixed rank order, FP32 accumulate, rounded once, so the sum no longer depends on the chunk's row count (item B3 of the kernel plan) | Window b3a (2026-11-19 13:45Z to 14:08Z): with the switch unset identical 7 of 7 to the eleven-line capture; with it set CLEAN but FAILED its closeness line (identical 0 of 7, median first divergence 13 where 25 were asked); on the board the exact sum costs 2.2 times the library's on a large call. Not in what serves |
| `PP3KIT9T-over-cut` | 9S + C3 | Kit 9S plus kit 9Q's three worker files: `v1/worker/gpu/v41_pp_handoff.py` (new), `gpu_worker.py` and `model_runner.py`: the launch option `VLLM_V41_PP_LEAN_HANDOFF` (default off), a leaner pipeline hand-off between stages, its geometry read from the model | Window c3b (2026-11-19 04:32Z to 05:02Z): both boots clean and identical 7 of 7; the switch FAILED its line (the round at 128K 0.133 ms shorter where 0.3 was asked). Not in what serves |
| `PP3KIT9Q-over-cut` | 9N + C3 | Kit 9N plus the same three worker files: the lean hand-off cut over 9N; it replaced kit 9P, whose window c3a did not boot (a hidden size of 4,096 assumed where the model's is 5,120) | Superseded by 9T when the serving kit moved to 9S before its window ran |
| `PP3KIT9U-over-cut` | 9S + C7 | Kit 9S plus `model_executor/layers/fused_moe/experts/expert_pack576.py` (new), `nvfp4_8r_native.py` and `nvfp4_8r_swiglu.py`: the launch option `VLLM_V41_EXPERT_PACKED576` (default off), expert records re-cut without their zero padding (item C7), with the source of its small library under `lib/v41qpn576/` (built from pinned engine sources, no CUDA device needed) | Window c7a (2026-11-19 05:06Z to 05:39Z): identical 7 of 7 and 756 / 878 / 878 MiB less a card by stage, but the decode rounds LONGER (+0.095 and +0.155 ms at 128K): FAILED its line. Not in what serves |
| `PP3KIT9R-over-cut` | 9N + C7 | Kit 9N plus the same three expert files and library source: the packed experts cut over 9N | Superseded by 9U when the serving kit moved to 9S before its window ran |
| `PP3KIT9N-over-cut` | 9733024 | Kit 9M's 15 files plus `models/deepseek_v41/index_tiled.py` and `attention_ops.py`: the launch option `VLLM_V41_INDEX_TC` (default off), the sparse-attention index scorer on the tensor cores, with its measuring library's source under `lib/v41idx3/` (built on the host in 49 s) and its tests under `tests/v41_idx/` | Served from 2026-11-18 20:19:16Z (windows p6 and p6b) to 2026-11-19 03:55:51Z with nine adopted lines; the 2026-11-18 bench ran on it; superseded by 9S |
| `PP3KIT9M-over-cut` | 2d1188a | Kit 9L with two files changed: `v1/worker/gpu_worker.py` prints the receive-buffer wait's line once per process by a plain logger call, and `v1/engine/core.py`'s statistics line prints the decode round (answer to answer) | Served from window p3r (2026-11-18 11:58Z) to p6b with five to eight adopted lines; superseded by 9N |
| `PP3KIT9L-over-cut` | ced635d | Kit 9 (engine ef44bfb: kit 8 plus the unconditional receive-buffer wait, the repaired token cache `VLLM_V41_PREFIX_TOKEN_CACHE` under `renderers/`, the drafter's faster GEMV `VLLM_V41_DRAFT_GEMV2`) plus `v1/engine/core.py` with three switches, each off by default: `VLLM_V41_PP_EMPTY_AHEAD` (a request's spacing steps are submitted while its step is in flight), `VLLM_V41_PP_ANSWER_NO_SLEEP` (no sleep between an answer and the next submission) and the diagnostic `VLLM_V41_PP_LOOP_STATS` | Window p3 (its first boot passed every measuring line; stopped by a witness that read a line the engine never prints); superseded by 9M |
| `PP3KIT8-over-cut` | 6950d68 | Kit 6c's paths plus three default-off switches: the middle-stage prompt skip `VLLM_V41_PP_MID_PREFILL_SKIP`, the turn pipeline `VLLM_V41_PP_TURN_CHUNKS` and the hold of decode steps after a prompt's end `VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD` | With every switch off it is kit 6c on the cards (window p1). Served from window p2s with the release setting and the middle-stage skip; its three switches are all adopted today (windows p2, p5, p5u) |
| `PP3KIT7-over-cut` | 1d51754 | Kit 6c plus `VLLM_V41_PP_SEND_OVERLAP` (a stage awaits its hand-off before the next forward) | Closed in window o2: no gain |
| `PP3KIT6c-over-cut` | e7b2417 | The first serving kit, 11 files over f20d155: three pipeline stages; the step spacing rule of the synchronous scheduler (a request's next step waits `pp_size` scheduling steps); up to eight slots; the middle stage's memory changes (layer 14's Engram held and applied on the first stage, the loader's stage filter); the card-memory line and the optional card limit (`models/deepseek_v41/card_limit.py`) | Served from window w5 to window p2s (2026-11-18 04:17Z); accepted in windows w5, o2, k1 and s4 |
| `PP3KIT6b`, `5c`, `4b`, `4`, `3`, `2`, `PP3KIT-over-cut`, `PP3KIT-over-baseline` | see each `MANIFEST` | Earlier steps: a version that did not load at three stages (6b), middle-stage memory (5c), slot labels for eight slots (4b), slots (4), the scheduling fix (3), the first kits | History |

Nothing is being built. Every kit of the plan's optimization list has had its window: 9S adopted (c2a); 9T, 9U and 9V
ran and failed their lines (c3b, c7a, b3a); 9Q and 9R were their cuts over 9N, superseded before a window. The split
count, adopted in window b4a, is the multiply library's own switch and not a kit (below). Kit 9P (the first
lean hand-off, window c3a: did not boot) is not published; 9Q carries its correction. The conversation store's kit
(over kit 9M, applied unchanged over 9N) ran its window on this set (st5 to st7, 2026-11-19): every state check passed
and its timing line failed; it stays private and off (below).

**The adopted lines** (`run/state/ADOPTED.env`, written from an allow-list by kit; the serving record carries their
digest; the serving boot refuses a set the record does not know), eleven since window b4a: `VLLM_V41_EMPTY_CACHE_TOKENS=8192` (window p2s),
`VLLM_V41_PP_MID_PREFILL_SKIP=1` (p2), `VLLM_V41_PP_EMPTY_AHEAD=1`, `VLLM_V41_PP_ANSWER_NO_SLEEP=1`,
`VLLM_V41_DRAFT_GEMV2=1` (p3c and p3r), `VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD=128` (p5), `VLLM_V41_PREFIX_TOKEN_CACHE=1`
(p4b), `VLLM_V41_PP_TURN_CHUNKS=3` (p5u; the accepted capture then became `accepted/quality-TC3.json`),
`VLLM_V41_INDEX_TC=1` (p6b), `VLLM_V41_WO_A_BATCHED_4K=1` (c2a; the accepted capture then became
`accepted/quality-TC3-WOA4K.json`), `TM_GEMM_TUNE=max_splits=1` (b4a; the accepted capture then became
`accepted/quality-TC3-WOA4K-S1.json`). The receive-buffer wait is on by the kit's default since 9L and is not a line.

**Files of kit 6c** (paths under `vllm/`): `model_executor/model_loader/gguf_loader.py`,
`models/deepseek_v41/{card_limit.py (new), contract.py, dense_tm.py, model.py, pp_commit.py, slots.py}`,
`v1/core/sched/{scheduler.py, v41_multi_seq.py}`, `v1/worker/gpu/{model_runner.py, v41_multi_seq_state.py}`.

## Installing

    bash kit/PP3KIT9S-over-cut/apply.sh <engine-src>/vllm <venv>/lib/python3.12/site-packages/vllm <backup-dir>

It refuses unless every file it replaces has its base sha256 in both trees, saves the originals to the backup
directory, copies the new files and checks every result. `rollback.sh` restores the backup. Install only while
nothing serves from those trees. The serving profile then needs `VLLM_PP_LAYER_PARTITION=14,14,12`, pipeline
parallel size 3 and tensor parallel size 4 ([docs/OPERATIONS.md](../docs/OPERATIONS.md)).

## The conversation store's kit

The conversation store and pinned-prefix cache are a second kit, maintained in a private working repository
(26 files, five shared with kit 6c, engine `e6df5ed`; rebased on kit 9M, it applies unchanged over kit 9N). It was in
the serving trees with the store off from 2026-11-17 15:47Z to 2026-11-18 04:17Z (window p2s took it out); with the
store off it changed no token and no memory mark (windows st1, st3b, st4). Every server test of the store passed on
this build in window st4, and again in window st7 on the serving set of 2026-11-19 (the dump, a restart on the saved
set, exact answers after the restart, memory equal to the store-off boot, 0 waits over 1,704 and 624 save lines); its
timing line failed there (first token of the larger turns in the four-chat rehearsal 4.718 and 3.722 s with the store
on against 3.596 and 3.590 s off, limit 3.696). The overlay was rolled back; the store is a failed optimization on this
build as its rule stands. It is not published in this release and is not switched on in what serves.

## Switches that are not kits

- `TM_GEMM_TUNE=max_splits=1`, the TurboMind multiply library's own switch: one split of the inner dimension at
  every row count, so a prompt's last chunk no longer rounds by its size. Tried as arm G (windows g1 and t1):
  INCONCLUSIVE at its quality line against the all-native reference. **Adopted in window b4a** (2026-11-19) by the
  closeness rule against the build's own capture: identical 1 of 7, CLEAN, median first divergence 45 to 48 tokens;
  cold reads within 0.8 % of the ten lines', one fresh request +1.3 %. The eleventh adopted line; in what serves.
- `VLLM_V41_ATTN_TIMING=1` and vLLM's torch profiler: the time budget's timers (window t1). Never in a serving boot.
- `VLLM_V41_PP_LOOP_STATS=64`: the engine core's own counters, printed every 64 answers (kit 9L and later). A
  diagnostic of the windows, never served.
- `VLLM_V41_PP_LEAN_HANDOFF=1` (kits 9Q and 9T), `VLLM_V41_EXPERT_PACKED576=1` (kits 9R and 9U) and
  `VLLM_V41_TP_EXACT_SUM=1` (kit 9V): the switches of the three failed windows; the packed switch also needs packs
  re-cut by a private tool and the library under `lib/v41qpn576/`. None is a line.

## Tests

CPU tests of the kit are in `tests/models/`; they run inside an engine tree with the kit applied
(`pytest --noconftest tests/models`). A kit's test must go through the server's real entry point: two kits of
2026-11-17 passed CPU tests that did not, and failed on the cards (windows s1 and w4).
