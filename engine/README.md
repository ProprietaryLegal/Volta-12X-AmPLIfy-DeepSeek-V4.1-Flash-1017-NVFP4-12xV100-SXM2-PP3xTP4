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
private working repository were reworded in the scheduler and model-runner files, and the `MANIFEST` new-file
sha256 of those files is recomputed (a header line in each `MANIFEST` says so); the code is unchanged.

| Kit (`kit/`) | Engine commit | What it is | State |
|---|---|---|---|
| `PP3KIT6c-over-cut` | e7b2417 | The build's kit, 11 files over f20d155: three pipeline stages; the step spacing rule of the synchronous scheduler (a request's next step waits `pp_size` scheduling steps); up to eight slots; the middle stage's memory changes (layer 14's Engram held and applied on the first stage, the loader's stage filter); the card-memory line and the optional card limit (`models/deepseek_v41/card_limit.py`) | **Serving**; accepted in windows w5, o2, k1 and s4 |
| `PP3KIT8-over-cut` | 6950d68 | Kit 6c's paths plus three default-off switches: the middle-stage prompt skip `VLLM_V41_PP_MID_PREFILL_SKIP`, the turn pipeline `VLLM_V41_PP_TURN_CHUNKS` and a hold of the decode steps after a prompt's end | With every switch off it is kit 6c on the cards (window p1). No switch is in what serves; the middle-stage skip is not yet measured |
| `PP3KIT7-over-cut` | 1d51754 | Kit 6c plus `VLLM_V41_PP_SEND_OVERLAP` (a stage awaits its hand-off before the next forward) | Closed in window o2: no gain |
| `PP3KIT6b`, `5c`, `4b`, `4`, `3`, `2`, `PP3KIT-over-cut`, `PP3KIT-over-baseline` | see each `MANIFEST` | Earlier steps: a version that did not load at three stages (6b), middle-stage memory (5c), slot labels for eight slots (4b), slots (4), the scheduling fix (3), the first kits | History |

**Files of kit 6c** (paths under `vllm/`): `model_executor/model_loader/gguf_loader.py`,
`models/deepseek_v41/{card_limit.py (new), contract.py, dense_tm.py, model.py, pp_commit.py, slots.py}`,
`v1/core/sched/{scheduler.py, v41_multi_seq.py}`, `v1/worker/gpu/{model_runner.py, v41_multi_seq_state.py}`.

## Installing

    bash kit/PP3KIT6c-over-cut/apply.sh <engine-src>/vllm <venv>/lib/python3.12/site-packages/vllm <backup-dir>

It refuses unless every file it replaces has its base sha256 in both trees, saves the originals to the backup
directory, copies the new files and checks every result. `rollback.sh` restores the backup. Install only while
nothing serves from those trees. The serving profile then needs `VLLM_PP_LAYER_PARTITION=14,14,12`, pipeline
parallel size 3 and tensor parallel size 4 ([docs/OPERATIONS.md](../docs/OPERATIONS.md)).

## The conversation store's kit

The conversation store and pinned-prefix cache are a second kit, maintained in a private working repository
(26 files, five shared with kit 6c, engine `e6df5ed`). It goes over kit 6c and is in our serving trees with the
store off: with the store off it changes no token and no memory mark (windows st1, st3b, st4). Every server test of
the store passed on this build in window st4. It is not published in this release and is not switched on in what
serves.

## Switches that are not kits

- `TM_GEMM_TUNE=max_splits=1`, the TurboMind multiply library's own switch: one split of the inner dimension at
  every row count. Tried as arm G (windows g1 and t1): INCONCLUSIVE at its quality line. Not in what serves.
- `VLLM_V41_EMPTY_CACHE_TOKENS=8192`: the allocator's cache released only at prompt steps of 8,192 rows or more.
  Passed every line in window p1; adoptable.
- `VLLM_V41_ATTN_TIMING=1` and vLLM's torch profiler: the time budget's timers (window t1). Never in a serving boot.

## Tests

CPU tests of the kit are in `tests/models/`; they run inside an engine tree with the kit applied
(`pytest --noconftest tests/models`). A kit's test must go through the server's real entry point: two kits of
2026-11-17 passed CPU tests that did not, and failed on the cards (windows s1 and w4).
