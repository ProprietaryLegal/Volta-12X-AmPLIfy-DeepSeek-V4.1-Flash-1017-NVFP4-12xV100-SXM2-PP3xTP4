# Why the same layer-0 window rows differ between a 2,292-row and a 2,308-row tail chunk on twelve cards

Report of a read-only code review, 2026-11-17, written to this file by the build's owner from the reviewer's
notes. Labels: READ =
code, RUN = executed on a CPU machine, MEASURED = dump/log/receipt, ESTIMATE = the reviewer's arithmetic. Paths are
under `vllm/` unless noted. TM = `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/`; ops =
`csrc/sm70_turbomind/ops/`. Twelve-card tree, read only. NOT YET MEASURED ON
A CARD: the split counts are the reviewer's CPU re-implementation of the library's cost model
(`tools/tm_dispatch_emulation.py`).

## Short answer

Cause (READ + RUN + MEASURED): layer 0's window row is a per-row function of one token and its position. The only
operation on its path whose arithmetic depends on the chunk's row count is the wkv projection (blk.0.attn_kv,
Q8_0, K 5,120, N 512). For M > 32 it runs the TurboMind GEMM with kDefault dispatch (ops
`awq_sm70_gemm.cu:2696-2708`, max_m = 32), whose cost model picks the split-K count from M. The CPU
re-implementation of that cost model gives the following, for every occupancy swept (CTA_M 128: 1-4; smaller
tiles: 1-8):

| rows M | (CTA_M, split-K) | where |
|---|---|---|
| 2,292 | 128, 10 | twelve-card O tail |
| 2,308 | 128, 1 | twelve-card Q tail |
| 6,376 / 6,392 | 128, 2 / 2 | eight-card O / Q tails |
| 3,696 / 3,712 | 128, 2 / 2 | S7 tails |
| 4,092 / 8,188 | 128, 5 / 5 | full chunks of either build |

2,292 and 2,308 straddle 2,304 = 18 x 128 tiles. The other pairs straddle no multiple of 128, 64 or 32. Serial
split-K adds each split's FP32 partial in order (TM `epilogue.h:601-631, 667-680`), so another split count is
another summation tree. A minority of FP16 outputs then round differently, and kv_norm, RoPE and FP8 e4m3 QDQ
(group 128) turn that into sparse page-byte differences (ESTIMATE).

MEASURED (dumps s0r0; s0r1-s0r3 identical): exactly three page arithmetics, as the three split counts predict:
- pack B block 1605: twelve-Q = eight-card (4e24be), twelve-O differs (aab153);
- pack B blocks 1601/1604/1607: twelve-O = eight-card, twelve-Q differs;
- block 1610 (the Q-only rows, same tokens in both packs): one hash per build (5428ef twelve, ebf047 eight),
  different between builds;
- pairs with equal splits are equal: eight-card O = Q (both 2), S7 3,696 = 3,712 (both 2).

Not determined: the real occupancies (the sweep never changed the choice); whether CTA_M alone changes per-element
order (not needed: every differing pair also differs in split); why 10 vs 2 flips fewer blocks than 1 vs either;
the physical block ids behind Ra2.

## 1. Path from the chunk's rows to the page write (stage 0)

- Chunk cut: `v1/core/sched/scheduler.py`; the store hit is admitted as external tokens at :1106-1112. This sets
  M. READ
- Allocator release: `models/deepseek_v41/model.py:1120-1131` and :1183-1192, gated at >= 2,048 rows. Memory only;
  both builds take it. READ
- Embedding: `model.py:1158-1160` -> `model_executor/layers/vocab_parallel_embedding.py:598-628`. The TP4
  all-reduce goes custom (< 8 MiB) or NCCL by size; at 23.5 MB it is NCCL, but it is exact: each row has one
  non-zero rank, the other three add 0. READ
- Expand with pre = (1,0,0,0): `model.py:1164-1166`. None. Layer 0 is not an Engram layer (engram_layer_ids
  [1, 14], config.json). READ
- hc mixes: `model.py:1501-1504` -> `model_executor/kernels/mhc/deepseek_v41.py:197` (Triton up to the row cap)
  and :304-311 (F.linear in 2,048-row windows). This has a row-count branch but is NOT on the page path: layer 0's
  attn_mix is used only after attention. READ
- collapse + attn_norm: `mhc/deepseek_v41.py:393-460`, one program per row. Exact copy of the embedding row, then
  RMS norm. None. READ
- WKV (the cause): `model.py:630-631` (dense_tm.fused only for 1-8 rows, `models/deepseek_v41/dense_tm.py:489-498`)
  -> `model_executor/layers/quantization/gguf_turbomind.py:686-727` (family 3) -> :100-121 (rows > 64:
  restore_and_apply) -> `gguf_dense_hmma.py:186-238`. The BLAS band is refused: LOG
  "local_shape_has_no_prefill_calibration" for Q8_0 in both boot logs. Then -> ops `awq_sm70_gemm.cu:2649-2718` ->
  ops `gemm_runtime.cpp:446-452` (M > 32: kDefault) -> TM `gemm.cu:715-770` (batch_tail only for 32 < M <= 64;
  exact-M cache at `dispatch_cache.cu:291`; no fast target, `gemm.cu:205-312`) -> Find(top_k = 1), the cost model
  at TM `context.cu:117-238`, ranked at `gemm.cu:770-870`. Tiles: U8 affine CTA_M 128/64/32/16/8 x 128 x 32 (TM
  `kernel/sm70_884_8.cu:67-78`). Splits 1..16 (`gemm.cu:538-539`). The 32 MiB partials workspace suffices for
  every split (`gemm.h:16-17`, `kernel_impl.h:262-293`). THE SPLIT COUNT IS A FUNCTION OF M. READ + RUN
- kv_norm: `model.py:118-131`, the torch FP32 expression above the short-row path. No evidence of M dependence.
  READ
- attn_prep: `attn_fuse.py:200-223`, 1-6 rows only, so prefill falls back. READ
- RoPE: `quant_rope.py:159-201`, per (row, head). FP8 QDQ: `quant_rope.py:39-90`, per 128-group. None. READ
- Page write: `model.py:662-669` -> `attention_ops.py:39-58`, per row, through the block table. No arithmetic. READ
- After the page write, so not on this path: 64-row attention passes, the wo_a gate at 8,188 rows
  (`dense_tm.py:501-508`), prefill skip, all TP all-reduces.
- Consequence: the row at position p is f_M(token_p, p). MEASURED support: block 1610 is equal across packs
  within a build.

## 2. What the two builds do differently

- Engine trees: `diff -rq` shows 31 differing files. On this path only `model.py` (stage plan, export, tail
  Engram) and `dense_tm.py` (tail-Engram role) differ, and neither touches layer 0's projection. Every .so
  (`_C.abi3.so` included) is identical. The eight-card tree was restored without the store files after st2b; the
  store patch (the pinned-prefix cache's engine patch, in a private working repository) touches nothing on this path.
  READ
- Switches: the same 56 VLLM_* exports. `swap_arm12.sh` adds only VLLM_PP_LAYER_PARTITION and
  VLLM_V41_PP_DIAGNOSTIC_LAYOUT. READ
- Profiles: same GGUF, same kernel-config (prefill_min_m 1, dequant_workspace_bytes 33554432). Differences are PP,
  max-num-batched-tokens, max-num-seqs, residency, prefill_warm, iq_tail and reserves; none reaches layer 0's
  attention. READ
- Boot logs: "V4.1 TurboMind dense ... skipped: []" (wkv converted) and the Q8_0 BLAS band refused, on both
  builds. MEASURED
- So the only branch taken differently is the wkv split count, through M: twelve-card O 10, Q 1; eight-card O and
  Q 2; full chunks 5 on both. RUN

## 3. Ranked causes

1. wkv split-K chosen from the row count (TurboMind default dispatch).
   - READ: the section 1 chain.
   - RUN: `tm_dispatch_emulation.py` (float32 where the C++ uses float, stable ArgSort). Margins: 2,308 picks
     splits 1 over 2 by 0.5 % of the score; 2,292 picks 10 over (64, 5) by 2.4 %.
   - MEASURED, differing blocks of 1601-1609:
     - s10 vs s2 (twelve O vs eight): A 0/9, B 2/9 (1605, 1608);
     - s1 vs s10 (twelve Q vs twelve O): A 6/9, B 5/9;
     - s1 vs s2 (twelve Q vs eight): A 6/9, B 4/9 (1601, 1604, 1607, 1608);
     - equal-split pairs are equal: eight-card O/Q (both 2), S7 (both 2).
   - ESTIMATE: a 1-ulp FP16 change survives e4m3 QDQ (3 of 10 mantissa bits) in roughly 1 case in 100+, which
     fits the sparsity.
   - Refuted by: bit-identical wkv output at 2,292 and 2,308 rows; or TM_GEMM_TRACE showing the same split for
     both; or two M values with the same split and tile giving different bytes.
2. kv_norm's torch reduction changing configuration with M. READ/EXT, weak: it would not explain why twelve-Q
   matches eight in block 1605 of pack B. Refuted by the probe's kv_norm stage.

Excluded on this path (READ): NCCL (exact here); hc 2,048-row windows / F.linear (not on the page path); 64-row
attention passes, wo_a gate, prefill skip (after the page write); the 2,048 release (memory only); slot and
staging buffers (no arithmetic); pool sharing (explains Ra2, not O vs Q).

## 4. Smallest decisive test

- CPU (done): `tm_dispatch_emulation.py <M...>`. With TMN=1280 it gives the wq_a shape: splits 4 / 5 at 2,292 /
  2,308, and 1 at 4,028, 4,092, 8,124 and 8,188.
- One card, no model: `gemm_split_probe.py` (this folder). It runs blk.0.attn_kv through the engine's preparation
  on real embedding rows, for M in {2292, 2308, 6376, 6392, 3696, 3712, 4092, 8188}, with the split count traced,
  then bit-compares the pairs. It then runs again with TM_GEMM_TUNE=max_splits=1.
- Expected if cause 1 holds: 2,292 vs 2,308 differ; 6,376/6,392, 3,696/3,712 and 4,092/8,188 equal; 2,292 vs
  6,376 differ.

## 5. Candidate fixes

- B. Env TM_GEMM_TUNE=max_splits=1. Policy field `config/sm70_native.py:27`; parsed TM `gemm.cu:547-551` and
  `tuner/params.cc:22`.
  - Cost: no build, but it reaches every TurboMind GEMM including decode, so closeness and decode timing must be
    rerun.
  - Effect: removes the GEMM part of the cut dependence for all TurboMind projections; not the NCCL FP16 sums.
- A. A row-count-invariant prefill spec per (type, N, K), fixed at a reference M, in Gemm::Dispatch (TM
  `gemm.cu:715-770`) or at ops `awq_sm70_gemm.cu:2696`.
  - Cost: C++ change and a _C rebuild; tails use the full-chunk split (wkv: 5); small speed cost (ESTIMATE).
  - Effect: same scope as B; still leaves NCCL, F.linear (router/hc) and the attention pass that straddles a cut.
    Necessary, not sufficient.
- C. A profile key: none found that sets the split count.
- D. Tie the snapshot hit to its own read's blocks: a Python change; fixes Ra2-type mismatches.
- First to try: B with the probe; then A if B's decode cost is unacceptable.

## 6. Ra2 and the block pool

- Every read writes fresh blocks. A store restore is admitted as external tokens with no local hit
  (`v1/core/kv_cache_manager.py:579-582, 600-602`; `scheduler.py:1106-1112`). Those tokens get
  block_pool.get_new_blocks (`v1/core/single_type_kv_cache_manager.py:269-273`; `block_pool.py:386-422`). LOG:
  Qa1, Oa1, Qa2, Oa2 and both reference reads show "decision hit reason store" at 3,008; pool hits of 25,744 /
  25,728 were refused by the session guard. READ + MEASURED
- cache_full_blocks (`block_pool.py:265-340`) inserts duplicates under the same hash. insert (:75-91) keeps them
  in insertion order, and get_one_block (:62-73) returns the OLDEST SURVIVING duplicate. A duplicate leaves only
  when its block is reused (:424-459). READ
- _v41_snapshot_hit (`kv_cache_manager.py:257-282`) takes the slot's rings but gets its blocks by hash
  (find_longest_cache_hit -> get_cached_block, `block_pool.py:238-263`). Nothing ties those blocks to the read
  that made the snapshot. READ
- So "Ra2 started from the Q request's pages" follows from the code: Qa2 was cached before Oa2, and the dump shows
  Ra2's nine start hashes equal to Qa2's (MEASURED). Ra1 got O bytes because an older O-arithmetic block was still
  the oldest duplicate then. Physical block ids are not in the dump: could not determine.
- READ, not checked: layers 14 and 28 also keep paged window caches (`model.py:568-579`), so a snapshot hit gets
  those pages by the same rule, and there the O/Q mismatch would be larger. The st3b dump has no page pieces on
  s1r0.

## Files

- `tools/tm_dispatch_emulation.py` and `tools/gemm_split_probe.py` in this repository.
- Layer-0 tensor types: attn_kv Q8_0 [5120, 512]; attn_q_a Q8_0; attn_q_b / attn_output_a / attn_output_b Q5_K.
