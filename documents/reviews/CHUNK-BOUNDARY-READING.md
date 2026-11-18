# Why a differently cut prompt read rounds differently: a reading of the engine

Written 2026-11-17 on a CPU machine from the build's engine work copy (branch `pp3-12card-cap`, commit 71886d5),
read only. No card was used. Paths below are under `vllm/` unless they start with `csrc/` or `tests/`.

Labels: **READ** = from the code, so a hypothesis. **RUN** = I executed it on the CPU. **LOG** = a line of the
serving boot log `P3s1-X6.log` (not published: a server log) (boot of 2026-11-17 04:58). **EXT** = what I know of a library whose
source is not in this tree (NCCL, cuBLAS); the weakest label.

## Note of 2026-11-17 evening: what later measurements confirm, correct and extend

Added after windows st3b and st4 and the two probes (PLAN section 9: "Step 0 ... RUN", Amendments 22, 23, 25). The
body below is left as delivered.

- **Confirmed: the four-rank sum.** The step 0 probe (2026-11-17T11:19Z, `tools/allreduce_probe.py`) found the
  engine's all-reduce entry sends a prompt chunk's large sums through NCCL in FP16, in an order that depends on the
  buffer's size and the row's offset (7.15 % of elements differ for a 64-row shorter part, 8.62 % at another offset;
  32.89 % differ from the FP32 sum), and small sums through the custom path exactly. That is "row 2 (NCCL)" above,
  now MEASURED at the entry; whether every hidden-size sum of a chunk takes that entry is still not shown.
- **Confirmed and extended: the dense dispatch by row count** (row 3 above, "another split-K ... kernel", and the
  open item "Which TurboMind tile and split-K count the default dispatch picks"). It is a second, separate source of
  cut-dependent rounding. MEASURED on one card (Amendment 23; the split-count report (not in this first release)):
  layer 0's `wkv` projection takes 10 splits at 2,292 rows and 1 at 2,308, 5 at both 4,092 and 8,188; rows under
  different counts round differently, rows under one count are bit-identical, and `TM_GEMM_TUNE=max_splits=1` makes
  every compared pair identical. It explains window st3b's differing layer-0 window pages (tail chunks of 2,292
  against 2,308 rows). It does NOT explain the long-prompt difference between 4,092- and 8,188-row chunks (both
  full chunks fall on 5 splits): the four-rank sum remains that.
- **Consistent: nothing lossy crosses a chunk boundary.** The store's state checks restored a saved chat's state
  byte for byte on all twelve workers, at the store's own lengths, in windows st3b and st4 (Amendments 22, 25).
- **Answered from the "Could not determine" list:** `VLLM_V41_PREFILL_WOA_BATCHED` is 1 on the server (PLAN section 9,
  "The chunk-boundary reading", checked in the server's `run_serve.sh`).

## Short answer

The reading **supports the first half** of the hypothesis and **leaves the second half open**.

1. *Nothing lossy crosses a chunk boundary* (READ, one part RUN). Everything a later chunk reads from an earlier
   one is either integer state, FP16 rows stored as the whole read would hold them, packed rows that are a function
   of one row's FP16 values only, or the ratio-2 compressor's unrounded FP32 pair half. I found no group, window or
   ring that is handled approximately at a boundary. The one deliberate inexactness, the prefill skip, is confined
   to rows nothing reads later, wherever the chunks end (question 4).
2. *The arithmetic path is chosen by the forward's row count*: true in many places, but **only one explicit branch
   goes a different way for 8,124 against 8,188 rows** (`models/deepseek_v41/dense_tm.py:502`, off by default,
   state on the server not determined), and the campaign's own measurement rules it out as the whole cause: the
   5,694-token prompt differs by 0.040 between one chunk and 4,092 + 1,602 (the build's working log), and neither arm
   has an 8,188-row forward. The other pointers I was given (last partial attention pass, `norm.py`,
   `fp8_linear.py`, `prefill_skip.py`) do not separate 8,124 from 8,188: question 2.
3. What does change with the cut is inside **library calls whose result for a row can depend on the forward's
   size or on the row's offset in it**. The strongest candidate is the sum over the four TP ranks: LOG shows
   `Using ['CUSTOM', 'PYNCCL'] all-reduce backends ... for group 'tp:0'` with `nccl==2.27.5`; the custom one takes
   payloads under 8 MiB (819 rows of 5,120 FP16), so every prompt chunk's 80 hidden-size sums per forward
   (attention output and expert output of 40 layers) go through NCCL in FP16. EXT: NCCL's ring all-reduce adds the
   ranks in an order that rotates with the element's offset in the buffer. RUN (arithmetic on random FP16 values):
   rotating the order of a four-term FP16 sum changes the last bit of 42 % of elements. That would re-round almost
   every row on any change of cut, which fits "64 rows moves the answer as much as halving does". It is not
   proven: one probe on a TP4 group settles it (question 5, step 0).

Strongest evidence for "same model, other rounding": the boundary state list (question 3) plus the measured facts
(same configuration bit-identical; one-chunk prompts identical across configurations).
Strongest evidence against a simple reading of it: no explicit kernel choice explains a 64-row shift.

## Question 1: every branch on row count or chunk position on the prompt-read path

"8124/8188?" = does it go a different way for a full 8,124-row chunk against a full 8,188-row chunk.

| # | Where | Condition | The two sides | Same mathematics? | 8124/8188? |
|---|---|---|---|---|---|
| 1 | `models/deepseek_v41/dense_tm.py:502`; `prefill_dense_blas.py:25,189-196,226-232` | rows == 8188 and `VLLM_V41_PREFILL_WOA_BATCHED=1` (default 0, `envs.py:521`) | 8188: restore FP16 weights, one cuBLAS projection (`k3s4.project`); else the prepared projection of row 3 | same projection, other reduction order | **yes if the knob is on**; not determined for the server |
| 2 | `distributed/device_communicators/cuda_communicator.py:399-441`; `collective_provider.py:59-69`; `custom_all_reduce.py:135,346` | payload bytes < 8 MiB (FP16 [M, 5120]: M <= 819) | CUSTOM: four ranks added in FP32 in rank order, rounded once (`csrc/custom_all_reduce.cuh:450-460,739`); else NCCL, which adds in FP16 inside the library | same sum, other rounding (RUN: 33 % of elements differ in the last bit between the two) | both full chunks NCCL; inside NCCL the order depends on size and offset (EXT); a remainder chunk can change side |
| 3 | `models/deepseek_v41/dense_tm.py:478-498` (<= 8 rows: dmv13); `model_executor/layers/quantization/gguf_turbomind.py:104` (<= 64: `apply_segments`; more: `restore_and_apply`); `csrc/sm70_turbomind/ops/gemm_runtime.cpp:451` (rows > 32: default dispatch); `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/context.cu:158-238` | row count M | default dispatch picks the CTA tile and the number of split-K parts from a cost model in (M, N, K, SM count) | same GEMM; another split-K partitions each row's K sum differently | not determined (the cost model must be run); likely for a remainder chunk |
| 4 | `model_executor/kernels/mhc/deepseek_v41.py:197,304,307` | rows <= 6: one program per row; more: `F.linear(x.float(), weight.float())` (cuBLAS SGEMM, M = rows), norm in 2,048-row windows | router logits of every layer, HC mixes at Engram layers and stage starts | same; reduction order is cuBLAS's (EXT: may depend on M) | library internal |
| 5 | `models/deepseek_v41/attention_ops.py:1282-1284,1178,1194`; `prefill_fused_attn.py:46-49` | 64-row passes counted from the chunk's first row; last pass = rows mod 64; `torch.bmm` | the code's own measurement: cuBLAS takes another kernel for last-pass sizes 1 (and <= 17 for PV at width 128) | same | **no**: 8,188, 8,124 and 4,092 are all 60 mod 64, so both grids fall on the same positions and the last pass has the same size; only the pass straddling the cut is 60 against 64 rows |
| 6 | `attention_ops.py:963,1032-1034,243-244`; `index_fuse.py:101-109,211`; `index_topk.py:35-40` | rows x heads <= 64: decode kernels; key rows = max(chunk end // ratio, 512); query tile from the key count | prompt read: `torch.mm` tiles, or the fused v41pidx scorer (LOG: `VLLM_V41_INDEX_FUSE=ilp4` active); top-k is exact and breaks ties by index | same scores; the fused scorer is one custom kernel per row | key count differs, no arithmetic change by reading (with `torch.mm` it would be cuBLAS by shape) |
| 7 | `models/deepseek_v41/model.py:119-127`; `norm.py:15` | rows <= 6 | one-launch FP32 RMS per row; else the FP32 torch expression | same, possibly other order | no (a remainder of <= 6 rows only) |
| 8 | `model_executor/kernels/mhc/deepseek_v41_fused.py:398-458,475` | rows <= 8 | decode kernels; else per-token kernels in windows | same | no |
| 9 | `model_executor/layers/quantization/gguf_moe.py:466-469`; `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu:699,198-202` | rows <= max_tokens: decode forward; `prefill = (S + E - 1) / E > 8` picks row blocks | RB 1 against RB 2 or 4 (`nvfp4_8r_qpn_rb.py`) | the source states "the result depends on neither RB nor NT" | no arithmetic change claimed |
| 10 | `model.py:691,735,1460`; `attn_fuse.py:205,261`; `prefill_fused_attn.py:63`; `attention_ops.py:1169,1289` | rows <= 8 (or <= 6, >= 17) | decode buffers and fused decode kernels against the prompt-read path | same | no (a tiny remainder chunk only) |
| 11 | `models/deepseek_v41/expert_topp.py:43,211` | rows <= 6 and `VLLM_V41_PRUNE_TOPP` > 0 (default 0) | drops low-weight routes | **not the same operation** | no; decode only, off by default |
| 12 | `models/deepseek_v41/prefill_skip.py:87-108`; `model.py:1491-1513,1580-1593` | chunk ends >= 3,072 rows before the prompt end (LOG: active, last stage skips layers 28-39) | skipped chunk: those layers not run, their rings zeroed | not the same for the first rows after the skip; exact for every row read later (question 4) | which chunks skip differs; the 5,694-token case had none |
| 13 | `model_executor/kernels/deepseek_v41_compression.py:53,73` | first row of a forward; last row | first row takes its pair half from the committed state; last row saves its own | same values, same formula | cuts are multiples of 4, so no pair is cut |
| 14 | `model.py:1121-1131`; `cuda_communicator.py:426-431`; `engram_gather.py:45-51` | rows >= 2,048; large tensors | allocator release, in-place reduction, row blocks | memory only | no arithmetic |
| 15 | `models/deepseek_v41/fp8_linear.py:173-222` | rows <= 8, 9..24, <= 6 | drafter linears (`dspark.py:37`) | - | not on the target model's path |

READ for row 3: no V4.1 TP4 shape appears in the cuBLAS table `_AFFINE_BLAS_MIN_M`
(`model_executor/kernels/linear/mixed_precision/sm70_gguf.py:34-68`), so above 64 rows every dense projection should
run the one TurboMind GEMM; I did not see the server's admission report.

## Question 2: 8,124 rows against 8,188 rows

- Different explicit branch: only row 1, and only if its knob is on.
- Same explicit branch on both sides: rows 2 to 10 for the full chunks. So a 64-row shift is **not** explained by
  a kernel the Python code picks. In particular the attention passes (row 5) cannot explain it: the grids coincide.
- What still differs: (a) every row after the first cut sits 64 x k rows earlier in its forward, and every forward
  has another size; (b) the last chunk has 64 x k more rows; (c) rows 8,124 to 8,187 change forward. Any library
  whose rounding depends on the row's offset or the forward's size therefore re-rounds most rows. Row 2 (NCCL) has
  that property by EXT; rows 3 and 4 may have it for the last chunk (another split-K or cuBLAS kernel).
- An inference, not a measurement: a choice-by-size mechanism alone would leave rows 0 to 8,123 of the first chunk
  bit-identical when both sizes pick the same kernel, and would then move the answer less than halving the chunk
  does. Equal distances (0.0307 and 0.0331) point at an offset-dependent mechanism.

## Question 3: what the second chunk reads that the first chunk wrote

| State | Written | Read | Equal to a whole read's? |
|---|---|---|---|
| Window ring, FP16 [256, 512] per layer | `model.py:770,777`; `attention_ops.py:425-443` (last 256 rows) | `attention_ops.py:402-408,1103-1109`: positions before the forward's start | Yes: the same FP16 row the whole read uses as key. RUN: with the engine's `window_ids` and the ring rule, every row's 128 window slots are positions p-127..p for cuts at 1, 2, 3, 64, 127 to 129, 301, 383, 385, 4,092, 8,124, 8,184, 8,188 (0 wrong rows of 36,094) |
| Paged window cache of a stage's first layer | `model.py:662-669` (before attention) | `attention_ops.py:343-353` | Yes, same rows |
| Packed index keys, 68 B per group | `model.py:339-346`; `attention_ops.py:646-684` | all live keys, every chunk: `attention_ops.py:1032-1037` | Yes: packing reads one row's FP16 values only (`kv_pack.py`) |
| Packed compressed rows, 288 B, hot ring and host copy | `kv_store.py:157-168,333-345` | `attention_ops.py:1111-1122` | Yes, same reason; hot and host hold the same bytes. LOG: "staging 0 MiB", every row is in the hot ring |
| Ratio-2 compressor: last odd row's FP32 value and score | `deepseek_v41_compression.py:73-79`, committed `:191` | `:53-54`, first row of the next forward | Yes: unrounded FP32, the formula of a pair inside a chunk. Not exercised: the three chunk sizes are multiples of 4 |
| Engram hash history (3 token ids, position), int64 | `model.py:1148,1632-1635` | next forward's hashes | Yes, integers |
| Drafter's window cache (aux taps of layers 37-39) | every step, zeros on skipped chunks | drafter only, 128 + n rows | Exact for the rows it reads (cone); it proposes tokens, the target decides |
| Three stages: rows of KV source 20 in the last stage's store | `model.py:1456-1461` from the FP16 rows handed over | as the rows above | Yes; within one forward, not across chunks |

"Equal" means: given equal inputs. The inputs are equal only if the row arithmetic before them is, which is the
open half of the hypothesis.

## Question 4: boundary-straddling structures

None handled approximately. Checked: ratio-2 pairs (table above; `restore_prefix` asserts an even position,
`model.py:1699`); window rings across cuts shorter and longer than 128 and 256 rows (RUN above); compressed rows
older than the hot ring (`kv_store.py:392-400` copies the host bytes; unused on the server); index keys (position
addressed, no ring); top-512 (each row's limit is its own position, `index_topk.py:35`, not the chunk's end).

The prefill skip is the one place where a later chunk reads something a whole read would not hold: zeros in the
rings of the skipped layers (`model.py:1585-1588`). Its cone argument (`prefill_skip.py:15-35`) does not depend
on where chunks end: the first executed chunk starts at s <= N - 3,072 for any chunk size (`skip_step`, c + n <=
p - K), and the last row is exact from s + 2,413 (19 layers, one or two stages) or s + 1,524 (12 layers, three
stages). Rows s to s + 1,524 of layers 28-39 are wrong and never read again. READ.

## Question 5: the test "read whole and in two chunks must agree (to 0.009)"

Step 0, before any engine test (minutes, one TP4 group, no model): all-reduce four ranks' random FP16
[8188, 5120] tensors through `tensor_model_parallel_all_reduce`, then their first 8,124 rows alone, and compare
the common rows bit for bit; repeat with [819, 5120] (custom path). If the common rows differ in the first case
and not in the second, row 2 is confirmed. The same probe shape works for rows 3 and 4 (one dense projection,
`F.linear`) with identical input rows in an 8,188-row and an 8,124-row call.

The test itself:
- **Entry point**: the server process as it serves: API, `v1/core/sched/scheduler.py` (the cut is made at
  `:823-828`), `v1/worker/gpu/model_runner.py` with `model_states/deepseek_v41.py`, then
  `DeepseekV41ForCausalLM.forward`. Not `attend_paged_latents` or a model method called by hand.
- **Two arms that differ in the cut only**: same profile, same `max-num-batched-tokens` (so every buffer, graph
  and warm-up is the same), one boot with `long_prefill_token_threshold` = c (the scheduler's own cut,
  `scheduler.py:717-727,823-824`) and one without, prompt N <= 8,188 so the whole arm is one forward. Changing
  `max-num-batched-tokens` instead also changes buffer sizes. A second form needs no second boot and is what a
  user does: send the first c tokens, then the whole prompt in the same session, so the rest is read from a
  session hit (`restore_prefix`); it adds one decode step and limits c.
- **Prompts**: the seven cold long prompts the serve chain already compares (the build's state page), cuts at 4,092, at 64, at an odd
  position (the only way to exercise the ratio-2 pair), and one prompt long enough to skip.
- **Compare, outside in**: (1) top-20 logprobs of the first answer tokens at temperature 0: mean absolute
  difference before the first divergence <= 0.009, and the divergence index; (2) the last prompt row's final
  hidden row; (3) state after the read, bytes: `store.index[s]` and `store.host[s]` rows up to N // ratio
  (`kv_store.py:220-234`), each layer's `window_ring`, the compressor's `partial` and `position`, the hash
  `history`. The first layer and position where bytes differ names the operation: source 2's rows depend on
  layers 0 to 2 only. `pp_boundary_tap` (`model.py:1056-1058,1376-1378`) already hands out the FP32 residual of
  every row at a stage boundary.
- **Expected values from outside the code**: the whole arm is the reference; 0.009 is the owner's number.
- **What exists under `tests/`**: `tests/models/test_deepseek_v41_attention_ops.py` (CUDA; gathers and top-512
  on synthetic tensors; the place for "one call against two calls with the ring insert between" per kernel),
  `tests/models/test_deepseek_v41_pp_forward.py` and `test_deepseek_v41_pp3_handoff.py` (CPU; model.py methods
  taken by source with stub kernels: control flow only), `tests/v1/core/test_balanced_prefill_chunks.py` (CPU;
  drives the real scheduler and lists the chunk sizes: the base for asserting the cut sequence and the skip
  decision of each arm). Nothing under `tests/` drives the real model through the runner.
- **CPU**: the test cannot run on a CPU. The model refuses anything but SM70 TP4 (`model.py:1775-1801`) and its
  kernels are Triton, CUDA libraries and NCCL. On a CPU only the cut sequence, the skip decision and the row
  addressing (as RUN above) can be checked; they show the bookkeeping, not the numbers.

## Question 6: the single change most likely to make a row independent of the cut

Reduce prompt-read tensors over the TP ranks with the arithmetic decode already uses: the custom all-reduce's
FP32 sum in fixed rank order, rounded once, applied to large tensors in whole-row slices under 8 MiB (819 rows)
instead of handing them to NCCL (`cuda_communicator.py:399-441`). Each element's result then depends on its four
inputs only, and prompt read and decode agree in this operation too. This is "most likely" by the reasoning of
question 2, not by measurement; step 0 decides it.

Cost: I cannot estimate it from the code. Known: 80 such sums per full forward, 10 slices each at 8,188 rows,
and a copy of each slice into the registered 8 MiB buffer first (`custom_all_reduce.py:479-496`); NCCL reduces
large tensors in place today. It needs a timing on the cards.

It will not be sufficient alone if rows 3 or 4 also vary: then the dense dispatch must be fixed per (N, K) for
all M above 64 (one tile, one split-K count; cost: a non-optimal split on small remainder chunks), and the router
logits need a per-row kernel (`_gate_cols`, `model_executor/kernels/mhc/deepseek_v41.py:247-271`, exists for 7 to
24 rows in 8-row launches; at 8,188 rows that is 1,024 launches per layer as written). Row 5 matters for 4,092
against 8,188 only.

## Could not determine

- Whether `VLLM_V41_PREFILL_WOA_BATCHED` (row 1) or `VLLM_V41_PREFILL_FUSED_ATTN` is on in the serving profile:
  the profile is on the server. Their activation lines ("V4.1 S4 prefill wo_a native-batched candidate active",
  "V4.1 prefill fused attention active") appear 0 times in both boot logs here, which contain no long prompt
  read, so that is not evidence. One `grep` of a serving log that read a long prompt answers it.
- Which algorithm NCCL 2.27.5 uses for 42 to 84 MB among four V100s, and whether its order depends on offset
  there. That is EXT.
- Which TurboMind tile and split-K count the default dispatch picks for each V4.1 projection at 8,188, 8,124,
  4,092 and at remainder sizes; whether cuBLAS SGEMM (`F.linear`) rounds a row differently by M.
- Whether `VLLM_BALANCED_PREFILL_CHUNKS` (`scheduler.py:348,497-522`) is on; it would change every chunk size.
- Why the distance is as large as 0.03: the top-512 selection as amplifier is the campaign's assumption; nothing
  I read measures it.
