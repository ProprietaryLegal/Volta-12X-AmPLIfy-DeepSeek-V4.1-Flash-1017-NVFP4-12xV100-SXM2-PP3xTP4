# Kernel and optimization plan: Volta 12X AmPLIfy

DeepSeek-V4.1-Flash-1017 on twelve Tesla V100-SXM2-32GB, three pipeline stages of four tensor-parallel cards, every
routed expert in NVFP4. Written 2026-11-17 on the owner's instruction to plan every custom kernel and optimization.

This plan stands on four documents in this repository. Read them before changing it.

| Document | What it gives this plan |
|---|---|
| The kernel inventory of 2026-11-17 (not in this first release: it cites private working records on most rows) | Every custom kernel, library and switch of the engine as this build runs it (79 rows): on, off or not reached; exactness; recorded gain; gates this build does not meet; generic code in hot paths (gaps G1 to G12) |
| `documents/reviews/UPSTREAM-REVIEW-20261117.md` | What upstream has that applies to three stages (1Cat-vLLM, v100-skinny, vLLM and the others on the watch list) |
| `research/kernels/CUSTOM-KERNEL-OPPORTUNITIES-20261109.md`, `research/eight-card/LANE-RESEARCH.md` (in the Volta research hub, [volta-v100-research](https://github.com/ProprietaryLegal/volta-v100-research)) | The ranked kernel opportunities and the measured time budget of the eight-card build |
| The build plan, section 9 (not in this first release; its results are in `docs/measurements.md`) | Every window's result on this build and the rule each was judged by |
| `documents/reviews/CUT-DEPENDENCE-GEMM-SPLIT-20261117.md` | Why a prompt read's last chunk rounds by its row count (the dense library's split count; item B4) |
| `research/TWELVE-CARD-TIME-BUDGET.md` | The time budget of three stages from window t1: step 0, section 3 |

Commands for every window named here: `docs/OPERATIONS.md`.

Labels: MEASURED (this build, with its window), ESTIMATED (arithmetic or a reading of code), UPSTREAM (someone
else's figure, taken as right until measured here, by the owner's rule of 2026-11-01).

## 1. Where the build stands

State on 2026-11-17T20:46Z (PLAN Amendments 27 to 31). The build is **the owner's default DeepSeek server**
(since about 18:14Z). It serves on `127.0.0.1:18042` in its standing configuration: four
slots, 4,096-row prompt chunks, cache release at 2,048 rows, card limit 31,998 MiB, kit 6c, with the store's kit
in the trees and the store off. Test arms boot on `:18043` and every test stops what serves. The V100s run at
200 W since 18:58Z (before: 150 W); compare only arms of one power setting.

MEASURED, four slots, 150 W (window s4; the standing configuration since then, PLAN Amendment 19):

| Quantity | Figure |
|---|---|
| One request | 128.1 tok/s fresh text, 176.6 repeated |
| Four requests together | 306.6 tok/s in sum fresh (60.5 to 85.5 each), 453.4 repeated |
| Prompt read, one 62,263-token prompt | 3,284 tok/s |
| Four documents of 110,000 to 140,000 tokens read at once | first token after 129 to 166 s each |
| Turn on a long chat, four long chats open | 4.1 s under 2,048 new tokens; 9.3 s above |
| Fullest card with four long chats | 30,398 MiB of 32,510 |
| Load | 134 s |

MEASURED, four slots, 200 W (on the serving arm, 19:27Z, one run; PLAN Amendment 29):

| Quantity | Figure |
|---|---|
| One request | 129.7 tok/s fresh text, 179.2 repeated |
| Four requests together | 312.1 tok/s in sum fresh, 447.5 repeated |
| Prompt read, the same 62,263-token prompt | 3,733 tok/s (+13.7 % against 150 W); first token 16.68 s |

MEASURED, window t1 at 200 W (arm T, PLAN Amendment 31; first figures from the arm's log):

| Quantity | Figure |
|---|---|
| One 4,092-row prompt chunk, median forward span by stage | 793 / 891 / 844 ms (stage 1 p90 1,310 ms) |
| Wall time per chunk of a read (62,260 tokens at 3,712 tok/s) | 1,102 ms; the slowest stage's span is 0.81 of it (about 200 ms per chunk outside any stage's forward) |
| One-request decode round | 25.1 ms (26.5 ms with the profiler) |
| First read of 93,028 tokens | first token after 25.1 s |
| Turn of about 880 new tokens on a 128,617-token chat | 1.24 s to the first token (on 249,276 tokens: 2.22 s) |

Three slots (windows o2 and k1, 150 W): one request 129.5 / 177.3 tok/s; three together 288.0 / 444.6 in sum (the
eight-card build: 198.2 and 299.8); prompt read 3,286 tok/s; fullest card with three long chats 30,054 MiB.

The conversation store and pinned-prefix cache passed every server test on this build in window st4 (PLAN Amendment 25)
and are not yet switched on in what serves (tier A, A2).

Three facts from the inventory govern everything below.

1. **No kernel or switch has a per-switch measurement on this build.** Every recorded gain is from the eight-card
   build of 2026-11-03 to 11-12, most of them on quantized experts this build does not have.
2. **A third of the exported levers never run here.** The expert kernels for quantized (IQ) records, the IQ
   prompt-read kernels and the fused three-tier tail are switched on and not reached: the build has 15,360 native
   experts and no warm tier. On the eight-card build these carried about 3 ms of a decode round.
3. **One request per step.** A step of the pipeline carries one request (`v41_multi_seq.py:718`). Three stages carry
   three steps. A fourth slot keeps a fourth chat warm; four together gave 6 % more than three on fresh text in
   window s4 (306.6 against 288.0 tok/s), not a fourth more.

## 2. Method for every item

These are the rules of the Volta repositories, applied here without exception.

1. **Upstream first.** Name the upstream rows the item touches and start from their code and figure.
2. **Size before building.** State the time the item can remove, from a measured budget, before any code.
3. **A switch, default off.** New paths arrive behind a `VLLM_V41_*` switch that is off by default.
4. **A numerics contract.** Each item says whether it is bit-exact with what it replaces. If it is not, it is
   judged by the closeness rules of PLAN section 9 against the accepted capture, and the owner is told it moves tokens.
5. **Tests through the real entry point,** run against the engine installed on the server before a window. Every
   pattern a check applies to a log is first run on a real log of that kind of boot (PLAN Amendment 17).
6. **A rule before its data.** Each window arm has its lines and predictions written into PLAN before it runs.
7. **End to end decides.** A kernel that wins in isolation is kept only if the whole build is faster in a window.
8. **Workload-shaped lines.** A line that claims a gain for the target workload (long-context agent work: one request
   at a time, prompts of 60,000 to 200,000 tokens that grow by small turns) is measured on cells of that shape, never
   on short cells alone.
9. **Same power, same arithmetic.** A speed line compares with a baseline taken at the same power limit (PLAN
   Amendment 28); a quality line compares with a reference captured in the arm's own arithmetic where the change
   alters rounding (the lesson of arm G's line b, Amendment 31).
10. **A new kit's new log lines meet every pattern of the driver** before the window (PLAN Amendment 21), and the
   serving record `SERVE-CHECKED` is stale after any kit, profile, configuration or card-limit change: the full
   check runs on `:18043` before the build serves on new bytes (Amendment 27).

## 3. Step 0: a time budget for three stages

Nothing below can be sized honestly without it. The eight-card budget (`research/eight-card/LANE-RESEARCH.md`) is of
another layout and another expert mix (it is in the Volta research hub).

- **What:** arm T (`run_window_pp3.sh TAG armT`; PLAN Amendment 30): the standing configuration with the torch
  profiler and `VLLM_V41_ATTN_TIMING=1`, ten profiler cycles shaped on the target workload (a ~94,000-token first read;
  turns of ~880 and ~4,200 new tokens on a ~128,000-token chat; a turn at ~250,000; one beside a decoding chat);
  the budget parsed on a CPU machine by `tools/timebudget_parse.py`.
- **Ran:** window t1, 2026-11-17 20:25:06Z to 20:36:31Z at 200 W (`chain_budget.sh t1 g1 G`): CAPTURED (120 trace
  files, attention-timer lines from all three stages, 0 tracebacks, 0 out-of-memory lines; PLAN Amendment 31).
- **What it gave so far** (from the arm's log): a 4,092-row chunk takes 793 ms on the first stage and 891 ms on the
  middle one, the middle stage slower by 12 % because of the index (53 / 142 / 222 ms by stage, growing with
  context); within a chunk the experts, shared layer, router and their sums take 318 / 334 / 271 ms, attention
  119 / 130 / 112, the output projection with its sum 104 / 113 / 90; about 200 ms per chunk lie outside any
  stage's forward; a one-request decode round is 25.1 ms; a small turn on a 128,617-token chat reaches its first
  token in 1.24 s. The section 1 table holds the figures.
- **The deliverable exists:** `research/TWELVE-CARD-TIME-BUDGET.md` (written 2026-11-17 from the trace parse of window
  t1), the table every later item cites. Its main figures, MEASURED in window t1, one run, profiled:
  - **Decode round, one request, short chat: 26.25 ms** (25.1 without the profiler) = busy 7.06 / 6.59 / 8.82 ms on
    the three stages, which run in series, plus 3.78 ms outside any kernel. By kind over the three stages: experts
    7.70 ms (29 %), dense multiplies 5.39 (21 %), drafter 2.74 (10 %), attention 2.24, hc norms 1.71, router 1.25,
    index 0.77. At about 128,000 tokens 28.07 ms (index 2.78), at about 249,000 tokens 30.98 ms (index 4.47): what
    grows with context is the index.
  - **Prompt chunk:** in three full reads of 133,000 to 138,000 rows the cards idle 164 to 203 ms inside each
    4,092-row step on stages 0 and 1, and the host spends 163 to 187 ms per step in `cudaMalloc` + `cudaFree`.
    HYPOTHESIS, not established: the cache release after a prompt step of 2,048 rows or more
    (`VLLM_V41_EMPTY_CACHE_TOKENS`, default 2,048) costs 16 to 19 % of a first read (item D11).
  - **The front of a turn:** the engine's first token after 1,183 ms at 128,617 tokens = 386 ms before the scheduler
    (template, tokenization, hashing, hand-over) + 797 ms; at 249,276 tokens 2,108 = 848 + 1,260 ms. The front
    costs 3.0 to 3.4 microseconds per token of the whole prompt on every request, hit or not (item D12).
- **Still unmeasured:** the turn of about 4,200 new tokens. `tools/turn_probe.py` replays an answer without its
  reasoning, so its turns left the stored prompt at the last token after 200-token answers and were read cold
  (36 to 38 s; one 88.4 s beside a second chat); by rule T2 those cells do not count. The probe is to be corrected
  (answers within the ring's reach, as `tools/mem_rehearsal.py`'s turns are) and that turn measured again, which
  can ride with kit 8's arm.

## 4. The plan, in order

Each row: what it is, where it comes from, what is known, what it costs. "Window" means one arm with the serving build
down; several arms can share a window.

### Tier A: in motion

| # | Item | State |
|---|---|---|
| A1 | Four slots | DONE: window s4 passed every line; four slots stand (PLAN Amendment 19) and serve as the owner's default since 2026-11-17 18:16:40Z (Amendment 27) |
| A2 | Conversation store and pinned-prefix cache on this build | PASSED EVERY SERVER TEST in window st4 at four slots (PLAN Amendment 25): the state checks on twelve workers, M12 and M1 (at the lengths of Amendment 23), S2 after a restart at 130,000 tokens, the pinned-prefix cache's check after the restart, 0 waits over every save; as it would serve, 7 of 7 read cold and no card memory cost. The restore pair restored a turn after a restart in 1.23 s and FAILED its rule as written: a restored or cached turn is not token-identical to a cold read of the whole chat. NOT SERVING. Left (Amendment 25): the store's kit version that makes its folders private itself (built by the store's former owner, kits not rebuilt); the first-token comparison with four grown chats, store on against off, in one boot order (the larger turns stood 2 to 3 s above store-off boots once); the owner told that a restored chat answers in the model's other rounding. Then an amendment that switches it on, and a full check (the record goes stale) |

### Tier B: the three largest levers

| # | Item | Source | What is known | Exact? | Cost |
|---|---|---|---|---|---|
| B1 | **Batched decode on the pipeline**: several requests in one step, micro-batches rotating through the stages, a request's next step queued while one is in flight | UPSTREAM 1Cat-vLLM pull request 1482 (merged 2026-11-11, 16 V100s in four stages): `VLLM_V41_BATCH`, `PP_MICRO`, `PP_AHEAD` | UPSTREAM at 1K context: one request 19.3 ms per round, four requests about 628 tok/s in sum, 64 requests 1,809. This build: 288 tok/s at three, and no more at four. Our own earlier group steps on eight cards LOST (0.80 to 0.92 times, 2026-11-06) with kernels sized for 6 rows; three built kernels that act only in group steps exist and were never judged (x-resident dense for 9 to 24 rows, stacked attention chain, cross-expert down). Upstream's path has no prefix cache; ours keeps prompts, and the design must keep them | No. Upstream: "Concurrent outputs are not bitwise equal to single-request outputs". Needs the owner's word before it serves | A study first (one week of reading and a design note against our slot scheduler), then a redesign in pieces: scheduler, graphs per row count, kernels above 6 rows. Several windows |
| B2 | **A prompt pipeline across the stages**: one chunk per stage in flight; replaced by kit 8 | UPSTREAM 1482 (`VLLM_V41_PREFILL_PIPELINE`), 1416 (merged 2026-11-09: "a prompt prefilled in n chunks over S stages takes about n + S - 1 chunk times") | READ and a CPU model of the real loop and scheduler (2026-11-17): the engine ALREADY keeps one prompt chunk step per stage in flight (three here; a chunk counts as computed when it is scheduled, the batch queue holds `pp_size` steps); `MULTI_SEQ_PREFILL_BURST` limits prompt steps only beside a decoding request owed steps; upstream's switch lifts a limit of two that ours does not have. A lone read gains 0 % from scheduling (ESTIMATE). Also READ and MEASURED: only the LAST stage skips an early chunk (`model.py:1039-1043`; window s4's skip lines come from `Worker_PP2` only), so such a chunk costs stages 0 and 1 fourteen layers each. What replaces the item, as kit 8 (branch `pp3-12card-pipe`, three switches off by default; CPU only when this row was written; see window p1 in `docs/measurements.md`): (1) the turn pipeline `VLLM_V41_PP_TURN_CHUNKS` (the last rows of a read in several chunks so the three stages overlap on a turn; being written); (2) the middle-stage skip `VLLM_V41_PP_MID_PREFILL_SKIP` (built, 27 CPU tests; gain 0 if stage 0 is the slower of stages 0 and 1, else (t1 - t0) / t1: window t1 measured stage 1 the slower, 891 against 793 ms); (3) the last-prompt-step hold `VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD=N` (built, 16 CPU tests; ESTIMATE: a small turn beside a 250,000-token read 14.1 s today, 4.9 s at N = 128) | (1) changes the cut, so another rounding, judged by closeness; (2) exact under the existing skip's contract (line b: 7 of 7); (3) scheduling only, exact | Kit 8 with its arm P and `chain_pipe.sh`. State: `kit/PP3KIT8-over-cut` (engine `6950d68`) is in this repository; with its switches off it is kit 6c on the cards (window p1); the switches are not in what serves. One window |
| B3 | **Large four-card sums in full precision** | This build's probe of 2026-11-17 (PLAN, "Step 0 ... RUN"); upstream has no fix for this model, but two precedents on another model: a tiled long-prompt reduce through the custom path, and a fused reduce with the norm | MEASURED: a sum too large for the custom path (over 8 MiB; every prompt chunk) is added by NCCL in half precision in an order that depends on buffer size and offset; a 64-row shift re-rounds 7 to 9 % of the elements; against the exact sum a third of all elements differ by a last bit. 80 such sums per prompt forward, not timed (gap G1). No window has shown a quality loss from it | It replaces inexact arithmetic by exact: answers on long prompts move once, then no longer depend on the cut. A new accepted capture is needed | One engine change for this build and the eight-card build (the eight-card build lists it too). One window: the owner's test "the same prompt read whole and in two chunks must agree" is its acceptance |
| B4 | **A prompt read whose arithmetic does not depend on the last chunk's row count** (found 2026-11-17 in the store window st3b) | This build's one-card probe `tools/gemm_split_probe.py` (PLAN Amendment 23); `documents/reviews/CUT-DEPENDENCE-GEMM-SPLIT-20261117.md`. The dense library's switch `TM_GEMM_TUNE=max_splits=1` exists in the engine; no upstream change found | MEASURED: the general dense multiply picks its split count from the row count (10 at 2,292 rows, 1 at 2,308, 5 at 4,092 and 8,188), and rows computed under different counts round differently (0.5 % of layer 0's window projection values); with `max_splits=1` 120 of 120 row-count pairs are bit-identical. In the probe the call was faster at most row counts (0.40 against 0.57 ms at 2,292; 1.51 against 1.87 ms at 8,188) and slower at two (0.78 against 0.62 ms at 3,696). It reaches every such multiply, decode included: not timed in a round | Same model in one rounding instead of several: answers on prompts whose last chunk falls on another split count move once. A new accepted capture is needed. It does not remove B3's dependence on the chunk size | First an arm with the switch (no build): answers against the references, decode round time, the long-prompt capture. If decode pays, a per-shape split fixed for prompt reads only (a change in the library's dispatch and a rebuild). One window. RAN TWICE, NO VERDICT (arm G, PLAN Amendments 24, 29, 31): g1 (19:20Z) stopped at line a by a witness that asked for evidence that cannot exist (a titled worker's `/proc` environment is blank; the switch WAS delivered: 16 library lines "max_splits 1"); t1 (20:36Z to 20:40Z) passed the corrected line a and ended INCONCLUSIVE at line b: against the all-native reference N2 CLEAN and NEARER than the eight-card build (mean \|dlogprob\| 0.0079 within 0.018), but the median first divergence 96 tokens where 104 are asked (the standing build: 208). Reading, not established: N2 was captured under the default dispatch, whose split counts the standing build shares, so the two stay token-identical longer for a reason that is not closeness. A fair line needs a reference captured in the arm's own arithmetic; the switch is taken up again with kit 8's arm. Lines c to g have not run. Note: `tools/speed_floor.py` compares line e with window s4 (150 W), Amendment 28 with the 200 W baseline: reconcile before the rerun |
| B5 | **A snapshot hit that starts from the pages its own read wrote** | READ in the engine (`block_pool.py:62-91`, `kv_cache_manager.py:257-282`; report section 6): the pool keeps duplicates of a token hash and hands out the oldest surviving one | MEASURED in st3b's dump: a request began with layer 0's window pages written by another request's read of the same tokens and its own slot state. Harmless when the two reads use one arithmetic (B3 and B4 make them so); until then two requests with a shared prefix can change each other's later answers in the last bit | Exact once B3 and B4 hold; otherwise it removes one way two roundings mix | A Python change in the cache manager for both builds; CPU tests; one arm |

### Tier C: kernel gaps at this build's shapes

Ordered by what step 0 is expected to show; each is confirmed or dropped by it.

| # | Item | Gap | What is known | Exact? | Cost |
|---|---|---|---|---|---|
| C1 | Whole 64-row passes in every chunk | G5 | The fused prompt-read attention takes whole 64-row passes only. A chunk here is 4,092 rows: 63 passes and a 60-row remainder that runs generic gather and `torch.bmm`, in every chunk and every layer, twice as often per token as on eight cards. A chunk of exactly 4,096 rows (a batch limit of 4,100) has no remainder. ESTIMATED from the code; not timed | The cut moves, so tokens move once | A profile and one window arm. No kernel work |
| C2 | The batched `wo_a` projection at this chunk size | G3 | The path admits exactly 8,188 rows (`dense_tm.py:502`); here it never runs. The read is 4 % slower than at 8,188-row chunks on one read each (MEASURED o2); cause not shown | Other reduction order: judged by closeness | Admit the build's chunk size in `prefill_dense_blas.py`, tests, one arm |
| C3 | The decode hand-off | G2 | Every round sends hidden states with metadata across two boundaries; the metadata-free path needs two stages, one slot and no drafter. Indicative on two stages: 1.91 ms of a 26.0 ms round. UPSTREAM 1482: only rank 0 sends; sampled tokens on their own communicator | Yes | Several files. Avoid upstream's `PP_RECV_EARLY` (shipped off, illegal memory access, cause open) |
| C4 | Native expert layer: five launches and a cast | G7 | 266.3 us per layer on one card; ceiling of one launch 1.0 to 1.2 ms of a round, with an earlier recommendation not to build it. UPSTREAM 1482 has fused gate/up + clip + SwiGLU kernels for the same record format (117 to 124 us per layer) | Bit-equal required | First a per-layer timing of ours beside upstream's kernels on one card. Build only if step 0 and that timing agree |
| C5 | Decode attention at one row and on window-only layers | G6 | The attention chain admits 2 to 8 rows of width 640; the other cases run `torch.bmm` twice | Byte-identical required | One kernel variant |
| C6 | Router logits and mixes above six rows in prompt read | G4 | `F.linear` in full precision through cuBLAS | Same arithmetic per row required | One Triton program |
| C7 | Zero padding out of each native record | none | 576 of 640 columns are data: 3.57 % fewer expert bytes. ESTIMATED 0.3 to 0.4 ms per round and room for 170 to 200 more experts per card (`research/kernels/CUSTOM-KERNEL-OPPORTUNITIES-20261109.md`) | Bit-equal | New packs (a re-cut of 279 GiB) and a kernel change. Large; only if memory or step 0 asks for it |
| C8 | The drafter's faster GEMV | none | `VLLM_V41_DRAFT_GEMV2` is in the engine, off in both builds, bitwise by its module, with no figure anywhere | Bitwise | One window arm. No code |

### Tier D: latency, start-up and housekeeping

| # | Item | What is known | Cost |
|---|---|---|---|
| D1 | First token in long chats | Reading explains 0.2 s of a small turn and 1.7 s of a large one; the rest (restore of the kept prompt, selection over the compressed rows, cache release twice as often per token as at 8,188 rows, waiting behind other chats' chunks) is not traced. Window t1 (200 W): a turn of about 880 new tokens on a 128,617-token chat 1.24 s to its first token, on 249,276 tokens 2.22 s; the 4,200-token turn unmeasured (section 3). | Step 0's parse; then kit 8 (B2) or its own item |
| D2 | Warm-up for the kernels that compile on the first requests | Eight Triton kernels compile after every boot (`_hash_request`, `_ring_insert`, `_begin`, `_unpack_index_keys`, `_index_topk`, `_gather_attention_packed`, `_post_update_kernel`, `_scatter_draft_tokens_kernel`). The rehearsal before serving absorbs them today. UPSTREAM 1502: the memory a cold compile keeps | Extend the warm-up; checked in any boot |
| D3 | Load time | 134 s, of which about 86 s is model load. Built in the eight-card build's work and not installed: batched prepack and pinned upload, 51.9 % faster on one card slice, exact; no full boot measured | One kit, one boot |
| D4 | A sync pipeline with drafts and a prompt longer than one chunk | UPSTREAM open fix 1537 (the engine dies). This engine has the same guard text and long reads pass in every window; the upstream test has not been run here | Run its unit test on our engine; port one file if it fails |
| D5 | Switches exported and never reached | The IQ expert levers, the keep-batch switch and the batched `wo_a` switch are exported with no effect here. Harmless; but every exported switch is part of the conversation store's fingerprint | Leave as they are until the store's windows have passed; then decide once, with the store's owner |
| D6 | Lookup drafts; a worker wait that does not spin | The eight-card build's work, built: lookup drafts showed no measurable change on the server and take card memory before the expert map is sized; the copy-event wait lost on a quiet host and won 3 to 5 % under load | The eight-card build decides; nothing here until then |
| D7 | Tools | UPSTREAM 1482's memory snapshot and step trace; a three-stage against two-stage equivalence test in the manner of vLLM 66291; a read of our hand-off buffer's dtype against vLLM 65755 (a full-precision residual truncated at a stage boundary) | Reading and CPU tests |
| D8 | The cold turn of a client that does not replay reasoning (both builds) | MEASURED twice: a turn whose prompt leaves its chat's stored prompt only at the last token (the generation prompt ends with a token a client replaying only the answer text does not send back) is read cold after any answer longer than the ring's reach: 43.84 s instead of 1.23 s in st4's restore pair, 36 to 38 s in window t1's probe, on chats of about 130,000 tokens (PLAN Amendments 25 and 31). A client that replays the reasoning with each answer does not meet it | Keep the snapshot's last block servable, or let the store look up at the grain below the match. CPU tests, one arm |
| D9 | A first stream event or a ping before a long read | A client that waits a fixed time for the first body byte can leave during the read of a very long prompt and send it again (the server sends no stream event before the whole prompt is read): the prompt is then read twice. A client-side keepalive helps; an engine-side message start or ping before the read is this build's to rank. Any first-token wait over about 300 s costs the read twice | A change in the API server's streaming path; one arm with a read over 300 s |
| D10 | The cached-token count in a reply's usage | A client cannot see from the server today how many prompt tokens a reply took from the cache (section 6, item 6) | A change in the API server's usage fields; CPU tests |
| D11 | **The cache release inside a first read** (found 2026-11-17 in window t1's budget; nothing run) | The cards idle 164 to 203 ms inside each 4,092-row prompt step on stages 0 and 1 while the host spends 163 to 187 ms per step in `cudaMalloc` + `cudaFree` (MEASURED, t1, `research/TWELVE-CARD-TIME-BUDGET.md`). HYPOTHESIS: the release after every prompt step of 2,048 rows or more (`VLLM_V41_EMPTY_CACHE_TOKENS`, default 2,048; window w5's arm F2e tested 512 and was rejected) is 16 to 19 % of a first read. Ranked high in the first-read group (section 6, item 2) | One arm, no code if the threshold is a setting: the release above the chunk size, judged on the read rate and on the rehearsal's highest card sample against the standing build. Its rule is not yet written |
| D12 | **The front end's per-token cost on a continued chat** | 386 ms of a 1,183 ms first token at 128,617 tokens and 848 of 2,108 ms at 249,276 lie before the scheduler (chat template, tokenization, hashing, hand-over to the engine): 3.0 to 3.4 microseconds per prompt token on every request, hit or not (MEASURED, t1, `research/TWELVE-CARD-TIME-BUDGET.md`). In the first-token group (section 6, item 2) | Being split into its parts on CPU; nothing built |

### Closed: measured and dropped

Kit 7 and its switch `VLLM_V41_PP_SEND_OVERLAP` (the hand-off awaited before the next forward: no gain in window
o2); cache release at 512 rows (same peak, slower first token); prompt chunks of 2,048 rows (not needed: 1,946 MiB
of room); a fifth slot (no speed in it); group steps with six-row kernels; the fused transition kernels; one program
per gate column; the tensor-core prompt-read kernel (21.8 % slower); lossy expert pruning; the confidence cut
switched on; one tensor-parallel group of eight with no pipeline (TP8 PP1: closed by the owner on 2026-11-17 after its window d9 was not faster); #1482's prompt-pipeline switch (nothing to port: B2). Sources and figures: the kernel inventory, section 3 (not in this first release).

Reopened: the late-layer skip on the middle stage. It was closed because the first stage was thought to bound the
read; window t1 timed the middle stage the slower (891 against 793 ms per 4,092-row chunk), and the inventory's row
saying the middle stage already stops after layer 20 is wrong (corrected 2026-11-17 evening). It is
switch C of kit 8 (B2).

## 5. What goes back upstream

- **The step spacing rule for a synchronous pipeline of three stages** (a request's next step waits `pp_size`
  scheduling steps; without it the first decode round after every prompt is wrong at three stages). Upstream has
  it only in the asynchronous scheduler of the other runner. It belongs with PLI Labs' pull request 1031.
- **The half-precision large sums** (B3), once fixed here: a finding and a fix for every V100 tensor-parallel build.
- **The tier digest of the two-tier expert loader path** is the store's own change; nothing of this build.

## 6. Order of work

The order follows what the target workload waits for: long-context agent work with one request at a time and
prompts of 60,000 to 200,000 tokens that grow by small turns (the owner's decision of 2026-11-17), not the size of
a lever on short prompts.

> **Order revised 2026-11-17T22:24Z (after an independent review).** For that workload one request's decode is the
> largest item, then the first read of a long prompt, then the server's front end. Now: (1) window 1: `off`,
> `gemv2`, `mid`, `rel`, `adopt`; (2) decode for one request: the drafter's existing kernel, tokens per round, the
> expert and dense paths against upstream #1482, the index at long context; (3) the repaired token cache, isolated;
> (4) the store, after its paired on/off comparison; (5) the turn pipeline and the hold; (6) the pinned-prefix cache,
> the release as a buffer-lifetime change, batched decode last. The token cache (D12) was HELD after it returned wrong
> ids after an edit at its cut, then repaired and re-proven on CPU. The cache release (D11) is predicted at up to
> about 9 % of a first read, not 16 to 19 %; window p1 measured +9.9 % at 93,000 tokens (`docs/measurements.md`).

1. **Decode for one request.** C3 to C8 as the time budget ranks them (the hand-off between stages, the expert
   layer's launches, attention at one row, the drafter); C4 only after the per-layer comparison with upstream's
   kernels.
2. **The first read and the first token of a turn.** D11 (the cache release inside each prompt step: `rel` passed in
   window p1), B2 with D1, C1 and C2; D12 (host time in front of the scheduler).
3. **No long context read twice.** A2: the store switched on in what serves (every server test passed in window
   st4), a store lookup that serves a turn leaving its chat's prompt at the last token (D8), and B5 (a snapshot hit
   on its own read's pages).
4. **One rounding.** B4 (the split count; arm G is built) and B3 (exact large sums), taken so that answers move once
   and one new accepted capture covers both.
5. **Several requests together.** The B1 study and its design note last: the workload runs one request at a time
   most of the time, and batched decode costs exactness and the prefix cache, which item 3 depends on.
6. D2, D3, D4, D7, D9, D10 in any window that has room.

Every item is entered in the plan with its rule before its window, and its result is added to `CHANGELOG.md` and
the table in section 1 of this file.
