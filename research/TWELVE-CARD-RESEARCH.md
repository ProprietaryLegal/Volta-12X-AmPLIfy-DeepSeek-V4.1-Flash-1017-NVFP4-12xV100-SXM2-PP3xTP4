# This build's own research

What was found out about twelve cards in three stages, and where each finding is written down. The copied research
of the earlier builds is described in `research/README.md`. Labels as in the rest of the repository: MEASURED (a
window or probe on the cards, named), READ (code), ESTIMATE (a model). Times are UTC, all on 2026-11-16 and
2026-11-17.

## Layout and engine

| Question | Finding | Where |
|---|---|---|
| Can three stages of four cards hold every routed expert in NVFP4? | Yes: one expert is 4,874,240 bytes per card; twenty layers do not fit one board; cuts at layers 14 and 28 are the only split that fits and is legal (a stage must start at an index source; the last stage holds the drafter) | `documents/plans/PLAN.md` sections 2 to 5 |
| What must the engine change for three stages? | The stage plan and hand-off with four tensors; the step spacing rule of the synchronous scheduler (without it the first decode round after every prompt is wrong at three stages: window w1); slots; the middle stage's memory | PLAN section 6; `engine/README.md`; `kit/PP3KIT6c-over-cut` |
| Is a long prompt read pipelined across the stages? | Yes, already: the engine keeps one prompt chunk step per stage in flight, so a lone read is at its bound from scheduling. Upstream's prompt-pipeline switch lifts a limit ours does not have; nothing of 1Cat #1482's pipeline is left to port (READ and a CPU model) | a CPU model of the build's scheduler, 2026-11-17 |
| Is a turn on a grown chat pipelined? | No: its new rows are fewer than one chunk and run all forty layers as one chunk through the three stages in series, so a turn reads far slower per token than a long read (window t1 and p1 turn figures in `docs/measurements.md`) | the time budget, section 4; kit 8 (in progress) |
| Which stages skip prompt chunks that lie 3,072 rows or more before the end? | Only the last (`model.py:1039-1043`; window s4's log has skip lines from the last stage's workers only). The middle stage computes layers 21 to 27 on such chunks for nothing. An earlier row of the kernel inventory said otherwise and is wrong | `docs/ARCHITECTURE.md` section 6 |

## Answers and rounding

| Question | Finding | Where |
|---|---|---|
| How close is the all-NVFP4 build to the all-native model? | Short prompts against an all-native reference computed with experts fetched from host memory: median first divergence 208 tokens, mean difference 0.0090 | PLAN section 9 (windows w5, o1, o2) |
| Why do long-prompt answers shift when the prompt is cut into other chunks? | Nothing lossy crosses a chunk boundary. A sum over the four cards too large for the engine's own path is added by NCCL in half precision, in an order that depends on the buffer's size and the row's place in it; small sums are added in full precision and rounded once. A 64-row shift of the cut moves answers as far as halving the chunk | `documents/reviews/CHUNK-BOUNDARY-READING.md`; PLAN section 9, "The chunk-boundary reading" and "Step 0 ... RUN" (probe of 2026-11-17 11:19Z) |
| Why do the same tokens give other layer-0 pages in a tail chunk of another size? | A second source: the dense multiply library picks its split count from a chunk's row count (10 at 2,292 rows, 1 at 2,308, 5 at 4,092 and 8,188), and rows under different counts round differently. MEASURED on one card in 25 seconds at sixteen row counts, all as a CPU model predicted; with `TM_GEMM_TUNE=max_splits=1` all 120 pairs are bit-identical. It is the engine's general code, not something of three stages | PLAN Amendment 23; the split-count report (not in this first release); `tools/gemm_split_probe.py` |
| Can a request start from pages another read wrote? | Yes: the block pool shares pages by token hash and a snapshot hit takes the oldest surviving duplicate (READ; seen in window st3b's dump). The base engine's rule on both builds | PLAN Amendments 22 and 23; kernel plan item B5 |
| Does the split switch serve as well as the standing build? | Yes, by the closeness rule: adopted in window b4a (2026-11-19) as the eleventh line. Arm G had passed its delivery line and ended INCONCLUSIVE at its short-prompt line against the all-native reference (nearer in distance, 0.0079 against 0.0090, but its first divergence at 96 tokens where 104 are asked; the reference shares the standing build's split counts). Against the build's own ten-line capture: identical 1 of 7, CLEAN, median first divergence 45 to 48 tokens; cold reads within 0.8 %, one fresh request +1.3 %. The bench on the eleven lines generated different text in every cell from the ten lines' bench, decode cells -10.1 to +9.0 % (mean -0.9 %), and the slow first legal and prose read of the two earlier benches did not recur (HYPOTHESIS: the library's split search, not established) | PLAN Amendments 24, 29, 31, 88, 89 and 92 (windows g1, t1, b4a; the bench of 2026-11-19 14:09Z) |
| Can the four-card sum be made exact, so a read's rounding no longer depends on the chunk's size? | It can (kit 9V: one-stage all-reduce, fixed rank order, FP32 accumulate), at 2.2 times the library's cost on a large call on the board, and it rounds farther from the build's captures than the closeness rule allows (identical 0 of 7, median first divergence 13 against the 25 the line requires); FAILED in window b3a, closed | PLAN Amendments 88 and 90 (window b3a) |
| Is a cached or restored turn token-identical to a cold read of the chat? | No. A restored turn read its last ten rows in a ten-row chunk; the cold read had read everything in other chunks. Equal in 26 of 137 positions (window st4). A restore compared with turns started at the same position is identical in state and answer | PLAN Amendment 25 |

## Slots, memory and the store

| Question | Finding | Where |
|---|---|---|
| How many slots? | Three accepted in window o2 (a third slot keeps a long chat warm while another is read: 2 s against 47 s). Four in window s4: every line passed, four together 306.6 against 288.0 tok/s for three (150 W). A slot does not batch: one request per step | PLAN section 9, Amendments 18 and 19; `documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md` section 1 |
| Does a card-memory limit cost anything? | No token and no memory mark changed under 31,998 MiB (window k1) | PLAN section 9, window k1 and Amendment 17 |
| Does the conversation store work on this build? | Yes: every server test passed in window st4 at four slots, including a 130,000-token chat restored from disk after a restart with its state equal and its answer identical, and no card memory with the store on (29,802 / 30,396 / 28,894 against 29,836 / 30,398 / 28,896 MiB). Its first store boot (st1) had been refused because the two-tier loader path recorded no tier digest; the fixed kit records it on twelve workers. Not yet switched on in what serves | PLAN Amendments 15 to 25 |
| What does a turn that leaves its chat's prompt at the last token cost? | A cold read of the whole chat after an answer longer than the ring's reach: 43.84 s (st4) and 36.0 to 37.9 s (t1) on about 130,000 tokens, against about a second. A client that replays the reasoning with each answer does not meet it | PLAN Amendments 25 and 31; kernel plan tier D |

## Time and power

| Question | Finding | Where |
|---|---|---|
| Where does the time of a prompt chunk go? | First figures (window t1, 200 W, one run): a 4,092-row chunk 793 / 891 / 844 ms on stages 0 / 1 / 2; experts, router and their sums 318 / 334 / 271 ms; the index 53 / 142 / 222 ms and growing with context; about 200 ms per chunk outside any stage's forward (wall 1,102 ms per chunk) | PLAN Amendments 31 and 32; `research/TWELVE-CARD-TIME-BUDGET.md` |
| Where does the time of a decode round go? | 26.25 ms per one-request round at 200 W in the traces (window t1): experts 29 %, dense multiplies 21 %, the drafter 10 %, 14 % outside any kernel; 28.1 ms at 128,000 tokens and 31.0 ms at 249,000, the growth being the index | `research/TWELVE-CARD-TIME-BUDGET.md` |
| Does releasing the allocator's cache less often speed a first read? | Yes: `VLLM_V41_EMPTY_CACHE_TOKENS=8192` read fresh text 12.6 % faster at 61,000 tokens and 9.9 % at 93,000 (2.8 % at 248,000) with identical answers and no higher card memory (window p1, one boot each); adoptable | `docs/measurements.md`, window p1 |
| What does 200 W give over 150 W? | The prompt read 3,733 against 3,284 tok/s (+13.7 %); decode within 2 % (one run each) | PLAN Amendments 28 and 29 |
| Which custom kernels run on this build, and which are switched on without effect? | 79 kernels and switches traced; about a third never run here (they serve quantized experts); twelve hot paths run generic code | `documents/reviews/KERNEL-INVENTORY-20261117.md` |
| What does upstream have for three or more stages? | 1Cat-vLLM pull request 1482 (16 cards, four stages, batched decode, a prompt pipeline across stages); nothing for the half-precision sums or the split count; our spacing rule has no upstream counterpart | `documents/reviews/UPSTREAM-REVIEW-20261117.md` |

## Not yet measured

- The split of a decode round and of a turn's first token by stage and phase: the trace analysis of window t1,
  `research/TWELVE-CARD-TIME-BUDGET.md`, written by the build's owner when it ends.
- A turn of about 4,200 new tokens on a grown chat, alone: window t1's probe read it cold; the probe is being
  corrected.
- Kit 8's three switches on the cards (the turn pipeline, the middle-stage skip, the hold for small turns).
- Hours of serving at four slots under a rule, and the store switched on in what serves.
- Whether the official Engram table (188.8 GiB, against the 4-bit table of 103.0 GiB the build loads) improves
  answers; it needs a reader for the official rows first (the owner has placed it last in the plan).
