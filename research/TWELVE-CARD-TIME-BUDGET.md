# The twelve-card build's time budget (step 0 of the kernel plan)

Where the time of a read, of a turn and of a decode round goes on Volta 12X AmPLIfy, measured with the engine's own
timers on cells shaped like long-context agent work (one request at a time, a long chat that grows by turns). One capture: window t1, arm T, 2026-11-17 20:25:06Z to 20:36:31Z, V100s at
200 W and 1,477 MHz, the standing configuration (four slots, 4,092-row prompt chunks, card limit 31,998 MiB, kit 6c,
store off) with vLLM's profiler and `VLLM_V41_ATTN_TIMING=1`. Rule: PLAN Amendment 30 (written before the data);
result: Amendment 31. Every figure is MEASURED in that one run unless it says ESTIMATE or READ.

- Capture: on the server under `<build>/logs/window/t1/` (120 trace files, `turns/turns.jsonl`, `windows-T.txt`,
  `runs-T.json`, `read-T0.out`) and the arm's log `run/state/P3t1-T.log`; not published (server logs), the parser's
  output `BUDGET.txt` is summarised here.
- Parser: `tools/timebudget_parse.py all` with `tools/classmap-nvfp4-12card.json` (28 minutes, one thread).
- Public text only. The arm and the method were built and tested on a CPU before the window.

## 1. What is valid and what is not

The parser's verdict line is `VERDICT INCONCLUSIVE: 7 cell(s) inconclusive, 0 check(s) not PASS`. All 29 checks
pass. The seven cells:

| Cell | Why it is inconclusive | Consequence |
|---|---|---|
| The 94,000-token first read (cycle 4) in the traces | The probe marked its own window "short" (a defect of `tools/turn_probe.py`'s window rule) | The chunk figures below come from the attention timers (269 chunks of the boot) and from the three cold reads of cycles 6 to 8; the agent's separate single-cycle read of cycle 4 is quoted as such |
| The small turn beside a second chat (cycle 10) | Short, and read cold | No figure for a turn beside a decoding chat |
| Five prefix-log joins (cycles 5 to 9) | The parser looked for the API's request id; the engine's line carries a suffix (parser defect, corrected in `turnprobe2.patch`) | Hits were established by hand from the arm's log: cycles 5 and 9 are hits; 6, 7, 8 and 10 were read COLD |

By rule T2 only cycles 5 and 9 are turn cells. Cycles 6, 7 and 8 were meant as turns of 4,200, 880 and 4,200 new
tokens and became full reads of 132,951, 133,877 and 138,053 rows (the probe replayed 200-token answers without
their reasoning: PLAN Amendment 31). **The 4,200-token turn is not measured.** As full reads those three cycles are
good data and are used below as such.

Checks that bound the method (all PASS): the profiler costs 5.67 % of a decode round (26.540 against 25.115 ms;
ceiling 6 %); the trace's round agrees with the client's to 0.06 %; the same tokens with the profiler on and off;
0 % of kernel time unclassified; the attention timers cost 0.6 % of read rate (3,712 against 3,733 tok/s).

## 2. A decode round for one request

One request, short chat (cycles 1 to 3, 378 rounds): **26.25 ms per round** with the profiler, 25.1 ms without. The
three stages work one after another; while one works the other eight cards wait.

| Per round, ms | Stage 0 (layers 0-13) | Stage 1 (14-27) | Stage 2 (28-39) | Sum | Share of the round |
|---|---|---|---|---|---|
| Experts | 2.80 | 2.65 | 2.25 | 7.70 | 29 % |
| Dense multiplies (projections) | 1.99 | 1.87 | 1.53 | 5.39 | 21 % |
| Drafter | 0.09 | 0.04 | 2.61 | 2.74 | 10 % |
| Attention | 0.74 | 0.81 | 0.69 | 2.24 | 9 % |
| hc norms | 0.60 | 0.59 | 0.52 | 1.71 | 7 % |
| Router | 0.43 | 0.44 | 0.38 | 1.25 | 5 % |
| Index | 0.18 | 0.28 | 0.31 | 0.77 | 3 % |
| Four-rank sums | 0.27 | 0.20 | 0.17 | 0.64 | 2 % |
| Head and sampling | | | 0.64 | 0.64 | 2 % |
| **Busy** | **7.06** | **6.59** | **8.82** | **22.47** | **86 %** |
| Outside any kernel (launch gaps, host, hand-off transfer) | | | | 3.78 | 14 % |

With context the round grows, and the growth is the index:

| Chat | Round, ms | Index, sum over stages, ms | Busy per stage, ms |
|---|---|---|---|
| Short (c1) | 26.25 | 0.77 | 7.06 / 6.59 / 8.82 |
| About 128,000 tokens (cycles 5 to 8, 275 rounds) | 28.07 | 2.78 | 7.73 / 7.58 / 9.94 |
| About 249,000 tokens (cycle 9, 38 rounds) | 30.98 | 4.47 | 8.38 / 8.21 / 10.63 |

Readings:
- Experts and dense multiplies are half the round. A round carries one row (plus the drafter's rows): this is
  per-layer overhead, 0.2 ms of expert work and 0.14 ms of dense work per layer, not arithmetic volume.
- The drafter's 2.6 ms buys the 2.3 to 3.4 tokens a round yields. Tokens per round, not the round, is the lever
  with the widest range (the benchmark's copying text reaches 179 tok/s on the same round).
- 3.8 ms per round is outside every kernel. That is the ceiling of what fewer launches or a tighter hand-off
  could return (ESTIMATE: at most +17 % on one request's rate).
- The index adds 2.0 ms per round at 128,000 tokens and 3.7 ms at 249,000: at 128,000 to 249,000 tokens of context the index
  costs as much as attention.

## 3. A 4,092-row prompt chunk

Attention timers, all 269 chunks of the boot (medians; p90 in brackets):

| ms per chunk | Stage 0 | Stage 1 | Stage 2 (the one chunk per read it computes; n = 9) |
|---|---|---|---|
| **Forward span** | **793** (882) | **891** (1,310) | **844** (1,057) |
| Experts, shared layer, router and their sums | 318 | 334 | 271 |
| Attention | 119 | 130 | 112 |
| Output projection and its sum | 104 | 113 (200) | 90 |
| Query and key projections | 65 | 65 | 64 |
| Index | 53 (114) | 142 (362) | 222 (437) |
| hc norms | 92 | 89 | 80 |
| Compressor | 15 | 16 | |
| Engram | 13 | | |

The wall time per chunk of an undisturbed read is 1,102 ms (62,260 tokens at 3,712 tok/s). The slowest stage's
span is 0.81 of it.

**Where the other 200 ms are (the capture's main new finding).** In the three cold reads of cycles 6 to 8 (33 or 34
chunks each) the traces give, per chunk on stages 0 and 1:

| Per chunk, ms | Stage 0 | Stage 1 |
|---|---|---|
| Card span | 1,018 to 1,045 | 1,016 to 1,049 |
| Card busy | 843 to 855 | 825 to 865 |
| Card idle inside the step (span less busy) | 164 to 202 | 183 to 203 |
| Host time in `cudaMalloc` + `cudaFree` | 182 to 187 | 163 to 175 |

The idle time of the cards inside a prompt step equals the host's time in device allocation and release. READ: the
engine empties the allocator's cache after a prompt step of 2,048 rows or more (`gpu_model_runner.py`, the release
this build serves with since arm F2e, PLAN section 9), so every 4,092-row chunk returns its work memory to the driver
and the next chunk asks for it again. HYPOTHESIS, not established: that release is the 16 to 19 % of a read that no
kernel accounts for. Arm F2e showed the same thing from the other side (release at 512 rows: the same peak, the
larger turns' first token twice as late). What it buys is headroom under the card limit; what it costs was not
known until this capture. The test is one arm: the release raised above the chunk, read rate and the memory
rehearsal's high-water mark against the standing build (rule to be written in PLAN before it runs).

Other facts of the chunk:
- The four-rank sums are 60 to 67 ms per chunk (6 to 7 %), sliced into 45 to 412 kernels; no waiting on a slow
  rank (arrival skew median 0.002 ms). A faster all-reduce has at most that to give.
- On every chunk the last stage skips it still runs the drafter's FP8 linear for 262 ms (the single-cycle
  read of cycle 4). It is off the critical path (the last stage is busy 272 ms of each 999 ms).
- Stage 1 is the slowest stage and computes layers 21 to 27 for nothing on a chunk the last stage skips (READ,
  the build's scheduler and model code). Kit 8's `VLLM_V41_PP_MID_PREFILL_SKIP` removes that half of its work.

## 4. A turn on a long chat

| | Turn of 937 new rows on 128,617 tokens (cycle 5) | Turn of 1,212 new rows on 249,276 tokens (cycle 9) |
|---|---|---|
| Client's first token | 1.244 s | 2.220 s |
| Engine's first token | 1,183 ms | 2,108 ms |
| Front: API arrival to queued (template, tokenization, hashing, hand-over) | **386 ms** | **848 ms** |
| Queue | 0 | 0 |
| Scheduled to first token | 797 ms | 1,260 ms |
| Card span, stage 0 / 1 / 2 (in series) | 246 / 269 / 318 ms | 360 / 413 / 484 ms |
| Residual nothing names | none (-50 ms) | none (-9 ms) |

Readings:
- The front costs 3.0 to 3.4 microseconds per token of the WHOLE prompt, on every request, hit or not: a third of
  a small turn's wait at 128,000 tokens and 40 % at 249,000. No timer splits it yet. A long agent task pays it on every
  request. It is being split on CPU with the build's own serving
  code; nothing is built.
- After scheduling nothing is hidden: the three stages' steps account for the wait. The stages run in series, so
  a turn waits for the sum of three steps. A step is not linear in rows: from the attention timers of this turn and
  of the 4,092-row steps before it at the same context, a stage step costs 55 to 72 ms fixed plus 0.16 to 0.18 ms
  per row (a CPU fit of the build's scheduler). ESTIMATE on that fit for a turn cut into
  pipelined chunks (kit 8's `VLLM_V41_PP_TURN_CHUNKS`): about 4,200 new rows: 2.37 s of stage time today, 1.53 to
  1.56 s in two or three chunks (-34 %), the client's first token about 2.97 s against 2.13 s (-28 %); about 880
  new rows: 646 ms today, 589 ms in two chunks (-9 %, about -5 % at the client), nothing at four. The fixed cost
  per step is why a small turn gains almost nothing. To be measured, not assumed (PLAN Amendment 32, arm P).
- Prediction check (Amendment 30): "0.7 to 0.9 s to the first token with under 0.1 s outside the three stages'
  steps" was wrong on the second half: the engine's part is 0.80 s, but 0.39 s lie in front of it.

## 5. What the budget says about the order of work

For long-context agent work (one request at a time, turns of a few hundred to a few thousand tokens on chats of
about 128,000, short answers, and first reads of about 100,000 tokens):

| Finding | Size | State |
|---|---|---|
| Device memory released and asked for again on every prompt chunk | 16 to 19 % of a first read (HYPOTHESIS) | One arm; no code if the threshold is a setting |
| The front end's per-token cost on a continued chat | 0.39 s of a 1.18 s turn | Being split on CPU |
| The middle stage's wasted layers on skipped chunks | Up to 12 % of a first read (ESTIMATE: the slowest stage becomes stage 0) | Kit 8, CPU-tested |
| A turn's rows through three stages in series | About a third of a 4,200-row turn's reading time; about nothing under 2,048 rows (ESTIMATE) | Kit 8, being coded |
| The index at long context | 2.0 to 3.7 ms of a 28 to 31 ms decode round; 142 to 222 ms of a chunk on stages 1 and 2 | No item yet |
| Launch gaps and hand-off in a decode round | 3.8 ms of 26.25 ms | No item yet |
| Expert and dense per-layer overhead in a decode round | 13.1 ms of 26.25 ms | No item yet (the kernel plan's tier C) |

## 6. Not measured

- A turn of 2,048 to 4,200 new tokens: the corrected probe (`turnprobe2.patch`: answers
  of 48 tokens, decode cycles of their own, a hit check per turn) at the next test stop.
- A turn beside a second chat that is reading or decoding.
- The last stage's computed chunk in the traces (one per read; the timers give 844 ms).
- Anything at 150 W; anything on the store; more than one run of any cell.
- The front end's parts.
