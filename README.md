# Volta 12X AmPLIfy

DeepSeek-V4.1-Flash-1017 served on twelve Tesla V100-SXM2 32 GB cards from 2018, with every routed expert at
NVIDIA's NVFP4 precision on the cards. The model runs as three pipeline stages of four tensor-parallel cards. All
15,360 routed experts are held in card memory: none is compressed and none is read from host memory.

This is the build's record as of the afternoon of 2026-11-19 (its fourth release): the engine changes as install kits
through kit 9V, the measurements with their conditions, the bench of the serving set as it stands after the split count
was adopted (code, legal and prose prompts of 1K to 256K tokens, one to three streams, cold and with the document
already in the cache; the morning's bench kept beside it), the architecture, the history of every test window, the kernel
and optimization plan, the time budget and the tools whose cleaning is done. What is not yet published is listed in
[NOT-YET-PUBLISHED.md](NOT-YET-PUBLISHED.md). It will be updated.

Built by **Nick Mermiges, Esq.**, founder of [PLI Labs](https://proprietarylegal.ai/labs.html) · [proprietarylegal.ai](https://proprietarylegal.ai)

[At a glance](#at-a-glance) · [What this is](#what-this-is) · [What runs](#what-runs) · [Measured](#measured) · [What worked](#what-worked) ·
[What did not work](#what-did-not-work-or-is-not-yet-good) · [In progress](#in-progress) · [Next](#next) ·
[Hardware](#hardware) · [Running it](#running-it) · [Repository layout](#repository-layout) ·
[Related builds](#related-builds)

Every figure on this page is labelled. **MEASURED** means a log or probe on our hardware, with its test window named
(windows are short labels such as `o2`, `s4`, `t1`, `p1`; dates and UTC times are in
[docs/measurements.md](docs/measurements.md)). **ESTIMATED** means arithmetic or a model, with its basis.
**UPSTREAM** means someone else's number. A figure from one run says so.

## At a glance

**157 to 175 tokens a second** on a code request with 128,000 tokens of context, a 128,000-token file read in
**28 seconds** (4,670 tok/s), **four conversations** kept warm with **524,288 tokens** of context each. MEASURED
2026-11-19 on the serving set (kit 9S, eleven adopted switches; the tables below are the afternoon's bench, 14:09Z),
175 W per card, 1,452 MHz clock cap, four slots, cold reads, thinking off, 1,024 tokens out, mean of two runs. Prose
figures, min-max and every stream are in [Measured](#measured). A one-stream decode figure is the figure of the text
the bench generated in that cell: the two benches of 2026-11-19 generated different text in every cell (one rounding
line differs between them) and their decode cells differ by about 10 % either way, so the band is the figure; across
texts of one kind a cell spreads about ±9 %, and two repeats of one text agree within 2 %.

**Code** (a source file as the prompt)

| Context (prompt tokens) | Decode tok/s, one request | Prompt read tok/s | First token, cold | Three requests together, tok/s summed | First token, document already cached |
|---|---:|---:|---:|---:|---:|
| **1K** | 184 | 1,567 | 0.7 s | 443 | 0.2 s* |
| **8K** | 176 | 3,267 | 2.5 s | 421 | 0.2 s |
| **32K** | 176 | 4,478 | 7.3 s | 404 | 0.3 s |
| **64K** | 157 | 4,697 | 13.9 s | 364 | 0.3 s |
| **128K** | 157 | 4,670 | 28.1 s | 360 | 0.4 s |
| **256K** | 153 | 4,518 | 58.0 s | 344 | 0.6 s |

**Legal** (a brief as the prompt)

| Context (prompt tokens) | Decode tok/s, one request | Prompt read tok/s | First token, cold | Three requests together, tok/s summed | First token, document already cached |
|---|---:|---:|---:|---:|---:|
| **1K** | 102 | 1,526 | 0.7 s | 249 | 0.2 s* |
| **8K** | 93 | 3,253 | 2.5 s | 219 | 0.2 s |
| **32K** | 92 | 4,354 | 7.5 s | 213 | 0.3 s |
| **64K** | 93 | 4,561 | 14.4 s | 215 | 0.3 s |
| **128K** | 93 | 4,538 | 28.9 s | 213 | 0.4 s |
| **256K** | 92 | 4,422 | 59.3 s | 207 | 0.7 s |

\* At 1K the request's fixed cost is most of the read, so the cache cannot show.

In this bench the first and the second cold read of every one-stream cell agree within 0.15 s. In the two benches
before it the first cold read of a legal or prose document was 1 to 11 s slower than its second repeat; that did not
recur on the eleven-line set. HYPOTHESIS, from the one line that differs: the multiply library's split search on a
shape it had not yet seen, which the fixed split count removes; not established by a boot built to test it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/decode-by-context-dark.svg">
  <img alt="Decode speed by context length: one request and three together, code and legal" src="docs/img/decode-by-context-light.svg" width="880">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/first-token-dark.svg">
  <img alt="First token on a cold read against a document already cached, log scale, code and legal" src="docs/img/first-token-light.svg" width="880">
</picture>

- **Three requests together** is each stream's own decode rate while all three stream, summed: the throughput once every
  prompt is in. The pipeline carries one request per step, so long prompts are read one after another: three cold
  128K code requests released together got their first tokens after 77.4, 90.5 and 90.5 s (28.07 s alone), their
  reads interleaved, and then decoded together; counted from the earliest first token to the last request's end, the
  three give 138 tok/s on code and 111 on legal. With the document already in the cache the wait disappears: three
  code streams on one cached 128K document decode 427 tok/s over that window (legal 230), within 2 % of the sum.
- **Cached documents.** A request on a document the server has already read gets its first token in under a second at
  every length. The first time a second or a third stream shares a document the server reads it once more; after that
  every stream hits.
- The charts are drawn from the tables of [docs/measurements.md](docs/measurements.md) by
  [docs/img/make_charts.py](docs/img/make_charts.py).

## What this is

- **The aim.** Every routed expert served from NVIDIA's NVFP4 pack of DeepSeek-V4.1-Flash-1017, whose expert values
  equal the official MXFP4 experts, on V100 cards.
- **Why twelve cards.** One expert in NVFP4 is 4,874,240 bytes on each of its four cards. All forty layers are
  278.9 GiB of expert records. Eight 32 GB cards hold 254 GiB (ARITHMETIC). So an eight-card build must compress
  some experts or read them from host memory. Twelve cards hold them all.
- **Why three stages cut at layers 14 and 28.** Twenty layers of experts are 34.86 GiB per card, more than one
  card. A stage must start at a layer that is an index source, and the last stage also carries the drafter and
  the head. That leaves one legal cut that fits: layers 0-13, 14-27 and 28-39 (ARITHMETIC). It is the layout of
  PLI Labs' 1Cat-vLLM pull request [#1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031), which ran this model on
  twelve cards at about 3 tok/s on an eager engine (UPSTREAM).
- **What had to be built.** The engine this build descends from accepted two stages cut at layer 20 and nothing
  else. The kit in this repository teaches it a third stage, fixes the scheduler for a pipeline of three, cuts the
  middle stage's memory and adds a card memory limit. The design is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **What it does and does not do.** It serves every routed expert at NVFP4 precision from card memory, with up to
  four conversations kept warm and a context of 524,288 tokens. One request decodes at 153 to 184 tok/s on code and
  83 to 102 tok/s on legal and prose prompts of 1K to 256K tokens (MEASURED 2026-11-19 at 175 W, the afternoon's
  bench below); the pipeline carries one request per step, so prompts are read one after another and three requests
  together decode at about 2.2 to 2.5 times one stream's rate once every prompt is in. It is not a batched-decode
  engine.
- **What it is tuned for.** Long-context agent work: one request at a time, prompts of 60,000 to 200,000 tokens that
  grow by small turns. Since 2026-11-17 it is the owner's default DeepSeek server.

## What runs

The standing configuration: the serving set since window b4a (2026-11-19 13:13Z): the set every optimization of the
plan's list left (window c2a, 03:55Z) plus the split count adopted in window b4a; re-recorded after window b3a
(14:08Z).

| Item | Value |
|---|---|
| Model | DeepSeek-V4.1-Flash-1017: 40 layers, 384 routed experts per layer, DSpark drafter |
| Routed experts | 15,360 in NVFP4 on the cards: 5,376 / 5,376 / 4,608 by stage; 0 in host memory |
| Non-expert tensors | the ISTA-DASLab GSQ-RCO GGUF |
| Layout | PP3 x TP4: three pipeline stages of four tensor-parallel cards, layers 0-13 / 14-27 / 28-39 |
| Slots | four (accepted in window s4); three remain available |
| Prompt chunks | 4,092 rows (`max-num-batched-tokens 4096`) |
| Cache release | only at prompt steps above 8,192 rows (`VLLM_V41_EMPTY_CACHE_TOKENS=8192`, windows p1 and p2s); the engine's default released at every step of 2,048 rows or more |
| Context | 524,288 tokens |
| Card limit | 31,998 MiB on every stage (`VLLM_V41_CARD_LIMIT_MIB=31998`, a fuse at the owner's line) |
| Engine | 1Cat-vLLM's DeepSeek-V4.1 package as carried in PLI Labs' engine (commit `f20d155`) with kit 9S over it (engine `9733024` plus the batched projection, 18 files): [`kit/PP3KIT9S-over-cut`](kit/PP3KIT9S-over-cut); every kit in [engine/README.md](engine/README.md) |
| Adopted switches | eleven lines: `TM_GEMM_TUNE=max_splits=1 VLLM_V41_DRAFT_GEMV2=1 VLLM_V41_EMPTY_CACHE_TOKENS=8192 VLLM_V41_INDEX_TC=1 VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD=128 VLLM_V41_PP_ANSWER_NO_SLEEP=1 VLLM_V41_PP_EMPTY_AHEAD=1 VLLM_V41_PP_MID_PREFILL_SKIP=1 VLLM_V41_PP_TURN_CHUNKS=3 VLLM_V41_PREFIX_TOKEN_CACHE=1 VLLM_V41_WO_A_BATCHED_4K=1` (the window of each is named under What worked; the split count is the multiply library's own switch, not a kit); the serving record carries their digest |
| Accepted capture | `accepted/quality-TC3-WOA4K-S1.json` (window b4a: the split count is another rounding) while it is adopted; `quality-TC3-WOA4K.json` (window c2a) before it; `quality-TC3.json` (window p5) before that; window o2's capture otherwise; decided in one place from the adopted lines |
| Power and clock | 175 W per card and a 1,452 MHz clock ceiling (cards at 1,447 MHz) since 2026-11-18 12:09:23Z. Before: 150 W and 1,477 MHz to 2026-11-17 18:58Z; 200 W to 2026-11-18 11:15:48Z (the host tripped under four long chats); 150 W from 11:45Z |
| Serving | every start check runs on a test port and is recorded; the serving boot on the public port sends no request and needs a record that matches the installed bytes ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), section 13) |

Not in what serves: the conversation store (every state check passed in windows st4 and st7; it failed its timing line
in st7 and stays off), the lean pipeline hand-off (failed its line in window c3b), the packed expert records (failed
theirs in window c7a), the exact four-card sum (kit 9V, `VLLM_V41_TP_EXACT_SUM`: failed its closeness line in window
b3a), the drafter off (the quality window's one-slot boots only), and the loop statistics `VLLM_V41_PP_LOOP_STATS` (a
diagnostic of the windows, never served).

## Measured

MEASURED on 2026-11-17, 2026-11-18 and 2026-11-19; each row names its window, and its power where it is not the
window's standing power (150 W through window sv1; 200 W from 2026-11-17 18:58Z; 175 W and the 1,452 MHz ceiling since
2026-11-18 12:09Z). The full tables with dates and times are in [docs/measurements.md](docs/measurements.md); the
morning bench of 2026-11-19 on the ten-line set is kept in
[docs/measurements-2026-11-19-morning.md](docs/measurements-2026-11-19-morning.md) and the bench of 2026-11-18 on the
set before it in [docs/measurements-2026-11-18.md](docs/measurements-2026-11-18.md).

### The serving set since window b4a (2026-11-19, kit 9S, eleven switches; 175 W, clock ceiling 1,452 MHz)

Kit 9S with the eleven adopted lines (table above): the set every optimization of the plan's list left, plus the split
count (`TM_GEMM_TUNE=max_splits=1`, window b4a, 12:30Z to 13:13Z). Every row MEASURED on 2026-11-19 at 175 W per card
and the 1,452 MHz ceiling; each window is named. The split count moves rounding, so the bench on this set generated
different text from the morning's in every cell and its decode cells moved with the text (-10.1 to +9.0 % by cell,
mean -0.9 %); the round itself did not move (one fresh request +1.3 % in window b4a).

| What | Figure | Window |
|---|---|---|
| Prefill, cold reads of 60,706 / 93,028 / 247,953 / 425,783 tokens, the only request, against the ten lines on the same boot kind | 4,659 / 4,756 / 4,552 / 4,314 tok/s against 4,664 / 4,747 / 4,515 / 4,327 (-0.1 / +0.2 / +0.8 / -0.3 %); first token 13.03 / 19.56 / 54.47 / 98.70 s | b4a |
| Decode at four slots, against the ten lines | one request fresh 137.3 against 135.6 tok/s (+1.3 %); one repeated and four together +4.7 and +4.3 % (the text moved; report only) | b4a |
| Closeness of the served answers to the ten-line capture (`quality-TC3-WOA4K.json`; the split count is another rounding) | identical 1 of 7, median first divergence 45 to 48 tokens, median of the mean \|dlogprob\| 0.0350 to 0.0366 nats; CLEAN | b4a |
| A cache hit against its own cold read (nine prompts of the bench at 32K to 128K, greedy, 32 tokens) | identical 4 of 9; median first divergence 25, median mean \|dlogprob\| 0.0275; every resend a hit by timing (0.25 to 0.45 s against 7.2 to 28.6 s cold) | b3a |
| Highest card sample, four-chat rehearsal (30 requests, largest prompt about 153,600 tokens) | 30,832 MiB; by stage 30,594 / 30,832 / 28,328 | b4a, b3a |
| The afternoon's bench, one stream decode at 128K (two repeats of one text; the spread across texts of a kind is about ±9 %) | code 157.4, legal 93.1, prose 85.6 tok/s (1K: 183.8 / 101.5 / 86.1; 256K: 153.4 / 92.4 / 84.5) | bench 14:09Z |
| The afternoon's bench, one cold read, prefill at 128K and 256K | code 4,670 and 4,518 tok/s (first token 28.07 and 58.03 s); legal 4,538 and 4,422 (28.88 and 59.28 s); prose 4,586 and 4,456 (28.58 and 58.83 s); first and second repeats within 0.15 s in every cell | bench 14:09Z |
| The afternoon's bench, three streams released together at 128K, sum of the streams' decode rates | code 360.0, legal 212.6, prose 198.3 tok/s (from the earliest first token to the last request's end: 138.0 / 110.8 / 106.9) | bench 14:09Z |
| The prefix measure: first token on a hit at 128K; three streams on one cached 128K document, the all-hit repeat | 0.43 / 0.44 / 0.44 s (code / legal / prose); code 426.6, legal 230.1, prose 221.0 tok/s over that window (432.7 / 232.0 / 222.5 summed) | measure 15:40Z |
| Highest card sample over the bench and the measure (2 h 15 min, 1,513 samples) | 30,418 MiB | bench 14:09Z |

### The set before it: since window c2a (2026-11-19, kit 9S, ten switches; served 03:55Z to 12:30Z; 175 W, clock ceiling 1,452 MHz)

Kit 9S with the ten adopted lines (table above): the set every optimization of the plan's list left. Every row
MEASURED on 2026-11-19 at 175 W per card and the 1,452 MHz ceiling; each window is named.

| What | Figure | Window |
|---|---|---|
| Prefill, cold reads of 60,705 / 93,025 / 247,948 / 425,779 tokens, the only request | 4,663 / 4,741 / 4,531 / 4,309 tok/s (first token 13.02 / 19.62 / 54.72 / 98.81 s); the same boot without the batched projection 4,476 / 4,559 / 4,358 / 4,028 | c2a |
| Decode round at about 128K of context, one request (medians of the loop's steady lines) | 25.895 ms (a one-step turn of 880 on a 128,585-token chat); 25.588 ms after a cold read of 127,663 tokens (c7a's reference boot, the same engine path as kit 9S); c3b's reference boot 26.253 / 25.954 | c7a, c3b |
| Decode round on a short chat, one request | 24.388 ms (a one-step turn), 24.284 ms (after a read) | c7a |
| Closeness of the served answers to kit 9N's accepted capture (the batched projection moves rounding) | identical 1 of 7, median first divergence 58 tokens, median of the mean \|dlogprob\| 0.0354 nats | c2a |
| Highest card sample, four-chat rehearsal (30 requests, largest prompt about 153,600 tokens) | 30,832 MiB (c2a, c7a), 30,830 (c3b); by stage 30,594 / 30,832 / 28,328 | c2a, c3b, c7a |
| Load, launch to ready | 164 to 197 s | st7, c3a |
| The morning's bench, one stream decode at 128K (two repeats of one text; the spread across texts of a kind is about ±9 %) | code 175.0, legal 94.2, prose 79.9 tok/s (1K: 178.3 / 97.3 / 85.6; 256K: 168.5 / 90.0 / 84.4) | morning bench |
| The morning's bench, one cold read, prefill at 128K and 256K | code 4,657 and 4,360 tok/s (first token 28.15 and 60.23 s); legal 3,944 and 4,094 (34.09 and 64.46 s, means of a slow first repeat and a fast second: 28.70 and 59.23 s on the second); prose 4,331 and 4,247 (30.40 and 61.89 s; second repeats 28.37 and 58.79) | morning bench |
| The morning's bench, three streams released together at 128K, sum of the streams' decode rates | code 391.5, legal 216.1, prose 192.3 tok/s (from the earliest first token to the last request's end: 142.9 / 112.3 / 105.7) | morning bench |
| The prefix measure again: first token on a hit at 128K; three streams on one cached 128K document, the all-hit repeat | 0.42 to 0.43 s; code 399.8, legal 242.2, prose 221.5 tok/s over that window (405.1 / 244.0 / 223.2 summed) | morning measure |
| Highest card sample over the bench and the measure (2 h 15 min, 1,522 samples) | 30,418 MiB | morning bench |

### The set before that: since window p6b (2026-11-18, kit 9N, nine switches; 175 W, clock ceiling 1,452 MHz)

Kit 9N with the nine adopted lines (today's ten without the batched projection). Every row MEASURED on 2026-11-18 at
175 W per card and the 1,452 MHz ceiling unless it says otherwise; each switch's own window is named.

| What | Figure | Window |
|---|---|---|
| Decode round at about 128K of context, one request | 25.8 ms (a one-step turn of 880 on a 128,585-token chat); 25.7 ms after a cold read of 127,663 tokens. The same kit without the tensor-core scorer: 26.6 / 26.7 ms; before the loop switches (200 W, p3b): 28.2 / 26.3 ms | p6b, p3b |
| Decode round on a short chat, one request | 24.4 ms (25.5 before the loop switches at 200 W) | p6b, p3b |
| Decode per token at 128K on forced 1,000-token sequences (a figure of the round on fixed text, not an answer speed) | 6.86 ms after a cold read (145.4 tok/s in the p6 TC boot), 7.53 ms after a one-step turn; the reference boot 7.09 / 7.88 | p6b, p6 |
| Prefill, one cold read of about 127,663 tokens | 4,587 tok/s (200 W, before the index change: 4,564 to 4,593) | p6b |
| Prefill, one cold read of 425,784 tokens, the only request on the serving port | first token 102.73 s, 4,145 tok/s (p1's `off` boot at 200 W before the skip and the release setting: 173.49 s, 2,454 tok/s) | p5u, p1 |
| Four requests together, fresh text, sum | 337.9 tok/s (the reference boot 316.6); repeated text 451.6 | p6b |
| First token of a turn alone on a 128,000-token chat (medians of two), with the turn chunks | +880 new tokens 1.131 s; +2,048 1.471 s; +4,200 2.319 s; +10,000 3.623 s (without the chunks 1.184 / 2.061 / 3.687 / 4.963) | p5 |
| A turn of 880 tokens beside another chat's 248,000-token read, with the hold (medians of three) | first token 3.26 s, whole turn 5.12 s (11.99 without the hold); the read's first token 60.99 s (62.66) | p5 |
| The front of a turn on a 128,000-token chat (template, tokenizing, hashing) with the token cache | 47.2 ms against 308.0 (median of 20 turns) | p4b |
| Two boots without the drafter, eight forced cells at 4.2K / 94K / 128K / 252K and after a turn, temperature zero | text equal 8 of 8 | q1 |
| Highest card sample in the four-chat rehearsal (30 requests, largest prompt about 153,600 tokens) | 30,814 MiB (ceiling 31,800) | p5u, p6b |
| Load, launch to ready | 134 s | p5 |

The reads with the middle-stage skip and the release setting, at 200 W (window p2, `adopt` against `off` in the
same window): 60,704 tokens 4,562 tok/s (+19.7 %), 93,027 4,515 (+22.1 %), 247,952 4,405 (+34.1 %), 425,776
4,223 (+53.2 %). At 150 W the 426K read took 152.0 s against 100.8 s at 200 W (windows p3r and p2): long prompts
about a third slower at 150 W; 175 W gives 102.7 to 104.5 s.

### Speed (2026-11-17, kit 6c, no switches)

| What | Figure | Window |
|---|---|---|
| One request, fresh text, four slots, 150 W | 128.1 tok/s | s4 |
| One request, repeated text, four slots, 150 W | 176.6 tok/s | s4 |
| Four requests together, fresh text, sum, 150 W | 306.6 tok/s (60.5 to 85.5 each) | s4 |
| Four requests together, repeated text, sum, 150 W | 453.4 tok/s (110.9 to 117.8 each) | s4 |
| Prompt read, one 62,263-token prompt, 150 W | 3,284 tok/s | s4 |
| 200 W, one request, fresh / repeated (one run) | 129.7 / 179.2 tok/s | after g1 |
| 200 W, four together, fresh / repeated, sum (one run) | 312.1 / 447.5 tok/s | after g1 |
| 200 W, prompt read, one 62,263-token prompt (one run) | 3,733 tok/s | after g1 |
| 200 W, reads of 60,706 / 93,025 / 247,953 / 425,777 tokens (one run) | 3,850 / 3,833 / 3,316 / 2,454 tok/s | p1 |
| 200 W, one-request decode round (one run) | 25.1 ms | t1 |
| Three requests together, fresh / repeated, sum, 150 W | 288.0 / 444.6 tok/s | o2 |
| One request, fresh / repeated, three slots, 150 W | 129.5 / 177.3 tok/s | o2 |
| Load, launch to ready | 134 s | s4 |

A step of the pipeline carries one request, so a slot does not batch: four requests together gave 6 % more than
three on fresh text (306.6 against 288.0 tok/s). These sums add each request's own rate; by wall clock four together
gave 242.6 tok/s (an independent review's reading). At 200 W the prompt read rose 13.7 % and decode stayed within
2 %.

**The afternoon's bench, 2026-11-19 (the serving set after the split count was adopted).** MEASURED on 2026-11-19, 14:09:20Z to
15:40:33Z, on the serving set of window b4a (kit `PP3KIT9S-over-cut`, the eleven adopted lines, four slots; the server not restarted for the
bench), 175 W per card and the 1,452 MHz clock ceiling: the same bench as the two before it, unchanged: code, legal and prose prompts of
1,024 / 8,192 / 32,768 / 65,536 / 131,072 / 262,144 tokens, read cold, one, two and three streams released together, two repeats each
(108 groups, 216 measured requests; greedy, 1,024 tokens out, thinking off; judge PASS completeness, the token digest of every lone repeat
matched its pair 198 of 198). "Prefill tok/s" is the prompt's tokens over the time to its first token. The two aggregate columns: **Total
decode tok/s** is all streams' output tokens over the window from the earliest first token to the last request's end (the bench tool's
arithmetic); with two or three cold streams the prompts are read one after another, interleaved, so the first tokens of three 128K code
streams came at 77.4, 90.5 and 90.5 s (28.07 s alone) and the three then decoded together. **Sum of decode rates** adds each stream's own
decode rate while all of them stream: the decode throughput once every prompt is in. Three findings. (1) A one-stream decode figure is the
figure of the text the bench generated: the split count is another rounding, so the greedy text moved in every cell against the morning's
bench and the decode per cell moved with it, -10.1 to +9.0 %, the mean of the 54 aggregate cells -0.9 %, while the decode round itself did
not change (window b4a: one fresh request +1.3 %); the spread across texts of one kind is about ±9 %, two repeats of one text agree within
2 %. (2) The cold reads are within -0.7 to +0.7 % of the morning's second repeats at every length. (3) The slow first cold read of a legal
or prose document seen in the two benches before (1 to 11 s slower than its second repeat) did not recur: the first and the second repeat
agree within 0.15 s in every one-stream cell (HYPOTHESIS, from the one line that differs: the library's split search on a shape not yet
seen, which the fixed split count removes; not established by a boot built for it). The morning's bench and its prefix tables are kept
unchanged in [docs/measurements-2026-11-19-morning.md](docs/measurements-2026-11-19-morning.md), the 2026-11-18 bench in
[docs/measurements-2026-11-18.md](docs/measurements-2026-11-18.md).

#### Code

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | cold, thinking off, 1,024 out. Mean [min–max], two runs.

| Prompt | Streams | TTFT s by stream | Prefill tok/s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates |
|---|---:|---|---|---|---:|---:|
| 1K | 1 | 0.65 [0.64–0.67] | 1567 [1538–1596] | 183.8 [183.7–184.0] | 183.8 [183.7–184.0] | 183.8 [183.7–184.0] |
| 1K | 2 | 0.82 [0.67–0.98] / 0.81 [0.67–0.95] | 1291 [1049–1533] / 1305 [1080–1530] | 171.7 [167.8–175.6] / 171.2 [167.1–175.4] | 334.1 [333.6–334.7] | 342.9 [342.7–343.2] |
| 1K | 3 | 1.09 [0.95–1.23] / 0.67 [0.67–0.67] / 1.09 [0.95–1.22] | 957 [832–1083] / 1532 [1526–1539] / 958 [838–1079] | 150.7 [147.7–153.6] / 142.1 [141.9–142.3] / 150.6 [147.6–153.7] | 425.2 [424.7–425.7] | 443.4 [443.1–443.7] |
| 8K | 1 | 2.51 [2.47–2.54] | 3267 [3220–3314] | 175.5 [173.9–177.2] | 175.5 [173.9–177.2] | 175.5 [173.9–177.2] |
| 8K | 2 | 5.22 [2.47–7.97] / 5.21 [2.47–7.95] | 2173 [1028–3317] / 2172 [1031–3312] | 148.5 [124.1–172.8] / 148.4 [124.0–172.8] | 179.4 [179.2–179.5] | 296.9 [296.8–296.9] |
| 8K | 3 | 8.02 [2.48–13.56] / 8.02 [2.48–13.56] / 8.75 [8.74–8.76] | 1956 [604–3308] / 1954 [604–3304] / 937 [936–938] | 143.0 [113.3–172.8] / 142.9 [113.3–172.5] / 135.0 [135.0–135.1] | 180.4 [180.3–180.5] | 421.0 [420.9–421.1] |
| 32K | 1 | 7.32 [7.31–7.33] | 4478 [4472–4485] | 175.8 [172.6–178.9] | 175.8 [172.6–178.9] | 175.8 [172.6–178.9] |
| 32K | 2 | 15.32 [12.06–18.58] / 15.34 [12.05–18.62] | 2240 [1763–2716] / 2239 [1760–2719] | 146.4 [117.6–175.1] / 146.5 [118.0–175.1] | 165.1 [165.0–165.3] | 292.9 [292.7–293.2] |
| 32K | 3 | 28.01 [25.59–30.42] / 15.98 [15.98–15.99] / 28.01 [25.59–30.43] | 1179 [1077–1280] / 2050 [2050–2050] / 1179 [1077–1281] | 155.1 [136.5–173.7] / 93.9 [93.8–93.9] / 155.3 [136.4–174.3] | 151.0 [150.9–151.1] | 404.2 [403.9–404.6] |
| 64K | 1 | 13.95 [13.95–13.96] | 4697 [4696–4698] | 157.4 [155.6–159.1] | 157.4 [155.6–159.1] | 157.4 [155.6–159.1] |
| 64K | 2 | 28.56 [25.20–31.92] / 28.56 [25.20–31.92] | 2327 [2053–2601] / 2327 [2053–2601] | 131.2 [107.3–155.2] / 131.4 [107.5–155.3] | 153.8 [153.7–153.8] | 262.6 [262.6–262.7] |
| 64K | 3 | 43.12 [35.73–50.51] / 40.62 [35.66–45.58] / 48.07 [45.53–50.62] | 1566 [1297–1834] / 1638 [1438–1838] / 1367 [1295–1439] | 120.4 [86.4–154.4] / 104.3 [86.5–122.0] / 138.8 [122.6–155.0] | 142.9 [142.8–142.9] | 363.5 [363.5–363.5] |
| 128K | 1 | 28.07 [28.06–28.07] | 4670 [4669–4671] | 157.4 [155.0–159.7] | 157.4 [155.0–159.7] | 157.4 [155.0–159.7] |
| 128K | 2 | 53.23 [53.21–53.25] / 60.28 [60.26–60.29] | 2462 [2462–2463] / 2174 [2174–2175] | 104.9 [104.9–105.0] / 155.8 [155.4–156.2] | 150.1 [150.0–150.3] | 260.8 [260.3–261.2] |
| 128K | 3 | 77.41 [77.35–77.46] / 90.48 [87.82–93.14] / 90.46 [87.86–93.05] | 1693 [1692–1695] / 1450 [1407–1492] / 1450 [1409–1492] | 83.5 [83.3–83.7] / 138.5 [121.0–156.0] / 137.9 [120.9–155.0] | 138.0 [137.8–138.2] | 360.0 [359.3–360.7] |
| 256K | 1 | 58.03 [57.98–58.07] | 4518 [4514–4521] | 153.4 [150.6–156.1] | 153.4 [150.6–156.1] | 153.4 [150.6–156.1] |
| 256K | 2 | 119.96 [119.87–120.06] / 112.32 [112.27–112.37] | 2185 [2184–2187] / 2334 [2333–2335] | 152.2 [151.9–152.5] / 99.0 [98.8–99.2] | 142.5 [142.3–142.7] | 251.2 [250.7–251.7] |
| 256K | 3 | 165.09 [164.96–165.22] / 179.15 [176.51–181.78] / 179.17 [176.23–182.12] | 1588 [1587–1589] / 1464 [1442–1485] / 1463 [1439–1488] | 78.6 [78.5–78.7] / 132.7 [114.9–150.5] / 133.1 [114.8–151.3] | 129.8 [129.8–129.8] | 344.4 [344.0–344.8] |

#### Legal

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | cold, thinking off, 1,024 out. Mean [min–max], two runs.

| Prompt | Streams | TTFT s by stream | Prefill tok/s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates |
|---|---:|---|---|---|---:|---:|
| 1K | 1 | 0.67 [0.66–0.68] | 1526 [1507–1545] | 101.5 [101.0–102.1] | 101.5 [101.0–102.1] | 101.5 [101.0–102.1] |
| 1K | 2 | 0.95 [0.95–0.95] / 0.67 [0.67–0.67] | 1075 [1073–1077] / 1520 [1519–1522] | 97.4 [97.1–97.7] / 95.0 [94.7–95.3] | 189.7 [189.2–190.3] | 192.4 [191.8–192.9] |
| 1K | 3 | 1.33 [1.24–1.41] / 0.67 [0.67–0.68] / 1.33 [0.96–1.70] | 775 [726–824] / 1524 [1515–1532] / 835 [604–1066] | 84.4 [83.4–85.4] / 80.3 [78.9–81.7] / 84.4 [83.5–85.3] | 240.3 [235.9–244.7] | 249.2 [247.7–250.7] |
| 8K | 1 | 2.52 [2.52–2.52] | 3253 [3251–3255] | 92.9 [92.5–93.4] | 92.9 [92.5–93.4] | 92.9 [92.5–93.4] |
| 8K | 2 | 5.28 [2.53–8.04] / 5.30 [2.53–8.07] | 2127 [1019–3234] / 2126 [1015–3236] | 82.0 [74.0–90.0] / 82.0 [74.4–89.5] | 120.9 [120.6–121.2] | 163.9 [163.5–164.4] |
| 8K | 3 | 5.70 [2.55–8.85] / 5.71 [2.55–8.87] / 13.71 [13.70–13.72] | 2068 [925–3211] / 2070 [923–3216] / 598 [597–598] | 67.3 [60.7–73.9] / 67.3 [60.7–73.9] / 84.3 [84.3–84.4] | 131.8 [131.7–131.8] | 218.9 [218.9–218.9] |
| 32K | 1 | 7.53 [7.52–7.53] | 4354 [4351–4356] | 92.5 [92.2–92.8] | 92.5 [92.2–92.8] | 92.5 [92.2–92.8] |
| 32K | 2 | 18.96 [18.94–18.97] / 12.41 [12.41–12.42] | 1729 [1727–1730] / 2640 [2639–2641] | 89.1 [88.6–89.5] / 71.2 [71.0–71.5] | 113.5 [113.0–113.9] | 160.3 [159.7–161.0] |
| 32K | 3 | 16.47 [16.46–16.48] / 28.63 [26.15–31.11] / 28.62 [26.16–31.07] | 1990 [1988–1991] / 1153 [1053–1253] / 1153 [1055–1252] | 54.5 [54.4–54.6] / 79.4 [74.1–84.8] / 79.4 [73.9–84.9] | 115.0 [114.9–115.1] | 213.3 [213.1–213.5] |
| 64K | 1 | 14.37 [14.34–14.39] | 4561 [4553–4570] | 93.4 [92.9–94.0] | 93.4 [92.9–94.0] | 93.4 [92.9–94.0] |
| 64K | 2 | 29.29 [25.94–32.65] / 29.30 [25.93–32.68] | 2267 [2007–2527] / 2267 [2005–2528] | 80.9 [71.7–90.1] / 81.1 [71.3–90.9] | 113.4 [113.1–113.7] | 162.0 [161.4–162.6] |
| 64K | 3 | 49.18 [46.64–51.72] / 41.71 [36.74–46.68] / 44.38 [36.73–52.04] | 1336 [1267–1405] / 1594 [1404–1784] / 1522 [1259–1784] | 80.0 [73.1–86.8] / 64.3 [53.6–75.0] / 70.5 [54.4–86.5] | 113.9 [113.1–114.6] | 214.7 [213.2–216.3] |
| 128K | 1 | 28.88 [28.87–28.90] | 4538 [4536–4541] | 93.1 [92.8–93.3] | 93.1 [92.8–93.3] | 93.1 [92.8–93.3] |
| 128K | 2 | 61.75 [61.73–61.77] / 54.66 [54.64–54.68] | 2123 [2122–2123] / 2398 [2397–2399] | 89.7 [89.7–89.8] / 69.9 [69.9–69.9] | 110.7 [110.6–110.7] | 159.6 [159.6–159.7] |
| 128K | 3 | 95.35 [95.35–95.36] / 79.56 [79.54–79.58] / 90.07 [90.04–90.11] | 1375 [1375–1375] / 1647 [1647–1648] / 1455 [1455–1456] | 86.1 [86.1–86.2] / 52.6 [52.6–52.7] / 73.8 [73.7–74.0] | 110.8 [110.8–110.9] | 212.6 [212.4–212.8] |
| 256K | 1 | 59.28 [59.25–59.31] | 4422 [4420–4424] | 92.4 [92.2–92.6] | 92.4 [92.2–92.6] | 92.4 [92.2–92.6] |
| 256K | 2 | 118.90 [115.04–122.75] / 118.92 [115.17–122.68] | 2207 [2136–2279] / 2207 [2137–2276] | 78.3 [67.6–89.0] / 78.4 [67.5–89.2] | 107.0 [107.0–107.1] | 156.7 [156.5–156.8] |
| 256K | 3 | 178.52 [170.18–186.87] / 178.53 [169.93–187.13] / 181.23 [181.10–181.37] | 1472 [1403–1540] / 1472 [1401–1543] / 1446 [1445–1448] | 67.7 [50.1–85.3] / 67.7 [50.2–85.3] / 71.5 [71.5–71.6] | 106.0 [105.9–106.0] | 207.0 [207.0–207.0] |

#### Prose

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | cold, thinking off, 1,024 out. Mean [min–max], two runs.

| Prompt | Streams | TTFT s by stream | Prefill tok/s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates |
|---|---:|---|---|---|---:|---:|
| 1K | 1 | 0.64 [0.64–0.64] | 1603 [1600–1607] | 86.1 [85.8–86.3] | 86.1 [85.8–86.3] | 86.1 [85.8–86.3] |
| 1K | 2 | 0.78 [0.64–0.92] / 0.78 [0.65–0.92] | 1348 [1107–1588] / 1343 [1117–1570] | 81.6 [80.8–82.5] / 81.6 [80.8–82.5] | 161.4 [161.4–161.4] | 163.3 [163.3–163.3] |
| 1K | 3 | 1.22 [1.21–1.22] / 0.80 [0.66–0.94] / 0.80 [0.66–0.94] | 842 [841–844] / 1318 [1085–1551] / 1322 [1089–1554] | 72.6 [72.6–72.7] / 70.6 [70.0–71.2] / 70.6 [69.9–71.4] | 209.6 [209.4–209.8] | 213.9 [213.7–214.1] |
| 8K | 1 | 2.49 [2.48–2.49] | 3296 [3294–3297] | 89.6 [88.8–90.3] | 89.6 [88.8–90.3] | 89.6 [88.8–90.3] |
| 8K | 2 | 7.97 [7.97–7.98] / 2.49 [2.48–2.49] | 1027 [1027–1028] / 3296 [3291–3300] | 86.9 [86.8–87.0] / 72.3 [72.2–72.5] | 118.6 [118.4–118.7] | 159.3 [159.1–159.5] |
| 8K | 3 | 11.19 [8.76–13.62] / 8.08 [2.53–13.62] / 5.65 [2.50–8.79] | 768 [601–935] / 1918 [601–3235] / 2106 [932–3279] | 76.8 [71.7–81.8] / 70.6 [59.5–81.6] / 65.7 [59.4–71.9] | 129.9 [129.7–130.0] | 213.0 [212.7–213.3] |
| 32K | 1 | 7.39 [7.38–7.41] | 4432 [4423–4441] | 94.7 [94.4–95.1] | 94.7 [94.4–95.1] | 94.7 [94.4–95.1] |
| 32K | 2 | 15.47 [12.17–18.76] / 15.47 [12.19–18.75] | 2220 [1747–2692] / 2218 [1748–2688] | 82.3 [72.9–91.7] / 82.4 [72.8–92.0] | 115.5 [115.4–115.6] | 164.7 [164.6–164.9] |
| 32K | 3 | 21.00 [16.16–25.84] / 28.30 [25.81–30.79] / 23.44 [16.16–30.72] | 1648 [1268–2028] / 1167 [1064–1269] / 1547 [1067–2027] | 66.1 [56.0–76.3] / 82.1 [76.5–87.7] / 71.7 [55.8–87.6] | 116.9 [116.7–117.0] | 219.9 [219.8–220.1] |
| 64K | 1 | 14.22 [14.21–14.22] | 4610 [4609–4611] | 83.3 [82.9–83.8] | 83.3 [82.9–83.8] | 83.3 [82.9–83.8] |
| 64K | 2 | 25.86 [25.63–26.10] / 32.62 [32.38–32.86] | 2534 [2511–2557] / 2009 [1994–2024] | 65.1 [65.0–65.3] / 80.7 [80.5–80.9] | 105.3 [105.1–105.5] | 145.8 [145.4–146.2] |
| 64K | 3 | 48.73 [46.27–51.20] / 41.24 [36.32–46.16] / 43.77 [36.21–51.34] | 1348 [1280–1417] / 1612 [1420–1804] / 1543 [1277–1810] | 71.8 [67.0–76.6] / 58.8 [50.2–67.4] / 63.3 [50.3–76.3] | 108.1 [107.9–108.3] | 194.0 [193.6–194.4] |
| 128K | 1 | 28.58 [28.54–28.62] | 4586 [4579–4593] | 85.6 [85.2–85.9] | 85.6 [85.2–85.9] | 85.6 [85.2–85.9] |
| 128K | 2 | 57.61 [54.04–61.17] / 57.60 [54.05–61.15] | 2284 [2143–2426] / 2284 [2143–2425] | 74.6 [65.8–83.4] / 74.6 [65.9–83.3] | 105.5 [105.4–105.6] | 149.2 [149.1–149.4] |
| 128K | 3 | 91.95 [89.31–94.59] / 91.89 [89.23–94.56] / 78.75 [78.67–78.82] | 1427 [1386–1468] / 1428 [1386–1469] / 1664 [1663–1666] | 74.3 [69.0–79.5] / 74.0 [68.6–79.5] / 50.0 [49.8–50.2] | 106.9 [106.7–107.1] | 198.3 [197.9–198.7] |
| 256K | 1 | 58.83 [58.76–58.91] | 4456 [4450–4461] | 84.5 [83.8–85.2] | 84.5 [83.8–85.2] | 84.5 [83.8–85.2] |
| 256K | 2 | 117.74 [113.86–121.63] / 117.77 [113.98–121.56] | 2229 [2155–2302] / 2228 [2156–2300] | 72.5 [63.1–81.9] / 72.4 [63.1–81.7] | 101.4 [101.3–101.4] | 144.9 [144.8–145.0] |
| 256K | 3 | 181.80 [178.16–185.44] / 173.27 [166.96–179.59] / 176.10 [168.30–183.89] | 1443 [1414–1471] / 1515 [1460–1570] / 1492 [1426–1558] | 72.2 [66.4–78.0] / 56.8 [47.5–66.1] / 62.6 [47.2–77.9] | 101.7 [101.5–102.0] | 191.6 [191.3–191.8] |

**With prefix sharing, again: the same measure on the eleven-line set.** For each length and kind one cold "warm read" put the
document in the prefix cache; then one, two and three streams on that document, released together, two repeats (108 groups,
216 measured requests; MEASURED 2026-11-19 15:40:33Z to 16:24:31Z on the same serving set, 175 W, the 1,452 MHz
ceiling, right after the bench above, the server not restarted). A hit is a first token under a quarter of the warm read's.
Hits 150 of 216 as in the two measures before: the 36 requests at 1K are not hits by that rule, and 30 requests re-read the
document, one per new slot the first time a second or a third stream shared it, never after. On a hit the first token at 128K
came in 0.43 to 0.44 s and at 256K in 0.63 to 0.73 s; three streams on one cached 128K document, the all-hit repeat: code
432.7 tok/s summed and 426.6 over the window, legal 232.0 and 230.1, prose 222.5 and 221.0. 0 of 150 hits reproduced the
warm read's token digest and all 30 re-reads did (the judge's count of 36 adds six of the 1K requests). The warm reads were
within -0.4 to +0.3 % of the morning's at 32K and longer (code 28.03 s at 128K, 57.83 at 256K). Hit decode rates moved with
the generated text as in the bench above.

#### Code, with prefix sharing

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | document in the prefix cache, thinking off, 1,024 out. Mean [min–max], two runs; the warm read is the one cold read that put the document in the cache.

| Prompt | Streams | TTFT s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates | Hits | Overlap |
|---|---:|---|---|---:|---:|---|---|
| 1K | warm read | 0.66 (1,547 tok/s) | 184.4 | - | - | - | - |
| 1K | 1 | 0.21 [0.21–0.22] | 185.6 [182.8–188.3] | 185.6 [182.8–188.3] | 185.6 [182.8–188.3] | 0/2 | together |
| 1K | 2 | 0.50 [0.28–0.72] / 0.21 [0.20–0.22] | 176.9 [175.3–178.4] / 169.4 [162.0–176.9] | 337.4 [322.0–352.8] | 346.3 [337.3–355.3] | 0/4 | together |
| 1K | 3 | 0.63 [0.34–0.93] / 0.33 [0.25–0.40] / 0.30 [0.22–0.39] | 155.0 [154.2–155.9] / 148.8 [143.6–154.1] / 148.7 [140.7–156.8] | 439.2 [417.7–460.8] | 452.6 [438.5–466.7] | 0/6 | together |
| 8K | warm read | 2.47 (3,316 tok/s) | 177.5 | - | - | - | - |
| 8K | 1 | 0.22 [0.22–0.23] | 188.1 [185.4–190.8] | 188.1 [185.4–190.8] | 188.1 [185.4–190.8] | 2/2 | together |
| 8K | 2 | 0.25 [0.21–0.30] / 3.14 [0.23–6.06] | 152.4 [123.7–181.0] / 176.3 [173.2–179.4] | 266.1 [174.1–358.1] | 328.7 [297.0–360.5] | 3/4 | together |
| 8K | 3 | 0.27 [0.22–0.33] / 3.35 [0.27–6.42] / 0.32 [0.22–0.42] | 136.1 [114.4–157.9] / 160.1 [156.7–163.5] / 136.1 [116.6–155.7] | 355.9 [246.3–465.6] | 432.3 [394.5–470.2] | 5/6 | together |
| 32K | warm read | 7.31 (4,485 tok/s) | 179.4 | - | - | - | - |
| 32K | 1 | 0.26 [0.26–0.27] | 167.7 [166.3–169.1] | 167.7 [166.3–169.1] | 167.7 [166.3–169.1] | 2/2 | together |
| 32K | 2 | 7.81 [0.32–15.29] / 0.27 [0.25–0.28] | 169.3 [160.4–178.2] / 112.2 [65.6–158.8] | 207.9 [98.6–317.1] | 281.5 [243.8–319.2] | 3/4 | together |
| 32K | 3 | 8.05 [0.37–15.72] / 0.39 [0.35–0.43] / 0.29 [0.29–0.29] | 157.2 [139.0–175.3] / 102.0 [63.8–140.1] / 100.7 [63.6–137.7] | 278.2 [144.3–412.0] | 359.8 [302.7–416.8] | 5/6 | together |
| 64K | warm read | 13.89 (4,719 tok/s) | 160.3 | - | - | - | - |
| 64K | 1 | 0.31 [0.30–0.32] | 162.9 [161.5–164.3] | 162.9 [161.5–164.3] | 162.9 [161.5–164.3] | 2/2 | together |
| 64K | 2 | 11.93 [0.37–23.49] / 0.40 [0.36–0.45] | 157.3 [154.2–160.5] / 109.0 [62.2–155.8] | 188.7 [69.3–308.0] | 266.3 [222.7–309.9] | 3/4 | together |
| 64K | 3 | 12.34 [0.65–24.03] / 0.38 [0.38–0.39] / 0.45 [0.44–0.45] | 148.3 [136.8–159.9] / 96.5 [60.5–132.6] / 96.9 [60.7–133.2] | 249.0 [102.1–395.8] | 341.8 [281.1–402.5] | 5/6 | together |
| 128K | warm read | 28.03 (4,677 tok/s) | 161.3 | - | - | - | - |
| 128K | 1 | 0.43 [0.41–0.45] | 175.4 [173.8–177.0] | 175.4 [173.8–177.0] | 175.4 [173.8–177.0] | 2/2 | together |
| 128K | 2 | 0.48 [0.45–0.51] / 18.91 [0.56–37.25] | 114.6 [63.9–165.3] / 163.8 [159.7–167.9] | 188.8 [47.4–330.2] | 278.4 [223.6–333.2] | 3/4 | together |
| 128K | 3 | 0.57 [0.53–0.60] / 19.30 [0.65–37.95] / 0.62 [0.53–0.70] | 102.0 [61.5–142.4] / 152.8 [144.1–161.5] / 103.5 [61.3–145.7] | 248.1 [70.1–426.1] | 358.3 [284.4–432.2] | 5/6 | together |
| 256K | warm read | 57.83 (4,533 tok/s) | 158.5 | - | - | - | - |
| 256K | 1 | 0.64 [0.63–0.64] | 149.6 [146.9–152.3] | 149.6 [146.9–152.3] | 149.6 [146.9–152.3] | 2/2 | together |
| 256K | 2 | 34.72 [1.07–68.37] / 0.80 [0.79–0.82] | 150.0 [143.1–156.8] / 99.0 [59.6–138.4] | 151.5 [27.6–275.3] | 249.0 [216.4–281.5] | 3/4 | together |
| 256K | 3 | 35.15 [1.13–69.17] / 0.87 [0.83–0.92] / 1.07 [1.00–1.14] | 140.9 [124.1–157.7] / 89.0 [57.4–120.7] / 90.1 [58.4–121.8] | 200.6 [41.0–360.3] | 320.1 [273.5–366.6] | 5/6 | together |

#### Legal, with prefix sharing

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | document in the prefix cache, thinking off, 1,024 out. Mean [min–max], two runs; the warm read is the one cold read that put the document in the cache.

| Prompt | Streams | TTFT s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates | Hits | Overlap |
|---|---:|---|---|---:|---:|---|---|
| 1K | warm read | 0.67 (1,519 tok/s) | 102.3 | - | - | - | - |
| 1K | 1 | 0.21 [0.20–0.21] | 99.0 [98.4–99.7] | 99.0 [98.4–99.7] | 99.0 [98.4–99.7] | 0/2 | together |
| 1K | 2 | 0.26 [0.21–0.32] / 0.56 [0.25–0.87] | 91.7 [89.0–94.4] / 95.6 [93.9–97.3] | 182.8 [178.0–187.7] | 187.3 [186.3–188.4] | 0/4 | together |
| 1K | 3 | 0.30 [0.25–0.34] / 0.56 [0.29–0.83] / 0.27 [0.23–0.31] | 81.2 [79.3–83.2] / 84.2 [82.9–85.5] / 81.1 [79.6–82.6] | 242.5 [237.8–247.3] | 246.5 [244.5–248.6] | 0/6 | together |
| 8K | warm read | 2.52 (3,251 tok/s) | 93.3 | - | - | - | - |
| 8K | 1 | 0.22 [0.22–0.23] | 95.9 [95.3–96.4] | 95.9 [95.3–96.4] | 95.9 [95.3–96.4] | 2/2 | together |
| 8K | 2 | 3.17 [0.24–6.11] / 0.33 [0.25–0.41] | 90.1 [90.1–90.1] / 82.6 [73.8–91.3] | 149.3 [118.8–179.7] | 172.7 [163.9–181.4] | 3/4 | together |
| 8K | 3 | 3.33 [0.28–6.38] / 0.30 [0.27–0.34] / 0.37 [0.34–0.40] | 80.6 [79.4–81.9] / 73.4 [67.2–79.7] / 73.8 [67.5–80.1] | 201.4 [165.0–237.9] | 227.9 [216.6–239.1] | 5/6 | together |
| 32K | warm read | 7.51 (4,363 tok/s) | 93.1 | - | - | - | - |
| 32K | 1 | 0.27 [0.27–0.28] | 98.2 [97.6–98.8] | 98.2 [97.6–98.8] | 98.2 [97.6–98.8] | 2/2 | together |
| 32K | 2 | 0.34 [0.30–0.37] / 7.88 [0.31–15.45] | 71.8 [50.2–93.4] / 91.7 [90.4–93.0] | 131.6 [77.3–185.8] | 163.5 [140.6–186.4] | 3/4 | together |
| 32K | 3 | 8.10 [0.27–15.92] / 0.35 [0.31–0.39] / 0.42 [0.39–0.45] | 83.0 [80.9–85.2] / 64.5 [47.4–81.5] / 64.8 [47.6–81.9] | 176.7 [111.1–242.3] | 212.3 [180.2–244.4] | 5/6 | together |
| 64K | warm read | 14.26 (4,596 tok/s) | 94.7 | - | - | - | - |
| 64K | 1 | 0.33 [0.32–0.35] | 97.1 [96.7–97.4] | 97.1 [96.7–97.4] | 97.1 [96.7–97.4] | 2/2 | together |
| 64K | 2 | 0.42 [0.40–0.44] / 14.33 [0.39–28.26] | 63.6 [34.3–92.8] / 93.1 [92.4–93.8] | 118.7 [52.8–184.7] | 156.7 [128.2–185.2] | 3/4 | together |
| 64K | 3 | 0.50 [0.49–0.51] / 0.41 [0.40–0.43] / 14.69 [0.44–28.94] | 57.1 [33.3–80.8] / 56.7 [33.3–80.1] / 86.0 [80.4–91.6] | 158.7 [77.3–240.1] | 199.8 [158.2–241.4] | 5/6 | together |
| 128K | warm read | 28.74 (4,560 tok/s) | 94.4 | - | - | - | - |
| 128K | 1 | 0.44 [0.44–0.44] | 93.8 [93.3–94.2] | 93.8 [93.3–94.2] | 93.8 [93.3–94.2] | 2/2 | together |
| 128K | 2 | 0.53 [0.49–0.56] / 23.95 [0.57–47.33] | 58.5 [28.7–88.3] / 91.3 [88.9–93.7] | 106.0 [35.5–176.5] | 149.8 [122.4–177.2] | 3/4 | together |
| 128K | 3 | 0.62 [0.59–0.65] / 24.52 [0.73–48.31] / 0.58 [0.52–0.65] | 52.6 [27.9–77.3] / 85.8 [77.7–93.9] / 52.4 [28.0–76.7] | 141.1 [52.4–229.9] | 190.8 [149.8–231.8] | 5/6 | together |
| 256K | warm read | 59.24 (4,425 tok/s) | 92.7 | - | - | - | - |
| 256K | 1 | 0.70 [0.68–0.72] | 90.2 [89.9–90.6] | 90.2 [89.9–90.6] | 90.2 [89.9–90.6] | 2/2 | together |
| 256K | 2 | 0.81 [0.78–0.84] / 39.68 [0.96–78.41] | 56.3 [28.3–84.3] / 89.2 [85.3–93.0] | 95.8 [23.1–168.5] | 145.5 [121.3–169.6] | 3/4 | together |
| 256K | 3 | 1.04 [0.83–1.26] / 0.88 [0.83–0.92] / 40.35 [1.15–79.54] | 50.8 [27.5–74.1] / 49.8 [27.6–72.0] / 83.3 [73.5–93.2] | 124.6 [34.2–214.9] | 183.9 [148.3–219.6] | 5/6 | together |

#### Prose, with prefix sharing

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | document in the prefix cache, thinking off, 1,024 out. Mean [min–max], two runs; the warm read is the one cold read that put the document in the cache.

| Prompt | Streams | TTFT s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates | Hits | Overlap |
|---|---:|---|---|---:|---:|---|---|
| 1K | warm read | 0.64 (1,601 tok/s) | 87.2 | - | - | - | - |
| 1K | 1 | 0.20 [0.20–0.21] | 91.6 [91.1–92.0] | 91.6 [91.1–92.0] | 91.6 [91.1–92.0] | 0/2 | together |
| 1K | 2 | 0.26 [0.21–0.31] / 0.48 [0.25–0.71] | 85.4 [83.2–87.5] / 85.0 [82.8–87.2] | 166.6 [159.1–174.1] | 170.3 [166.0–174.7] | 0/4 | together |
| 1K | 3 | 0.64 [0.49–0.79] / 0.22 [0.21–0.24] / 0.36 [0.29–0.43] | 75.1 [73.4–76.8] / 74.5 [73.5–75.5] / 75.1 [73.8–76.5] | 218.7 [211.8–225.7] | 224.8 [220.7–228.9] | 0/6 | together |
| 8K | warm read | 2.49 (3,284 tok/s) | 90.3 | - | - | - | - |
| 8K | 1 | 0.21 [0.21–0.22] | 93.5 [93.1–93.9] | 93.5 [93.1–93.9] | 93.5 [93.1–93.9] | 2/2 | together |
| 8K | 2 | 3.20 [0.30–6.09] / 0.26 [0.24–0.27] | 88.1 [86.9–89.3] / 80.7 [72.6–88.9] | 147.0 [116.3–177.6] | 168.8 [159.5–178.2] | 3/4 | together |
| 8K | 3 | 3.33 [0.29–6.36] / 0.31 [0.27–0.34] / 0.29 [0.23–0.35] | 78.7 [78.0–79.5] / 72.1 [66.1–78.2] / 72.0 [66.4–77.6] | 197.3 [161.9–232.6] | 222.9 [211.9–233.8] | 5/6 | together |
| 32K | warm read | 7.39 (4,435 tok/s) | 95.5 | - | - | - | - |
| 32K | 1 | 0.27 [0.27–0.27] | 86.5 [86.1–87.0] | 86.5 [86.1–87.0] | 86.5 [86.1–87.0] | 2/2 | together |
| 32K | 2 | 7.84 [0.31–15.38] / 0.33 [0.29–0.37] | 87.4 [82.3–92.6] / 64.8 [47.0–82.6] | 121.3 [78.3–164.4] | 152.2 [139.6–164.9] | 3/4 | together |
| 32K | 3 | 0.38 [0.36–0.40] / 0.31 [0.30–0.33] / 8.14 [0.47–15.82] | 58.3 [44.4–72.1] / 58.1 [44.3–71.8] / 79.1 [72.4–85.7] | 163.4 [111.7–215.1] | 195.4 [174.4–216.4] | 5/6 | together |
| 64K | warm read | 14.11 (4,645 tok/s) | 84.9 | - | - | - | - |
| 64K | 1 | 0.34 [0.33–0.34] | 89.1 [88.7–89.6] | 89.1 [88.7–89.6] | 89.1 [88.7–89.6] | 2/2 | together |
| 64K | 2 | 14.24 [0.35–28.12] / 0.40 [0.39–0.42] | 84.4 [83.9–85.0] / 59.4 [33.5–85.4] | 110.5 [51.2–169.8] | 143.8 [117.3–170.4] | 3/4 | together |
| 64K | 3 | 0.49 [0.45–0.54] / 0.40 [0.39–0.42] / 14.62 [0.45–28.79] | 53.7 [32.4–75.0] / 53.4 [32.4–74.3] / 78.0 [74.5–81.5] | 148.8 [74.9–222.7] | 185.0 [146.3–223.8] | 5/6 | together |
| 128K | warm read | 28.49 (4,601 tok/s) | 87.6 | - | - | - | - |
| 128K | 1 | 0.44 [0.42–0.46] | 89.0 [88.6–89.4] | 89.0 [88.6–89.4] | 89.0 [88.6–89.4] | 2/2 | together |
| 128K | 2 | 0.56 [0.51–0.60] / 24.38 [0.51–48.26] | 55.7 [26.6–84.9] / 85.8 [84.1–87.5] | 101.3 [34.4–168.2] | 141.6 [114.1–169.0] | 3/4 | together |
| 128K | 3 | 0.65 [0.57–0.74] / 24.96 [0.63–49.28] / 0.59 [0.57–0.62] | 50.2 [25.9–74.4] / 80.8 [74.0–87.5] / 49.9 [26.0–73.8] | 135.8 [50.8–220.8] | 180.8 [139.4–222.3] | 5/6 | together |
| 256K | warm read | 58.87 (4,453 tok/s) | 85.6 | - | - | - | - |
| 256K | 1 | 0.69 [0.65–0.73] | 87.0 [86.5–87.5] | 87.0 [86.5–87.5] | 87.0 [86.5–87.5] | 2/2 | together |
| 256K | 2 | 39.60 [0.97–78.24] / 0.86 [0.85–0.87] | 83.7 [82.1–85.3] / 54.8 [28.1–81.5] | 92.7 [22.9–162.6] | 138.5 [113.5–163.5] | 3/4 | together |
| 256K | 3 | 0.93 [0.92–0.94] / 0.88 [0.87–0.90] / 40.15 [1.07–79.24] | 49.4 [27.5–71.4] / 49.3 [27.4–71.1] / 78.6 [71.9–85.3] | 123.4 [34.0–212.9] | 177.3 [140.1–214.5] | 5/6 | together |

### Long chats and turns

| What | Figure | Window |
|---|---|---|
| A turn on a chat of about 128,600 tokens, alone, all prefix hits (server prefill / client first token, median of two), 200 W | +880 new tokens 0.746 / 1.243 s; +2,048: 1.744 / 2.230 s; +4,200: 3.307 / 3.752 s; +10,000: 4.724 / 5.131 s | p1 |
| A turn of 205 tokens beside another chat's 250,000-token read, 200 W | first token 4.29 s, whole turn 14.15 s (alone 1.18 s and 1.72 s) | p1 |
| Four documents of 110,000 to 140,000 tokens read at once | first token after 129 to 166 s each | s4 |
| A turn while four such chats are open | under 2,048 new tokens: median 2.18 to 4.12 s; larger turns 9.33 to 10.67 s (four boots) | s4, st3b, st4, sv1 |
| A turn on a long chat after another chat was read | 2.1 and 2.6 s at three slots; 46.7 s at two (its slot was taken and 138,824 tokens read again) | o2 |
| A 131,466-token turn restored from the conversation store after a restart | first token after 1.23 s (43.84 s read cold) | st4 |

### Where the time goes (window t1, 200 W, one run)

From the engine's own timers and the torch profiler ([research/TWELVE-CARD-TIME-BUDGET.md](research/TWELVE-CARD-TIME-BUDGET.md)):

| What | Stage 0 | Stage 1 | Stage 2 |
|---|---|---|---|
| One 4,092-row prompt chunk, median forward span | 793 ms | 891 ms (p90 1,310) | 844 ms |
| Experts, shared layer, router and their sums | 318 ms | 334 ms | 271 ms |
| Attention | 119 ms | 130 ms | 112 ms |
| The index (grows with context) | 53 ms | 142 ms | 222 ms |

A one-request decode round is 26.25 ms in the traces: experts 29 %, dense multiplies 21 %, the drafter 10 %, 14 %
outside any kernel. It grows to 28.1 ms at 128,000 tokens and 31.0 ms at 249,000; the growth is the index. The
cards idle inside every prompt chunk step while the host frees and re-allocates card memory, and the server's front
end costs about a third of a small turn's wait on a 128,000-token chat.

### Memory (card total 32,510 MiB; the line 31,998 MiB)

| What | Figure | Window |
|---|---|---|
| After warm-up, by stage, four slots | 28,795 / 29,013 / 27,495 MiB | s4 |
| Highest card sample with four grown chats at once | 30,398 MiB (29,838 / 30,398 / 28,892); 30,364 to 30,446 MiB on the later boots that measured it | s4 to p1 |
| Room under the line on the fullest (middle) stage, four slots | 1,600 MiB | s4 |
| The conversation store switched on, four grown chats | 29,802 / 30,396 / 28,894 MiB against 29,836 / 30,398 / 28,896 with it off (within the sampler's resolution) | st4 |
| Card limit | on all twelve workers; no token and no memory mark changed | k1 and every boot since |
| Highest card sample, kit 9N with the nine lines, four-chat rehearsal (30 requests, largest prompt about 153,600 tokens) | 30,814 MiB; the one-slot boots without the drafter 30,020 | p5u, p6b, q1 |
| Highest card sample, kit 9S with the ten lines, the same rehearsal | 30,832 MiB (30,594 / 30,832 / 28,328 by stage); over the morning's bench and prefix measure, 2 h 15 min, 30,418 | c2a, c3b, c7a, morning bench |
| Highest card sample, kit 9S with the eleven lines, the same rehearsal | 30,832 MiB (30,594 / 30,832 / 28,328 by stage); over the afternoon's bench and prefix measure, 2 h 15 min, 30,418 | b4a, b3a, bench 14:09Z |

### Closeness to the reference

The experts themselves are NVIDIA's NVFP4 pack, whose values equal the official MXFP4 experts (UPSTREAM: NVIDIA's
model card; checked on sampled layers in our earlier work). The reference below is an all-native reference computed
with experts fetched from host memory: every expert native, on two stages.

| What | Figure | Window |
|---|---|---|
| Short prompts: median first divergence from the reference | 208 tokens | w3, w5, o1, o2 |
| Short prompts: median of the mean absolute log-probability difference | 0.0090 | same |
| Seven cold long prompts against the accepted capture | identical 7 of 7 on every boot that ran it since k1 (re-checked with the strict judge on s4 and st4) | k1 to p1 |
| Copy tasks, cold, 4,000 tokens, one request each | 584 of 584, 467 of 467, 536 of 534 words | o2 |
| Seven cold long prompts against the capture of the turn-chunks boot (`quality-TC3.json`, another rounding of the same arithmetic) | identical 7 of 7 on every boot held to it (p5t to q1); the streamed text of the tensor-core scorer equal to the reference boot's 22 of 22 | p5t to p6b |
| Two boots without the drafter, eight forced cells (4.2K to 252K tokens, read and turn), temperature zero | text equal 8 of 8 | q1 |
| The batched projection's boot (kit 9S) against kit 9N's accepted capture: another rounding | identical 1 of 7, median first divergence 58 tokens, median mean \|dlogprob\| 0.0354 nats (the line: at least 25 and at most 0.0614) | c2a |
| Seven cold long prompts against the capture of that boot (`quality-TC3-WOA4K.json`) | identical 7 of 7 on every boot held to it (the reference and candidate boots of c3b and c7a, the reference boot of b4a) | c3b, c7a, b4a |
| The split count's boot (the eleven lines) against the ten-line capture: another rounding | identical 1 of 7, median first divergence 45 to 48 tokens, median mean \|dlogprob\| 0.0350 to 0.0366 nats (the line: at least 25 and at most 0.0614); CLEAN | b4a |
| Seven cold long prompts against the capture of that boot (`quality-TC3-WOA4K-S1.json`) | identical 7 of 7 on the reference boot of b3a (kit 9V with its switch unset, today's arithmetic) | b3a |
| The exact four-card sum's boot (kit 9V, `VLLM_V41_TP_EXACT_SUM=1`) against `quality-TC3-WOA4K-S1.json` | identical 0 of 7, median first divergence 13 tokens (at least 25 asked), median mean \|dlogprob\| 0.0348; CLEAN: FAILED its line | b3a |
| A cache hit against its own cold read (nine bench prompts, 32K to 128K, greedy, 32 tokens) | identical 4 of 9, median first divergence 25, median mean \|dlogprob\| 0.0275 | b3a |

These figures say how closely the build tracks the all-native model. They do not by themselves say that its answers
are better for a given kind of work.

## What worked

- **A third pipeline stage on the fast engine.** With a quantized expert mix the three-stage engine reads CLEAN
  against a two-stage reference boot of the same experts (0 of 12 early divergences) and passes the alone-against-together check (window w3).
- **The scheduler's spacing rule.** Under synchronous scheduling a request's next decode step now waits one step per
  pipeline stage. Without it the first decode round after every prompt was computed on another request's tokens
  (window w1: 15 of 15 answers differed). Found by running the real engine loop on a CPU.
- **Every routed expert on the cards.** 15,360 hot and 0 cold on every rank of every all-NVFP4 boot since window w3.
- **The middle stage's memory.** Kits 5c and 6c take the embedding off the middle stage and apply layer 14's
  Engram on the first stage: the two-slot peak fell from 31,808 to 31,336 MiB (windows w3 and w5).
- **Three, then four slots.** 4,092-row chunks halve a prompt read's working memory on the middle stage, which made
  room for a third slot (window o2) and a fourth (window s4): the fourth changed no token and cost about 345 MiB on
  the fullest card.
- **The card limit.** A fuse at the owner's line on every stage; it changed no token and no memory mark (window k1).
- **The conversation store on three stages.** Every server test of the store passed in window st4: the state check
  on twelve workers, a 130,000-token chat restored from disk after a restart with its state equal to the saved state
  and an identical answer, 0 waits over 2,976 saves.
- **The cause of a second rounding, found by reading and proven on one card in 25 seconds.** The dense multiply
  library picks its split count from a chunk's row count; a probe with no model confirmed it at all sixteen row counts
  a CPU model had predicted ([docs/measurements.md](docs/measurements.md), the split-count probe).
- **Serving without checking on the public port.** The full checks run on a test port and write a record; the
  serving boot sends no request and needs a matching record. A move or a return takes about 3 minutes without a server.
- **200 W.** The prompt read rose from 3,284 to 3,733 tok/s; decode moved by under 2 % (one run). 200 W tripped the
  host on 2026-11-18 under four long chats; the build runs at 175 W since (below).
- **The release setting** (`VLLM_V41_EMPTY_CACHE_TOKENS=8192`, window p1, serving since p2s). The engine released
  its cached card memory at every prompt step of 2,048 rows or more; at 4,092-row chunks that was every step. Now
  only steps above 8,192 rows release it: reads +12.6 % at 61K and +9.9 % at 93K (200 W), identical answers, no
  higher memory peak. The first switch adopted through the serving record.
- **The middle-stage prompt skip** (`VLLM_V41_PP_MID_PREFILL_SKIP=1`, window p2). The middle stage no longer
  computes its late layers on prompt chunks the skip rule allows: reads +10.4 % at 61K to +49.6 % at 426K alone,
  and with the release setting +19.7 % to +53.2 % (4,223 tok/s at 425,776 tokens against 2,757; 200 W). Predicted
  +3 to +8 %: the prediction was low by half and more, recorded as wrong.
- **The decode loop's idle gaps closed** (`VLLM_V41_PP_EMPTY_AHEAD=1`, `VLLM_V41_PP_ANSWER_NO_SLEEP=1`, window p3c).
  The engine core's own counters, printed every 64 answers, showed what the time budget had read in the traces:
  after a normal one-step turn the loop made no empty step in 0 of 64 answers, slept twice and blocked on the
  drafter once per round, 2.83 ms from an answer to the next submission. With the spacing step submitted ahead the
  round shortened by 1.16 ms on a short chat and 1.18 ms at 128K (-4.6 / -4.2 %), the counts 64 of 64. The
  drafter's faster GEMV (`VLLM_V41_DRAFT_GEMV2=1`) rode in the same window with no loss (-1.28 / -1.34 ms a round
  against `off` with all three).
- **The prompt-end hold** (`VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD=128`, window p5). A short turn beside another chat's
  248,000-token read: the whole turn 5.12 s against 11.99 (-57 %), the read's first token not slower.
- **The token cache** (`VLLM_V41_PREFIX_TOKEN_CACHE=1`, window p4b). The API server keeps the last eight prompts'
  text and ids and tokenizes only what a chat added: the front of a turn on a 128,000-token chat 47.2 ms against
  308.0 (median of 20); replies equal in text 27 of 27; its own counts exact at the 100th call. An independent
  review had found the first version returned wrong ids after an edit at its cut; the repaired cut rule was proven by
  two fuzzes and then by three edits on the cards.
- **The turn chunks** (`VLLM_V41_PP_TURN_CHUNKS=3`, windows p5 and p5u). A turn's last rows read as a head and three
  1,024-row chunks so the stages overlap: first token of a +4,200 turn on a 128,000-token chat 2.319 s against
  3.687; +10,000 3.623 against 4.963. Another rounding of the same arithmetic, so the build's accepted capture
  became `accepted/quality-TC3.json`, decided in one place from the adopted lines.
- **The sparse-attention index scorer on the tensor cores** (`VLLM_V41_INDEX_TC=1`, kit 9N, windows p6 and p6b).
  Designed from one captured decode step's index operands (window p3l), bit-exact by its own guards: identical 7
  of 7, the streamed text equal 22 of 22; the round at about 128K 0.80 to 1.25 ms shorter (-3.0 to -4.6 %) in two
  windows; short chats within 0.07 ms; four together +6.7 %; the scorer's counters equal to the drafter's full-width
  verification steps on every worker (6,641 = 6,641).
- **The batched weight-only projection at 4,092 rows** (`VLLM_V41_WO_A_BATCHED_4K=1`, kit 9S, window c2a). The
  attention output projection of a prompt chunk in one batched launch, the tenth adopted line: cold reads 4,663 /
  4,741 / 4,531 / 4,309 tok/s for 60,705 / 93,025 / 247,948 / 425,779 tokens, +4.2 / +4.0 / +4.0 / +7.0 % against the
  same boot with it off (first token 13.02 / 19.62 / 54.72 / 98.81 s); another rounding of the same arithmetic
  (identical 1 of 7 against kit 9N's capture, median first divergence 58 tokens, median mean |dlogprob| 0.0354 nats,
  within the lines), so the accepted capture became `quality-TC3-WOA4K.json`; peak 30,832 MiB, no out-of-memory line.
  Predicted about 2.7 %: wrong, low. In the morning's bench its cold code reads were 4.8 to 5.0 % faster at 32K and longer.
- **The split count** (`TM_GEMM_TUNE=max_splits=1`, the multiply library's own switch, window b4a; the eleventh adopted
  line). The dense multiplies' library takes one split of the inner dimension at every row count instead of choosing
  by row count, so a prompt's last chunk no longer rounds by its size. Against the ten lines on the same boot kind:
  cold reads of 60,706 / 93,028 / 247,953 / 425,783 tokens 4,659 / 4,756 / 4,552 / 4,314 tok/s (-0.1 / +0.2 / +0.8 /
  -0.3 %), one fresh request 137.3 against 135.6 tok/s (+1.3 %); another rounding (identical 1 of 7 against the ten-line
  capture, median first divergence 45 to 48 tokens, median mean |dlogprob| 0.0350 to 0.0366 nats, CLEAN, within the
  lines), so the accepted capture became `quality-TC3-WOA4K-S1.json`; peak 30,832 MiB. Its earlier arm (windows g1 and
  t1) had ended INCONCLUSIVE at a short-prompt line against the all-native reference; the window of 2026-11-19 judged
  it against the build's own capture by the closeness rule every rounding change since has met.
- **The decode round is repeatable boot to boot** within 0.6 % (windows p3b and p3c), which is what makes a 0.5 ms
  line decidable; the reference's round at 128K moved by about 0.4 ms between p6 and p6b, the A/A band for that
  cell.
- **Two boots without the drafter give the same text** (window q1): eight forced cells at 4.2K to 252K tokens and
  after a turn, temperature zero, equal 8 of 8.
- **A kit's new log lines are collected before a window judges by them** (window p3l): one boot with every switch
  on, nothing judged, whose product is the real log; every later pattern met its real line first.
- **The bench, twice on 2026-11-19** (the afternoon's tables above; the morning's on the ten-line set kept beside
  them): one stream decodes 153 to 184 tok/s on code and 83 to 102 on legal and prose from 1K to 256K of prompt;
  prefill 4,514 to 4,698 tok/s at 64K and longer on code, 4,420 to 4,611 on legal and prose; the first token of a 256K
  prompt after 58 to 59 s; three streams together 2.2 to 2.5 times one stream's decode once every prompt is in; on a
  cached document the first token in 0.43 to 0.44 s at 128K and three streams at 427 / 230 / 221 tok/s (code / legal /
  prose) over their window. The first and the second repeat of every cold read agree within 0.15 s.

## What did not work or is not yet good

- **The first three-stage boot answered wrongly** (window w1). The diagnostic window w2 ran with the fault still in
  and ended with Xid 43 on the four cards of the first stage. Fixed by the spacing rule (window w3).
- **Two kits passed CPU tests that did not go through the server's real path.** Kit 4's six-slot arm died on a label
  table sized for four slots (window s1); kit 5b's first stage refused to boot because the GGUF loader dropped the
  moved Engram's tensors (window w4). Both were fixed and now have tests through the real entry point.
- **Five arms were stopped by the build's own checks, each meeting a real log of its kind for the first time:** the
  out-of-memory pattern that matched the card limit's own announcement (k1: a second boot), the rank count that met
  the store kit's digest lines (st3), arm G's environment witness, which read an environment a titled worker blanks
  (g1), the turn probe, which replayed answers without reasoning and so read four turns cold (t1), and the kit-8 witness,
  which failed the middle-stage skip on vLLM's warning about an unregistered environment variable (p1). Each is
  fixed and tested on the real log.
- **The equality judge accepted incomplete answers.** It compared two answers through the shorter one, so a cut
  answer read as identical. An independent review found it; the judges were made strict (equal length, equal ids, a
  missing key INCONCLUSIVE), and the three historical identity results were re-checked on whole sequences and stand.
- **The token cache returned wrong token ids** on a hit when an edit removed a special token at its cut (found by
  the same review before it ever served). It was held, repaired and re-proven by two independent fuzzes (one of
  3,200 edits, 0 sequences different from a full encode), then by three edits on the cards (window p4b), and serves.
- **A differently cut prompt read rounds differently, from two sources.** Large four-card sums go through NCCL in
  half precision in an order that depends on the buffer's size; and the dense library's split count depends on a
  chunk's row count. Both are the engine's general arithmetic; no token is shown wrong by either. The
  library's switch that fixes the split count (`TM_GEMM_TUNE=max_splits=1`) ended INCONCLUSIVE in its first arm
  (window t1) and was adopted in window b4a by the closeness rule (What worked). The exact four-card sum (kit 9V,
  `VLLM_V41_TP_EXACT_SUM=1`: each rank's part sent to every rank and summed in one fixed order in FP32) ran in window
  b3a and failed its closeness line: identical 0 of 7 against the eleven-line capture, median first divergence 13
  tokens where 25 were asked; its cost on the board, measured with no model, is 2.2 times the library's sum on a large
  call (1.180 against 2.632 ms). The exact order is a different rounding from the one the build's captures were taken
  in, so the rule's closeness line, written for small moves, failed it; the window's prediction had named the speed
  line as the one that fails: wrong on the deciding line. Closed; the switch stays default-off in kit 9V.
- **A cached or restored turn is not token-identical to a cold read of the same text** (window st4). The rule line
  that asked for it was the wrong line and stays FAILED as written.
- **A prefix-cache hit is not token-identical to its own cold read either** (window b3a, on the eleven lines): nine
  prompts of the bench at 32K to 128K, read cold and then resent, greedy, 32 tokens: identical 4 of 9 (prose 3 of 3,
  code at 64K), median first divergence 25 tokens, median mean |dlogprob| 0.0275; every resend a hit by timing (0.25 to
  0.45 s against 7.2 to 28.6 s cold). The hit's new rows are read in a smaller chunk than the cold read's, so the
  four-card sums round in another order (the first source above). The server sends no cached-token count on these
  requests, so hits are known by timing (In progress).
- **A turn that leaves its chat's prompt at the last token is read cold** after an answer longer than the ring's
  reach: 36 to 44 s instead of about a second on a 130,000-token chat. Clients that replay the reasoning with each
  answer do not meet it.
- **Ideas measured and closed.** Releasing the allocator's cache at 512 rows (same peak, slower first token);
  awaiting each prompt hand-off (kit 7: no memory gained). Several predictions were wrong and are recorded as wrong.
- **No serving run of hours under a rule** at four slots. At two slots the serving arm crept 58 MiB in half an hour.
- **More requests at once than stages gain little.** A step carries one request.
- **One request does not get faster from a third stage:** its round carries the same forty layers of work.
- **Answer quality in use is not established** by these measurements.
- **Six and eight slots:** no verdict.
- **Six stops by a check that had not met a real log line** (windows p1, p3, p4 among them): a warning line the API
  server prints for every unregistered switch, a worker line the engine never prints, an option parser that refused
  the value 0. Each cost a window and was fixed on the real log; since window p3l a kit's lines are collected first.
- **Two-sided no-harm lines** failed a boot for being 2.02 % FASTER (window p3b), and a one-in-twenty slow turn
  would have withdrawn the token cache: the lines are one-sided now, and a gain line tolerates one slow cell in ten.
  The thresholds did not move.
- **A judge whose units did not match** failed the tensor-core scorer on its first window (p6): scorer counter steps
  against engine-core "answers", which count prompt chunks too. An independent reviewer read the logs; the corrected
  line compares the counters with the drafter's own verification count and carries hand-computed tests from the real
  lines. The window was rerun, not re-judged.
- **A witness that waits for a one-time line under four-chat traffic** passed by chance in p4b and failed by chance
  in p5t: the first-use line of the empty-ahead switch needs a lone one-step request, which the check now sends.
- **The power trip at 200 W** (2026-11-18 11:15:48Z): the host ended without a shutdown sequence under the heaviest
  load of the day (four long chats with the loop switches on). Caps to 150 W by the owner's rule, then 175 W and a
  1,452 MHz ceiling. Long prompts read about a third slower at 150 W than at 200 W; the figures since are 175 W
  figures. The restart tool of the rule could not have started the build (it required kit 6c): to be fixed to serve
  what the record names.
- **Predictions wrong on the generous side and the stingy side.** The middle-stage skip gave twice its predicted
  gain; the tensor-core scorer's second-window gain was 0.80 to 0.96 ms against "about 1.2"; a 420,000-token read
  was faster than predicted twice; the closing bench's code decode at 128K was 161 tok/s against a predicted 133, and
  its legal and prose prefill fell under a band drawn from code cells; the batched projection gave 4.0 to 7.0 % where
  about 2.7 was predicted; the packed experts' rounds were longer where shorter was predicted; the morning bench's
  decode cells moved up to 8.6 % with the generated text where "within the band" was predicted; the split count's
  repeated and four-together decode cells moved +4.7 and +4.3 % where under 2 % either way was predicted (the text
  moved); the exact four-card sum was predicted to fail on speed and failed on closeness first. Each stays recorded
  as written.
- **Several streams do not read their prompts together.** With two or three requests released at once on prompts of
  8K and longer the reads interleave: three cold 128K code requests got their first tokens after 77.4, 90.5 and
  90.5 s where one alone takes 28.07 s, and a read that shares the cards with a decoding stream runs at roughly half
  its lone rate (all three benches; the cause is not established). They decode together once every prompt is in.
- **The first cold read of a legal or prose document was slower than its second repeat**, by 1 to 11 s at 32K and
  longer, in the benches of 2026-11-18 and the morning of 2026-11-19 (code's repeats agreed within 1.1 s); the
  morning's legal and prose one-stream first-token cells are the mean of the two. On the eleven-line set it did not
  recur: every one-stream cell's two repeats agree within 0.15 s. HYPOTHESIS: the library's split search on a shape
  not yet seen, removed by the fixed split count; not established by a boot built for it.
- **The conversation store failed its timing line** (windows st5 to st7, 2026-11-19). Every state check passed (the
  dump, a restart on the saved set, exact answers after the restart, memory equal to the store-off boot, 0 waits on
  the save path), but the first token of the larger turns in the four-chat rehearsal was 4.718 and 3.722 s with the
  store on against 3.596 and 3.590 s off (limit 3.696); the small turns straddled their limit (INCONCLUSIVE). The
  overlay was rolled back; the store stays off.
- **The lean pipeline hand-off** (item C3; kits 9P, 9Q and 9T). Its first window (c3a) did not boot: the kit assumed
  a hidden size of 4,096 where the model's is 5,120, and its CPU test had taken the number from inside the test. Cut
  again with the geometry read from the model (kit 9Q, then 9T over 9S), it ran in window c3b and failed its line:
  the round at 128K after a cold read 0.133 ms shorter where 0.3 was asked, the turn cell unreadable against two
  boots whose spreads overlapped. Closed.
- **The packed expert records** (item C7; kits 9R and 9U, window c7a). The packs re-cut without their zero padding
  answered the accepted prompts token for token and used 756 / 878 / 878 MiB less a card by stage, but the decode
  rounds were longer, not shorter (+0.095 and +0.155 ms at 128K; +0.27 and +0.90 % on short chats); the prediction
  (0.33 to 0.41 ms shorter) was wrong in sign. Closed.
- **Two kernel items closed at their timing gate without a window.** The native expert layer in one launch (C4) was
  sized on CPU against the engine's own fused kernels: its best case saves 0.205 ms a round, under the 0.300 ms gate.
  The gaps between prompt chunks (C1) measure under 0.37 % of a chunk, under the 2 % gate.
- **The Engram precision item is still open.** The quality window could not decide it (no verified local extraction
  of the official host-memory FP8 Engram table, revision 1bae0e10, exists in the trees): INCONCLUSIVE by
  construction, not a pass.

## In progress

- **Nothing of the plan's optimization list is being built.** Every item has ended: the batched projection and the
  split count adopted (windows c2a and b4a); the conversation store, the lean pipeline hand-off, the packed expert
  records and the exact four-card sum failed their lines (windows st7, c3b, c7a and b3a); the one-launch expert layer
  and the prompt-chunk gaps closed at their timing gates without a window. Their kits are in `kit/` as default-off
  switches (9Q, 9T, 9R, 9U, 9V); the store's kit stays private.
- **A cached-token count in the server's usage.** A hit is known by timing only: the engine's usage block carries no
  count of the prompt tokens the cache supplied (0 of 216 requests of the prefix measure sent one). The build's
  users want the count to tell a hit from a re-read without a stopwatch.
- **Partial-prefix reuse.** In the prefix measure a warm request whose first 113,392 tokens were in the pool read
  its whole prompt cold; whether the engine can start a read from a shared prefix of another request's pages, and
  under what rule, is being investigated before any change.
- **Batched decode on the pipeline** (tier B of the kernel plan): the next window, about 36 hours out when the cards
  are idle for it; its rule carries the exactness question (whether batched rows round as single ones do) written
  before its data.
- **The Engram table's precision.** The build loads a 4-bit table of 103.0 GiB; the official one is 188.8 GiB and
  fits in host memory. INCONCLUSIVE in window q1 by construction; a reader for the official rows is needed first.
- **The restart after a power trip** serving what the record names.

## Next

The order of the remaining work: the cached-token count in usage, then the findings of the build's users in order
(the partial-prefix reuse question first), each handled as a window or a kit with a rule written before its data;
batched decode on the pipeline (tier B of
[documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md](documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md)) in about 36 hours
when the cards are idle for it; the Engram reader and its quality window; then the rest of tier B as ranked (the
snapshot hit tied to its own read; then tier D), each window ending adopted or failed, one change per boot; then the
bench again and this page updated again. The split count and the exact large sums are decided (windows b4a and b3a).
Power and clocks stay the owner's: the figures of the day are 175 W figures; 200 W tripped the host once under four
long chats.

## Hardware

- A twelve-V100-SXM2 server: three NVLink boards of four 32 GB cards (32,510 MiB each). Tensor parallelism stays
  inside one board; the pipeline hand-off between boards goes over PCIe. Two boards sit behind one PEX88096 switch,
  the third behind a PLX 8749. The first hand-off stays inside the PEX switch; the second crosses host bridges.
- Volta has FP16 `mma.sync.m8n8k4` only: no FP4 or FP8 tensor cores. Every 4-bit format is decoded in software into
  FP16 operands, so low-bit experts pay by reading fewer bytes, not by faster math.
- 150 W and 1,477 MHz per card until 2026-11-17 18:58Z; 200 W to 2026-11-18 11:15:48Z, when the host tripped under
  four long chats; 150 W from 11:45Z; 175 W and a 1,452 MHz clock ceiling (cards at 1,447 MHz) since 12:09:23Z.

## Running it

[docs/OPERATIONS.md](docs/OPERATIONS.md) says how the build is installed, started, checked and stopped. The window
driver and launcher that ran our windows are not in this release (they name our host's layout); the
operations page describes what they do so that the steps can be followed with your own scripts.

## Repository layout

| Path | What it is |
|---|---|
| `README.md`, `CHANGELOG.md` | this page; what changed, by date |
| `NOT-YET-PUBLISHED.md` | what this release leaves out, and why |
| `docs/` | [architecture](docs/ARCHITECTURE.md), [operations](docs/OPERATIONS.md), [measurements](docs/measurements.md) (the [morning bench of 2026-11-19](docs/measurements-2026-11-19-morning.md) and the [2026-11-18 bench](docs/measurements-2026-11-18.md) beside it), [history](docs/history.md), [upstream contributions](docs/upstream-contributions.md); [index](docs/README.md) |
| `documents/plans/` | the [kernel and optimization plan](documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md) |
| `documents/reviews/` | the [upstream review](documents/reviews/UPSTREAM-REVIEW-20261117.md), the [chunk-boundary reading](documents/reviews/CHUNK-BOUNDARY-READING.md) |
| `research/` | this build's own findings ([TWELVE-CARD-RESEARCH.md](research/TWELVE-CARD-RESEARCH.md)) and its [time budget](research/TWELVE-CARD-TIME-BUDGET.md) |
| `engine/` | [README.md](engine/README.md): every kit with its engine commit, what it is and its state |
| `kit/` | the engine changes as install kits (`apply.sh`, `rollback.sh`, `MANIFEST`, `files/`). `PP3KIT9S-over-cut` is the serving kit; `9Q`, `9T`, `9R`, `9U` and `9V` are the kits of the three failed windows, default-off switches over 9S |
| `tools/` | judges and probes: [README.md](tools/README.md) |
| `tests/` | CPU tests of the tools on real log lines (`tests/data/`); `tests/models/` runs inside the engine tree |
| `LICENSE`, `LICENSES/`, `NOTICE`, `CREDITS.md`, `CONTRIBUTING.md`, `SECURITY.md`, `Makefile` | licences, third-party notices, credits, how to contribute, security, the CPU checks |

The Volta architecture and kernel research these builds stand on is published separately in the research hub,
[volta-v100-research](https://github.com/ProprietaryLegal/volta-v100-research).

No model weights, packs, compiled libraries, tier maps, routing tallies, prompts, generated text or logs are in this
repository.

## Related builds

Other PLI Labs builds of DeepSeek-V4.1-Flash-1017 on V100 cards:

- [Volta Max uPLIft](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4): eight cards, two stages.
- [Volta Full NVFP4](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100): eight cards, two stages.
- [Volta sPLIt](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4): eight cards, two stages; the engine lineage this build descends from.

## Work with PLI Labs

[PLI Labs](https://proprietarylegal.ai/labs.html) is the open research arm of Proprietary Legal Intelligence. PLI builds
AI around an organization's own work product: retrieval over its documents, staged workflows, validation of every output
and attorney review before anything goes out. This stack is how we run that privately, on hardware we own. If you want
private, on-prem LLM serving of your own, from choosing the hardware to the measured stack and the workflows on top, we
consult on exactly that: [proprietarylegal.ai](https://proprietarylegal.ai).

## Licence

Original documentation and code are Apache-2.0 ([LICENSE](LICENSE)); see [NOTICE](NOTICE). The kit files are whole
engine files of 1Cat-vLLM and vLLM with our changes and keep that code's licence. No model weights, packs, wheels,
compiled libraries, routing tallies or prompt corpora are distributed here.

## Credits

This build stands on other people's work: DeepSeek (the model, the DSpark drafter, the Engram design), NVIDIA (the
NVFP4 pack), 1CatAI and the authors of 1Cat-vLLM's SM70 work (yangzhuxinyzx's #1277 package, Peuqui's #742 backend
and #1362), Dennis Vertzyas ([v100-skinny](https://github.com/dnv2003/v100-skinny)), ISTA-DASLab (the GSQ-RCO GGUF),
ggml, OpenMMLab (the TurboMind multiply library inside the engine), the vLLM project, and PLI Labs' eight-card
builds. Every source, with what we took from it, is in [CREDITS.md](CREDITS.md).
