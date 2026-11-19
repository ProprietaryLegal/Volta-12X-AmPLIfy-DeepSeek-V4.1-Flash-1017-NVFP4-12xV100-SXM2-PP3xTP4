# Volta 12X AmPLIfy

DeepSeek-V4.1-Flash-1017 served on twelve Tesla V100-SXM2 32 GB cards from 2018, with every routed expert at
NVIDIA's NVFP4 precision on the cards. The model runs as three pipeline stages of four tensor-parallel cards. All
15,360 routed experts are held in card memory: none is compressed and none is read from host memory.

This is the build's record as of the morning of 2026-11-19 (its third release): the engine changes as install kits
through kit 9U, the measurements with their conditions, the final bench of the serving set every optimization left
(code, legal and prose prompts of 1K to 256K tokens, one to three streams, cold and with the document already in the
cache), the architecture, the history of every test window, the kernel
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

**175 tokens a second** on a code request with 128,000 tokens of context, a 128,000-token file read in **28 seconds**
(4,657 tok/s), **four conversations** kept warm with **524,288 tokens** of context each. MEASURED 2026-11-19 on the
serving set (kit 9S, ten adopted switches), 175 W per card, 1,452 MHz clock cap, four slots, cold reads, thinking off,
1,024 tokens out, mean of two runs. Prose figures, min-max and every stream are in [Measured](#measured). A one-stream
decode figure is the figure of the text the bench generated in that cell: across texts of one kind it spreads about
±9 %, and two repeats of one text agree within 2 %.

**Code** (a source file as the prompt)

| Context (prompt tokens) | Decode tok/s, one request | Prompt read tok/s | First token, cold | Three requests together, tok/s summed | First token, document already cached |
|---|---:|---:|---:|---:|---:|
| **1K** | 178 | 1,558 | 0.7 s | 428 | 0.2 s* |
| **8K** | 175 | 3,244 | 2.5 s | 420 | 0.2 s |
| **32K** | 174 | 4,380 | 7.5 s | 400 | 0.3 s |
| **64K** | 160 | 4,636 | 14.1 s | 372 | 0.3 s |
| **128K** | 175 | 4,657 | 28.1 s | 392 | 0.4 s |
| **256K** | 168 | 4,360 | 60.2 s | 370 | 0.7 s |

**Legal** (a brief as the prompt)

| Context (prompt tokens) | Decode tok/s, one request | Prompt read tok/s† | First token, cold† | Three requests together, tok/s summed | First token, document already cached |
|---|---:|---:|---:|---:|---:|
| **1K** | 97 | 1,485 | 0.7 s | 245 | 0.2 s* |
| **8K** | 100 | 3,152 | 2.6 s | 254 | 0.2 s |
| **32K** | 93 | 4,070 | 8.1 s | 217 | 0.3 s |
| **64K** | 100 | 3,809 | 17.9 s | 230 | 0.3 s |
| **128K** | 94 | 3,944 | 34.1 s | 216 | 0.4 s |
| **256K** | 90 | 4,094 | 64.5 s | 205 | 0.7 s |

\* At 1K the request's fixed cost is most of the read, so the cache cannot show.

† In both benches the first cold read of a legal or prose document was 1 to 11 s slower than its second repeat (code's
repeats agreed within 1.1 s; the cause is not established). These cells are the mean of the slow and the fast repeat;
the second repeat alone read the 128K brief in 28.7 s and the 256K brief in 59.2 s (prose 28.4 and 58.8 s).

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
  128K code requests released together got their first tokens after 77.0, 87.4 and 92.5 s (28.15 s alone), their
  reads interleaved, and then decoded together; counted from the earliest first token to the last request's end, the
  three give 143 tok/s on code and 112 on legal. With the document already in the cache the wait disappears: three
  code streams on one cached 128K document decode 400 tok/s over that window (legal 242), within 2 % of the sum.
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
  four conversations kept warm and a context of 524,288 tokens. One request decodes at 160 to 178 tok/s on code and
  80 to 100 tok/s on legal and prose prompts of 1K to 256K tokens (MEASURED 2026-11-19 at 175 W, the final bench
  below); the pipeline carries one request per step, so prompts are read one after another and three requests
  together decode at about 2.2 to 2.5 times one stream's rate once every prompt is in. It is not a batched-decode
  engine.
- **What it is tuned for.** Long-context agent work: one request at a time, prompts of 60,000 to 200,000 tokens that
  grow by small turns. Since 2026-11-17 it is the owner's default DeepSeek server.

## What runs

The standing configuration: the serving set since window c2a (2026-11-19 03:55Z), the set every optimization of the
plan's list left; re-recorded after windows c3b and c7a (05:02Z and 05:39Z).

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
| Adopted switches | ten lines: `VLLM_V41_DRAFT_GEMV2=1 VLLM_V41_EMPTY_CACHE_TOKENS=8192 VLLM_V41_INDEX_TC=1 VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD=128 VLLM_V41_PP_ANSWER_NO_SLEEP=1 VLLM_V41_PP_EMPTY_AHEAD=1 VLLM_V41_PP_MID_PREFILL_SKIP=1 VLLM_V41_PP_TURN_CHUNKS=3 VLLM_V41_PREFIX_TOKEN_CACHE=1 VLLM_V41_WO_A_BATCHED_4K=1` (the window of each is named under What worked); the serving record carries their digest |
| Accepted capture | `accepted/quality-TC3-WOA4K.json` (window c2a: the batched projection is another rounding) while it is adopted; `quality-TC3.json` (window p5) before it; window o2's capture otherwise; decided in one place from the adopted lines |
| Power and clock | 175 W per card and a 1,452 MHz clock ceiling (cards at 1,447 MHz) since 2026-11-18 12:09:23Z. Before: 150 W and 1,477 MHz to 2026-11-17 18:58Z; 200 W to 2026-11-18 11:15:48Z (the host tripped under four long chats); 150 W from 11:45Z |
| Serving | every start check runs on a test port and is recorded; the serving boot on the public port sends no request and needs a record that matches the installed bytes ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), section 13) |

Not in what serves: the conversation store (every state check passed in windows st4 and st7; it failed its timing line
in st7 and stays off), the lean pipeline hand-off (failed its line in window c3b), the packed expert records (failed
theirs in window c7a), the dense library's split switch (inconclusive in window t1), the drafter off (the quality
window's one-slot boots only), and the loop statistics `VLLM_V41_PP_LOOP_STATS` (a diagnostic of the windows, never
served).

## Measured

MEASURED on 2026-11-17, 2026-11-18 and 2026-11-19; each row names its window, and its power where it is not the
window's standing power (150 W through window sv1; 200 W from 2026-11-17 18:58Z; 175 W and the 1,452 MHz ceiling since
2026-11-18 12:09Z). The full tables with dates and times are in [docs/measurements.md](docs/measurements.md); the
bench of 2026-11-18 on the previous set is kept in [docs/measurements-2026-11-18.md](docs/measurements-2026-11-18.md).

### The serving set since window c2a (2026-11-19, kit 9S, ten switches; 175 W, clock ceiling 1,452 MHz)

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
| The final bench, one stream decode at 128K (two repeats of one text; the spread across texts of a kind is about ±9 %) | code 175.0, legal 94.2, prose 79.9 tok/s (1K: 178.3 / 97.3 / 85.6; 256K: 168.5 / 90.0 / 84.4) | final bench |
| The final bench, one cold read, prefill at 128K and 256K | code 4,657 and 4,360 tok/s (first token 28.15 and 60.23 s); legal 3,944 and 4,094 (34.09 and 64.46 s, means of a slow first repeat and a fast second: 28.70 and 59.23 s on the second); prose 4,331 and 4,247 (30.40 and 61.89 s; second repeats 28.37 and 58.79) | final bench |
| The final bench, three streams released together at 128K, sum of the streams' decode rates | code 391.5, legal 216.1, prose 192.3 tok/s (from the earliest first token to the last request's end: 142.9 / 112.3 / 105.7) | final bench |
| The prefix measure again: first token on a hit at 128K; three streams on one cached 128K document, the all-hit repeat | 0.42 to 0.43 s; code 399.8, legal 242.2, prose 221.5 tok/s over that window (405.1 / 244.0 / 223.2 summed) | prefix measure |
| Highest card sample over the bench and the measure (2 h 15 min, 1,522 samples) | 30,418 MiB | final bench |

### The set before it: since window p6b (2026-11-18, kit 9N, nine switches; 175 W, clock ceiling 1,452 MHz)

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

**The final bench, 2026-11-19 (the serving set every optimization left).** MEASURED on 2026-11-19, 05:53:12Z to 07:24:46Z, on the
serving set of window c7a (kit `PP3KIT9S-over-cut`, the ten adopted lines, four slots; the server not restarted for the bench), 175 W per card
and the 1,452 MHz clock ceiling: the same bench as 2026-11-18, unchanged: code, legal and prose prompts of 1,024 / 8,192 / 32,768 / 65,536 /
131,072 / 262,144 tokens, read cold, one, two and three streams released together, two repeats each (108 groups, 216 measured requests;
greedy, 1,024 tokens out, thinking off; judge PASS completeness, the token digest of every lone repeat matched its pair 198 of 198). "Prefill
tok/s" is the prompt's tokens over the time to its first token. The two aggregate columns: **Total decode tok/s** is all streams' output
tokens over the window from the earliest first token to the last request's end (the bench tool's arithmetic; the 2026-11-18 text described
this column as starting at the first request's start, which was wrong: the numbers are unchanged, the words are corrected); with two or
three cold streams the prompts are read one after another, interleaved, so the first tokens of three 128K code streams came at 77.0, 87.4
and 92.5 s (28.15 s alone) and the three then decoded together. **Sum of decode rates** adds each stream's own decode rate while all of
them stream: the decode throughput once every prompt is in. Three findings. (1) A one-stream decode figure is the figure of the text the
bench generated: the batched projection is another rounding, so the greedy text moved in every cell of 8K and longer against 2026-11-18 and
the decode per cell moved with it, by up to 8.6 % either way, while the decode round itself did not change (windows c3b and c7a); the
spread across texts of one kind is about ±9 %, two repeats of one text agree within 2 %. (2) In both benches the first cold read of a legal
or prose document is 1 to 11 s slower than its second repeat (code's agree within 1.1 s); the legal and prose TTFT and Prefill cells
of one stream are the mean of a slow and a fast repeat (the second repeat alone: legal 28.70 s at 128K and 59.23 s at 256K, prose 28.37
and 58.79 s); the cause is not established. (3) The cold code reads of 32K and longer were 4.8 to 5.0 % faster than on 2026-11-18, the
batched projection's measure at the bench's lengths. The 2026-11-18 bench and its prefix tables are kept unchanged in
[docs/measurements-2026-11-18.md](docs/measurements-2026-11-18.md).

#### Code

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | cold, thinking off, 1,024 out. Mean [min–max], two runs.

| Prompt | Streams | TTFT s by stream | Prefill tok/s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates |
|---|---:|---|---|---|---:|---:|
| 1K | 1 | 0.66 [0.64–0.67] | 1558 [1519–1597] | 178.3 [177.7–178.9] | 178.3 [177.7–178.9] | 178.3 [177.7–178.9] |
| 1K | 2 | 0.80 [0.66–0.94] / 0.80 [0.65–0.94] | 1324 [1091–1557] / 1330 [1094–1565] | 167.0 [163.3–170.8] / 167.0 [163.4–170.6] | 326.1 [326.1–326.1] | 334.0 [333.9–334.2] |
| 1K | 3 | 1.16 [0.93–1.39] / 0.93 [0.65–1.21] / 1.16 [0.65–1.67] | 918 [737–1099] / 1208 [848–1569] / 1089 [612–1566] | 143.8 [143.7–143.9] / 140.2 [130.8–149.5] / 144.1 [138.7–149.5] | 402.6 [390.2–415.0] | 428.1 [424.0–432.2] |
| 8K | 1 | 2.53 [2.46–2.60] | 3244 [3156–3333] | 174.8 [173.2–176.4] | 174.8 [173.2–176.4] | 174.8 [173.2–176.4] |
| 8K | 2 | 5.18 [2.46–7.91] / 5.18 [2.46–7.91] | 2186 [1036–3336] / 2185 [1036–3333] | 147.4 [123.9–170.9] / 147.6 [123.6–171.5] | 179.1 [178.9–179.2] | 295.0 [294.5–295.4] |
| 8K | 3 | 7.97 [2.47–13.47] / 7.97 [2.46–13.47] / 8.68 [8.68–8.69] | 1961 [608–3314] / 1966 [608–3323] / 944 [943–944] | 142.6 [113.5–171.7] / 142.6 [113.5–171.7] / 134.8 [134.8–134.8] | 180.9 [180.9–181.0] | 420.0 [420.0–420.1] |
| 32K | 1 | 7.49 [7.28–7.69] | 4380 [4262–4499] | 174.1 [172.2–176.0] | 174.1 [172.2–176.0] | 174.1 [172.2–176.0] |
| 32K | 2 | 11.99 [11.99–11.99] / 18.53 [18.53–18.54] | 2734 [2733–2734] / 1768 [1768–1768] | 116.4 [116.3–116.5] / 172.2 [171.9–172.4] | 163.9 [163.7–164.0] | 288.6 [288.2–289.0] |
| 32K | 3 | 15.93 [15.93–15.94] / 25.49 [25.47–25.51] / 30.29 [30.27–30.31] | 2056 [2055–2057] / 1286 [1284–1287] / 1082 [1081–1083] | 93.3 [93.2–93.4] / 135.2 [135.1–135.3] / 171.9 [171.8–171.9] | 151.1 [151.0–151.2] | 400.4 [400.1–400.7] |
| 64K | 1 | 14.14 [13.91–14.37] | 4636 [4560–4711] | 160.3 [157.9–162.7] | 160.3 [157.9–162.7] | 160.3 [157.9–162.7] |
| 64K | 2 | 28.44 [25.11–31.77] / 28.44 [25.10–31.78] | 2336 [2063–2610] / 2336 [2062–2611] | 133.9 [109.0–158.8] / 133.9 [109.0–158.8] | 156.1 [156.1–156.1] | 267.8 [267.8–267.9] |
| 64K | 3 | 35.49 [35.47–35.51] / 45.27 [45.26–45.28] / 50.20 [50.19–50.21] | 1847 [1845–1848] / 1448 [1447–1448] / 1306 [1305–1306] | 87.9 [87.9–88.0] / 125.1 [125.1–125.2] / 158.4 [158.0–158.8] | 144.9 [144.9–145.0] | 371.5 [371.1–371.8] |
| 128K | 1 | 28.15 [27.95–28.34] | 4657 [4624–4690] | 175.0 [173.4–176.5] | 175.0 [173.4–176.5] | 175.0 [173.4–176.5] |
| 128K | 2 | 56.44 [52.95–59.93] / 56.45 [52.92–59.97] | 2331 [2187–2475] / 2331 [2186–2477] | 142.6 [112.6–172.7] / 142.8 [112.6–173.1] | 158.4 [158.3–158.4] | 285.5 [285.3–285.6] |
| 128K | 3 | 84.77 [77.04–92.50] / 84.79 [77.00–92.58] / 87.40 [87.38–87.41] | 1559 [1417–1701] / 1559 [1416–1702] / 1500 [1499–1500] | 130.1 [88.2–171.9] / 129.9 [88.4–171.4] / 131.5 [131.3–131.7] | 142.9 [142.8–143.0] | 391.5 [391.0–392.0] |
| 256K | 1 | 60.23 [57.83–62.62] | 4360 [4186–4533] | 168.5 [165.2–171.7] | 168.5 [165.2–171.7] | 168.5 [165.2–171.7] |
| 256K | 2 | 115.95 [112.11–119.80] / 115.94 [112.15–119.74] | 2263 [2188–2338] / 2263 [2189–2337] | 135.6 [104.8–166.5] / 135.9 [104.9–166.9] | 148.6 [148.5–148.7] | 271.5 [271.3–271.7] |
| 256K | 3 | 171.13 [165.54–176.72] / 182.39 [182.39–182.39] / 171.13 [165.48–176.78] | 1533 [1483–1584] / 1437 [1437–1437] / 1534 [1483–1584] | 102.5 [82.0–123.0] / 165.1 [164.5–165.7] / 102.8 [82.0–123.6] | 133.0 [132.9–133.1] | 370.4 [369.5–371.3] |

#### Legal

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | cold, thinking off, 1,024 out. Mean [min–max], two runs.

| Prompt | Streams | TTFT s by stream | Prefill tok/s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates |
|---|---:|---|---|---|---:|---:|
| 1K | 1 | 0.69 [0.65–0.74] | 1485 [1390–1579] | 97.3 [94.0–100.6] | 97.3 [94.0–100.6] | 97.3 [94.0–100.6] |
| 1K | 2 | 0.82 [0.67–0.96] / 0.82 [0.68–0.95] | 1298 [1062–1533] / 1289 [1077–1501] | 94.2 [93.0–95.4] / 94.2 [93.0–95.4] | 185.8 [185.8–185.9] | 188.4 [188.4–188.4] |
| 1K | 3 | 0.82 [0.67–0.96] / 1.18 [0.67–1.69] / 1.10 [0.95–1.25] | 1294 [1064–1525] / 1065 [606–1523] / 948 [822–1073] | 80.3 [78.1–82.6] / 82.5 [80.7–84.3] / 82.1 [79.7–84.4] | 237.6 [233.4–241.8] | 244.9 [242.1–247.7] |
| 8K | 1 | 2.60 [2.51–2.70] | 3152 [3035–3269] | 99.7 [91.1–108.3] | 99.7 [91.1–108.3] | 99.7 [91.1–108.3] |
| 8K | 2 | 8.01 [8.01–8.01] / 2.53 [2.53–2.53] | 1023 [1022–1023] / 3241 [3239–3243] | 104.3 [104.2–104.4] / 84.0 [83.9–84.1] | 133.8 [133.7–133.9] | 188.3 [188.1–188.5] |
| 8K | 3 | 8.09 [2.52–13.65] / 8.08 [2.53–13.62] / 8.83 [8.82–8.83] | 1923 [600–3246] / 1920 [601–3240] / 928 [927–929] | 84.0 [68.3–99.7] / 84.0 [68.2–99.8] / 85.6 [85.5–85.7] | 143.6 [143.5–143.7] | 253.5 [253.3–253.8] |
| 32K | 1 | 8.10 [7.49–8.71] | 4070 [3764–4376] | 92.9 [92.0–93.9] | 92.9 [92.0–93.9] | 92.9 [92.0–93.9] |
| 32K | 2 | 15.65 [12.38–18.93] / 15.64 [12.38–18.91] | 2189 [1731–2647] / 2190 [1733–2648] | 81.1 [72.1–90.1] / 81.2 [71.8–90.6] | 114.5 [114.3–114.8] | 162.3 [161.9–162.7] |
| 32K | 3 | 21.24 [16.45–26.04] / 28.48 [26.04–30.92] / 23.68 [16.43–30.93] | 1626 [1259–1992] / 1159 [1060–1259] / 1527 [1059–1994] | 65.3 [55.2–75.4] / 81.0 [75.4–86.6] / 70.9 [55.2–86.6] | 116.6 [116.6–116.6] | 217.2 [217.2–217.2] |
| 64K | 1 | 17.95 [14.30–21.59] | 3809 [3035–4582] | 99.5 [98.6–100.3] | 99.5 [98.6–100.3] | 99.5 [98.6–100.3] |
| 64K | 2 | 25.81 [25.81–25.81] / 32.50 [32.48–32.51] | 2539 [2539–2540] / 2017 [2016–2018] | 75.2 [75.1–75.3] / 96.7 [96.6–96.8] | 118.5 [118.4–118.7] | 171.9 [171.7–172.0] |
| 64K | 3 | 43.95 [36.51–51.39] / 46.39 [46.37–46.42] / 43.98 [36.60–51.36] | 1535 [1275–1795] / 1413 [1412–1413] / 1533 [1276–1791] | 75.0 [56.9–93.1] / 79.8 [79.8–79.8] / 75.2 [57.0–93.3] | 119.0 [118.9–119.0] | 230.0 [229.9–230.1] |
| 128K | 1 | 34.09 [28.70–39.48] | 3944 [3320–4568] | 94.2 [93.9–94.6] | 94.2 [93.9–94.6] | 94.2 [93.9–94.6] |
| 128K | 2 | 61.39 [61.37–61.42] / 54.39 [54.35–54.43] | 2135 [2134–2136] / 2410 [2408–2412] | 91.7 [91.4–91.9] / 71.0 [71.0–71.0] | 112.5 [112.4–112.6] | 162.6 [162.4–162.9] |
| 128K | 3 | 92.25 [89.59–94.91] / 79.22 [79.18–79.26] / 92.24 [89.65–94.83] | 1422 [1381–1463] / 1655 [1654–1655] / 1422 [1382–1462] | 81.4 [74.9–87.9] / 53.3 [53.3–53.3] / 81.4 [75.1–87.7] | 112.3 [112.2–112.4] | 216.1 [215.9–216.3] |
| 256K | 1 | 64.46 [59.23–69.68] | 4094 [3762–4426] | 90.0 [87.7–92.3] | 90.0 [87.7–92.3] | 90.0 [87.7–92.3] |
| 256K | 2 | 122.47 [122.41–122.53] / 114.92 [114.88–114.95] | 2140 [2139–2142] / 2281 [2280–2282] | 88.5 [88.2–88.8] / 67.4 [67.3–67.4] | 106.9 [106.8–107.0] | 155.9 [155.5–156.2] |
| 256K | 3 | 186.56 [186.51–186.62] / 169.70 [169.61–169.79] / 180.84 [180.80–180.89] | 1405 [1405–1406] / 1545 [1544–1546] / 1450 [1449–1450] | 84.3 [84.2–84.3] / 50.0 [49.9–50.1] / 70.9 [70.9–70.9] | 105.7 [105.6–105.8] | 205.1 [205.1–205.2] |

#### Prose

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | cold, thinking off, 1,024 out. Mean [min–max], two runs.

| Prompt | Streams | TTFT s by stream | Prefill tok/s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates |
|---|---:|---|---|---|---:|---:|
| 1K | 1 | 0.65 [0.63–0.67] | 1571 [1527–1615] | 85.6 [85.0–86.1] | 85.6 [85.0–86.1] | 85.6 [85.0–86.1] |
| 1K | 2 | 0.92 [0.92–0.93] / 0.65 [0.65–0.65] | 1110 [1105–1114] / 1578 [1577–1579] | 82.1 [82.0–82.1] / 80.4 [80.4–80.4] | 160.6 [160.6–160.7] | 162.5 [162.4–162.5] |
| 1K | 3 | 0.93 [0.64–1.21] / 1.01 [0.64–1.37] / 1.29 [0.93–1.65] | 1219 [849–1589] / 1169 [747–1591] / 858 [621–1096] | 70.2 [67.9–72.5] / 70.5 [69.9–71.2] / 71.9 [71.2–72.5] | 206.2 [203.1–209.3] | 212.6 [211.6–213.6] |
| 8K | 1 | 2.54 [2.47–2.61] | 3231 [3142–3320] | 93.5 [92.6–94.3] | 93.5 [92.6–94.3] | 93.5 [92.6–94.3] |
| 8K | 2 | 5.22 [2.49–7.94] / 5.20 [2.47–7.92] | 2163 [1031–3295] / 2176 [1034–3318] | 83.3 [75.6–91.1] / 83.2 [75.2–91.2] | 122.7 [122.5–122.8] | 166.5 [166.3–166.7] |
| 8K | 3 | 2.49 [2.49–2.49] / 13.50 [13.50–13.51] / 8.71 [8.71–8.72] | 3295 [3294–3296] / 607 [607–607] / 940 [940–940] | 62.2 [62.2–62.2] / 86.6 [86.5–86.6] / 75.8 [75.8–75.9] | 134.4 [134.4–134.4] | 224.6 [224.5–224.7] |
| 32K | 1 | 7.92 [7.38–8.46] | 4158 [3876–4440] | 86.9 [86.2–87.6] | 86.9 [86.2–87.6] | 86.9 [86.2–87.6] |
| 32K | 2 | 18.66 [18.66–18.66] / 12.13 [12.12–12.14] | 1756 [1756–1756] / 2702 [2700–2703] | 84.7 [84.6–84.8] / 68.3 [68.2–68.4] | 109.9 [109.8–110.1] | 153.0 [152.8–153.3] |
| 32K | 3 | 20.91 [16.10–25.72] / 23.35 [16.11–30.59] / 28.15 [25.68–30.62] | 1655 [1274–2036] / 1553 [1071–2034] / 1173 [1070–1276] | 61.9 [52.8–70.9] / 66.8 [52.8–80.8] / 75.7 [70.8–80.6] | 112.9 [112.8–113.0] | 204.4 [204.3–204.5] |
| 64K | 1 | 14.41 [14.10–14.71] | 4550 [4454–4647] | 86.2 [85.6–86.8] | 86.2 [85.6–86.8] | 86.2 [85.6–86.8] |
| 64K | 2 | 28.85 [25.52–32.17] / 28.87 [25.50–32.23] | 2303 [2037–2568] / 2302 [2033–2570] | 75.3 [66.9–83.6] / 75.4 [67.1–83.6] | 108.1 [108.0–108.2] | 150.6 [150.6–150.7] |
| 64K | 3 | 48.37 [45.87–50.87] / 36.04 [36.03–36.05] / 48.38 [45.87–50.89] | 1358 [1288–1429] / 1819 [1818–1819] / 1358 [1288–1429] | 74.8 [69.8–79.8] / 51.7 [51.7–51.7] / 74.7 [69.8–79.7] | 110.9 [110.8–110.9] | 201.2 [201.2–201.3] |
| 128K | 1 | 30.40 [28.37–32.43] | 4331 [4042–4621] | 79.9 [76.2–83.6] | 79.9 [76.2–83.6] | 79.9 [76.2–83.6] |
| 128K | 2 | 60.84 [60.83–60.85] / 53.81 [53.81–53.81] | 2154 [2154–2155] / 2436 [2436–2436] | 80.8 [80.8–80.8] / 64.2 [64.2–64.2] | 103.8 [103.8–103.9] | 145.0 [145.0–145.1] |
| 128K | 3 | 88.83 [88.75–88.90] / 94.10 [94.02–94.18] / 78.39 [78.32–78.46] | 1476 [1474–1477] / 1393 [1392–1394] / 1672 [1671–1674] | 66.6 [66.6–66.6] / 76.7 [76.7–76.8] / 49.0 [48.9–49.1] | 105.7 [105.5–105.8] | 192.3 [192.2–192.5] |
| 256K | 1 | 61.89 [58.79–64.99] | 4247 [4034–4459] | 84.4 [84.0–84.8] | 84.4 [84.0–84.8] | 84.4 [84.0–84.8] |
| 256K | 2 | 114.07 [114.03–114.11] / 121.72 [121.66–121.78] | 2298 [2297–2299] / 2154 [2153–2155] | 62.9 [62.8–63.0] / 81.5 [81.4–81.5] | 101.1 [101.0–101.3] | 144.3 [144.2–144.5] |
| 256K | 3 | 176.31 [167.73–184.90] / 173.49 [167.89–179.09] / 181.93 [179.15–184.72] | 1490 [1418–1563] / 1513 [1464–1561] / 1441 [1419–1463] | 62.3 [47.3–77.3] / 56.8 [47.4–66.1] / 71.6 [66.0–77.3] | 101.7 [101.6–101.7] | 190.7 [190.7–190.7] |

**With prefix sharing, again: the same measure on the kit 9S set.** For each length and kind one cold "warm read" put the
document in the prefix cache; then one, two and three streams on that document, released together, two repeats (108 groups,
216 measured requests; MEASURED 2026-11-19 07:24:46Z to 08:08:28Z on the same serving set, 175 W, the 1,452 MHz
ceiling, right after the bench above, the server not restarted). A hit is a first token under a quarter of the warm read's.
Hits 150 of 216 as on 2026-11-18: the 36 requests at 1K are not hits by that rule, and 30 requests re-read the document,
one per new slot the first time a second or a third stream shared it, never after. On a hit the first token at 128K came in
0.42 to 0.43 s and at 256K in 0.61 to 0.68 s; three streams on one cached 128K document, the all-hit repeat: code 405.1 tok/s
summed and 399.8 over the window, legal 244.0 and 242.2, prose 223.2 and 221.5. 0 of 150 hits reproduced the warm read's
token digest and all 30 re-reads did (the judge's count of 36 adds six of the 1K requests). The warm reads were 4.0 to
4.2 % faster than on 2026-11-18 at 128K and 256K (code 27.96 and 57.76 s). Hit decode rates moved with the generated text
as in the bench above (the rounding change moved the greedy continuation in every cell of 8K and longer).

#### Code, with prefix sharing

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | document in the prefix cache, thinking off, 1,024 out. Mean [min–max], two runs; the warm read is the one cold read that put the document in the cache.

| Prompt | Streams | TTFT s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates | Hits | Overlap |
|---|---:|---|---|---:|---:|---|---|
| 1K | warm read | 0.66 (1,541 tok/s) | 177.0 | - | - | - | - |
| 1K | 1 | 0.22 [0.19–0.25] | 179.3 [176.1–182.4] | 179.3 [176.1–182.4] | 179.3 [176.1–182.4] | 0/2 | together |
| 1K | 2 | 0.19 [0.19–0.20] / 0.48 [0.25–0.71] | 164.8 [157.4–172.2] / 171.3 [169.3–173.3] | 327.7 [311.8–343.6] | 336.1 [326.7–345.6] | 0/4 | together |
| 1K | 3 | 0.26 [0.21–0.31] / 0.29 [0.24–0.34] / 0.55 [0.27–0.83] | 144.5 [139.6–149.4] / 145.2 [138.5–151.9] / 149.8 [149.0–150.5] | 429.2 [411.5–446.9] | 439.5 [427.2–451.8] | 0/6 | together |
| 8K | warm read | 2.49 (3,295 tok/s) | 176.0 | - | - | - | - |
| 8K | 1 | 0.21 [0.21–0.22] | 180.0 [177.4–182.6] | 180.0 [177.4–182.6] | 180.0 [177.4–182.6] | 2/2 | together |
| 8K | 2 | 0.26 [0.22–0.29] / 3.16 [0.25–6.07] | 147.0 [120.1–173.9] / 172.2 [171.7–172.8] | 259.1 [173.3–344.9] | 319.2 [291.8–346.7] | 3/4 | together |
| 8K | 3 | 0.34 [0.33–0.34] / 0.27 [0.25–0.28] / 3.30 [0.23–6.36] | 131.9 [112.6–151.2] / 130.9 [111.8–150.0] / 155.4 [149.0–161.9] | 346.4 [246.9–445.8] | 418.3 [386.3–450.2] | 5/6 | together |
| 32K | warm read | 7.33 (4,468 tok/s) | 176.4 | - | - | - | - |
| 32K | 1 | 0.26 [0.26–0.26] | 185.0 [183.3–186.7] | 185.0 [183.3–186.7] | 185.0 [183.3–186.7] | 2/2 | together |
| 32K | 2 | 7.76 [0.27–15.26] / 0.31 [0.28–0.35] | 175.1 [174.5–175.8] / 126.2 [76.0–176.5] | 223.3 [98.4–348.3] | 301.4 [251.7–351.0] | 3/4 | together |
| 32K | 3 | 0.28 [0.27–0.30] / 8.03 [0.39–15.68] / 0.36 [0.33–0.39] | 112.9 [73.9–151.8] / 164.9 [154.3–175.5] / 113.6 [74.3–153.0] | 299.5 [144.7–454.3] | 391.4 [323.7–459.1] | 5/6 | together |
| 64K | warm read | 13.93 (4,705 tok/s) | 162.6 | - | - | - | - |
| 64K | 1 | 0.30 [0.29–0.30] | 170.1 [168.1–172.2] | 170.1 [168.1–172.2] | 170.1 [168.1–172.2] | 2/2 | together |
| 64K | 2 | 0.32 [0.31–0.33] / 11.88 [0.53–23.23] | 111.2 [63.2–159.1] / 163.2 [162.9–163.5] | 193.3 [70.1–316.6] | 274.4 [226.1–322.7] | 3/4 | together |
| 64K | 3 | 0.50 [0.47–0.52] / 0.40 [0.35–0.45] / 12.11 [0.38–23.84] | 101.8 [61.4–142.3] / 101.0 [61.1–140.9] / 151.6 [140.0–163.2] | 261.0 [103.2–418.9] | 354.4 [285.7–423.2] | 5/6 | together |
| 128K | warm read | 27.96 (4,687 tok/s) | 176.4 | - | - | - | - |
| 128K | 1 | 0.43 [0.42–0.44] | 163.3 [161.4–165.3] | 163.3 [161.4–165.3] | 163.3 [161.4–165.3] | 2/2 | together |
| 128K | 2 | 0.49 [0.45–0.53] / 19.03 [0.46–37.60] | 109.5 [62.3–156.7] / 166.4 [155.1–177.7] | 178.7 [47.7–309.7] | 275.9 [240.0–311.8] | 3/4 | together |
| 128K | 3 | 0.64 [0.60–0.68] / 19.37 [0.59–38.15] / 0.52 [0.52–0.52] | 98.6 [60.7–136.6] / 155.7 [134.9–176.6] / 97.0 [60.5–133.6] | 235.2 [70.7–399.8] | 351.4 [297.8–405.1] | 5/6 | together |
| 256K | warm read | 57.76 (4,539 tok/s) | 171.3 | - | - | - | - |
| 256K | 1 | 0.65 [0.63–0.66] | 167.3 [164.8–169.8] | 167.3 [164.8–169.8] | 167.3 [164.8–169.8] | 2/2 | together |
| 256K | 2 | 34.31 [1.07–67.54] / 0.71 [0.70–0.72] | 165.1 [158.8–171.4] / 107.0 [62.1–151.8] | 164.9 [28.1–301.7] | 272.1 [233.5–310.7] | 3/4 | together |
| 256K | 3 | 0.97 [0.81–1.14] / 34.68 [1.06–68.30] / 0.96 [0.74–1.18] | 99.3 [60.1–138.5] / 154.2 [137.0–171.4] / 96.4 [61.1–131.7] | 216.8 [41.8–391.8] | 349.9 [292.7–407.2] | 5/6 | together |

#### Legal, with prefix sharing

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | document in the prefix cache, thinking off, 1,024 out. Mean [min–max], two runs; the warm read is the one cold read that put the document in the cache.

| Prompt | Streams | TTFT s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates | Hits | Overlap |
|---|---:|---|---|---:|---:|---|---|
| 1K | warm read | 0.67 (1,539 tok/s) | 100.7 | - | - | - | - |
| 1K | 1 | 0.19 [0.19–0.19] | 103.0 [102.5–103.5] | 103.0 [102.5–103.5] | 103.0 [102.5–103.5] | 0/2 | together |
| 1K | 2 | 0.23 [0.19–0.27] / 0.47 [0.22–0.72] | 95.0 [92.3–97.7] / 96.2 [95.2–97.3] | 187.8 [181.2–194.3] | 191.2 [187.5–195.0] | 0/4 | together |
| 1K | 3 | 0.30 [0.20–0.40] / 0.60 [0.27–0.94] / 0.26 [0.20–0.32] | 83.7 [82.0–85.4] / 85.0 [84.2–85.8] / 83.5 [81.0–86.1] | 247.0 [238.3–255.8] | 252.2 [247.2–257.2] | 0/6 | together |
| 8K | warm read | 2.53 (3,234 tok/s) | 108.2 | - | - | - | - |
| 8K | 1 | 0.22 [0.21–0.22] | 97.0 [96.5–97.5] | 97.0 [96.5–97.5] | 97.0 [96.5–97.5] | 2/2 | together |
| 8K | 2 | 0.27 [0.25–0.29] / 3.17 [0.23–6.11] | 83.1 [74.2–92.0] / 97.5 [91.6–103.4] | 156.4 [129.8–183.0] | 180.6 [177.6–183.7] | 3/4 | together |
| 8K | 3 | 3.41 [0.31–6.52] / 0.35 [0.25–0.45] / 0.30 [0.24–0.37] | 87.0 [80.7–93.2] / 74.1 [67.9–80.4] / 74.1 [67.1–81.0] | 209.4 [177.9–240.9] | 235.2 [228.2–242.2] | 5/6 | together |
| 32K | warm read | 7.52 (4,356 tok/s) | 94.0 | - | - | - | - |
| 32K | 1 | 0.26 [0.25–0.26] | 97.4 [97.1–97.8] | 97.4 [97.1–97.8] | 97.4 [97.1–97.8] | 2/2 | together |
| 32K | 2 | 7.89 [0.33–15.44] / 0.28 [0.25–0.31] | 91.8 [91.3–92.3] / 70.8 [50.0–91.7] | 130.4 [77.7–183.2] | 162.6 [141.3–184.0] | 3/4 | together |
| 32K | 3 | 8.16 [0.32–16.00] / 0.38 [0.38–0.39] / 0.37 [0.31–0.43] | 83.0 [80.5–85.4] / 64.0 [47.1–80.8] / 64.1 [47.0–81.2] | 176.0 [110.9–241.2] | 211.0 [179.5–242.5] | 5/6 | together |
| 64K | warm read | 14.27 (4,592 tok/s) | 100.1 | - | - | - | - |
| 64K | 1 | 0.33 [0.32–0.33] | 97.8 [97.5–98.2] | 97.8 [97.5–98.2] | 97.8 [97.5–98.2] | 2/2 | together |
| 64K | 2 | 0.39 [0.36–0.41] / 14.26 [0.35–28.18] | 63.3 [34.4–92.2] / 95.5 [91.7–99.3] | 118.5 [53.7–183.2] | 158.8 [133.7–183.9] | 3/4 | together |
| 64K | 3 | 14.65 [0.44–28.87] / 0.48 [0.47–0.49] / 0.37 [0.35–0.38] | 88.6 [81.1–96.0] / 57.4 [33.4–81.4] / 57.0 [33.3–80.7] | 160.0 [78.4–241.6] | 203.0 [162.7–243.2] | 5/6 | together |
| 128K | warm read | 28.71 (4,565 tok/s) | 95.2 | - | - | - | - |
| 128K | 1 | 0.42 [0.41–0.42] | 98.6 [98.2–99.0] | 98.6 [98.2–99.0] | 98.6 [98.2–99.0] | 2/2 | together |
| 128K | 2 | 0.52 [0.52–0.53] / 23.25 [0.46–46.04] | 62.3 [31.3–93.3] / 94.0 [92.8–95.2] | 110.9 [36.4–185.4] | 156.3 [126.5–186.1] | 3/4 | together |
| 128K | 3 | 0.65 [0.60–0.69] / 0.56 [0.54–0.58] / 23.70 [0.51–46.89] | 56.2 [30.6–81.8] / 55.9 [30.5–81.3] / 88.0 [80.9–95.1] | 148.0 [53.7–242.2] | 200.1 [156.2–244.0] | 5/6 | together |
| 256K | warm read | 59.21 (4,428 tok/s) | 92.5 | - | - | - | - |
| 256K | 1 | 0.68 [0.63–0.73] | 94.5 [94.0–95.0] | 94.5 [94.0–95.0] | 94.5 [94.0–95.0] | 2/2 | together |
| 256K | 2 | 38.95 [0.78–77.12] / 0.96 [0.79–1.13] | 89.5 [86.4–92.6] / 59.6 [30.8–88.5] | 97.8 [23.4–172.2] | 149.1 [123.4–174.9] | 3/4 | together |
| 256K | 3 | 1.01 [0.98–1.05] / 0.87 [0.84–0.90] / 39.60 [1.02–78.18] | 53.7 [30.0–77.3] / 53.4 [30.0–76.8] / 85.0 [77.8–92.1] | 132.3 [34.7–229.9] | 192.0 [152.1–231.9] | 5/6 | together |

#### Prose, with prefix sharing

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | document in the prefix cache, thinking off, 1,024 out. Mean [min–max], two runs; the warm read is the one cold read that put the document in the cache.

| Prompt | Streams | TTFT s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates | Hits | Overlap |
|---|---:|---|---|---:|---:|---|---|
| 1K | warm read | 0.64 (1,592 tok/s) | 86.2 | - | - | - | - |
| 1K | 1 | 0.19 [0.19–0.19] | 84.5 [81.3–87.8] | 84.5 [81.3–87.8] | 84.5 [81.3–87.8] | 0/2 | together |
| 1K | 2 | 0.49 [0.27–0.71] / 0.20 [0.19–0.21] | 82.5 [81.8–83.2] / 81.1 [79.4–82.7] | 161.3 [157.2–165.3] | 163.6 [161.2–166.0] | 0/4 | together |
| 1K | 3 | 0.64 [0.36–0.93] / 0.26 [0.23–0.29] / 0.27 [0.24–0.30] | 72.7 [72.3–73.1] / 71.2 [69.8–72.5] / 71.3 [69.7–72.9] | 212.0 [206.8–217.3] | 215.2 [211.8–218.5] | 0/6 | together |
| 8K | warm read | 2.53 (3,233 tok/s) | 94.9 | - | - | - | - |
| 8K | 1 | 0.21 [0.21–0.22] | 93.3 [92.7–93.9] | 93.3 [92.7–93.9] | 93.3 [92.7–93.9] | 2/2 | together |
| 8K | 2 | 3.18 [0.28–6.08] / 0.22 [0.22–0.23] | 89.8 [88.7–90.9] / 80.2 [72.1–88.3] | 148.0 [119.6–176.4] | 170.0 [163.0–177.0] | 3/4 | together |
| 8K | 3 | 0.29 [0.27–0.31] / 0.35 [0.33–0.36] / 3.30 [0.23–6.37] | 72.0 [65.9–78.0] / 72.2 [66.2–78.3] / 80.3 [77.6–83.0] | 199.5 [166.5–232.5] | 224.5 [215.1–233.9] | 5/6 | together |
| 32K | warm read | 7.40 (4,430 tok/s) | 87.8 | - | - | - | - |
| 32K | 1 | 0.26 [0.25–0.27] | 93.6 [93.3–94.0] | 93.6 [93.3–94.0] | 93.6 [93.3–94.0] | 2/2 | together |
| 32K | 2 | 7.82 [0.29–15.36] / 0.32 [0.30–0.34] | 87.3 [85.7–88.8] / 69.2 [49.2–89.2] | 126.6 [75.8–177.5] | 156.5 [134.9–178.0] | 3/4 | together |
| 32K | 3 | 8.11 [0.39–15.82] / 0.29 [0.28–0.29] / 0.34 [0.34–0.35] | 79.4 [78.4–80.4] / 62.1 [46.4–77.8] / 62.3 [46.5–78.1] | 170.8 [108.6–233.1] | 203.8 [173.3–234.2] | 5/6 | together |
| 64K | warm read | 14.11 (4,644 tok/s) | 86.7 | - | - | - | - |
| 64K | 1 | 0.32 [0.31–0.33] | 82.9 [82.4–83.3] | 82.9 [82.4–83.3] | 82.9 [82.4–83.3] | 2/2 | together |
| 64K | 2 | 0.34 [0.30–0.38] / 14.23 [0.40–28.05] | 55.3 [32.5–78.2] / 82.2 [78.8–85.6] | 103.9 [51.6–156.3] | 137.5 [118.0–156.9] | 3/4 | together |
| 64K | 3 | 0.41 [0.32–0.51] / 14.63 [0.33–28.93] / 0.43 [0.42–0.44] | 50.3 [31.1–69.6] / 75.4 [68.8–82.0] / 50.2 [31.2–69.3] | 140.4 [74.7–206.2] | 175.9 [144.2–207.7] | 5/6 | together |
| 128K | warm read | 28.40 (4,615 tok/s) | 83.9 | - | - | - | - |
| 128K | 1 | 0.42 [0.41–0.43] | 89.7 [89.4–90.1] | 89.7 [89.4–90.1] | 89.7 [89.4–90.1] | 2/2 | together |
| 128K | 2 | 23.91 [0.48–47.33] / 0.56 [0.54–0.57] | 84.3 [83.8–84.8] / 56.9 [28.5–85.3] | 102.1 [34.7–169.5] | 141.2 [112.3–170.1] | 3/4 | together |
| 128K | 3 | 24.48 [0.63–48.32] / 0.64 [0.55–0.73] / 0.59 [0.56–0.62] | 79.1 [74.4–83.8] / 51.3 [27.7–74.8] / 50.9 [27.8–73.9] | 136.3 [51.1–221.5] | 181.2 [139.3–223.2] | 5/6 | together |
| 256K | warm read | 58.74 (4,463 tok/s) | 84.7 | - | - | - | - |
| 256K | 1 | 0.61 [0.60–0.63] | 80.0 [79.6–80.5] | 80.0 [79.6–80.5] | 80.0 [79.6–80.5] | 2/2 | together |
| 256K | 2 | 0.83 [0.75–0.91] / 40.26 [0.78–79.75] | 50.7 [25.7–75.6] / 79.9 [75.1–84.7] | 86.2 [22.5–150.0] | 130.5 [110.4–150.7] | 3/4 | together |
| 256K | 3 | 40.98 [0.92–81.05] / 0.98 [0.89–1.07] / 0.89 [0.84–0.94] | 75.4 [66.1–84.8] / 45.8 [25.1–66.5] / 45.4 [25.1–65.8] | 115.0 [33.3–196.8] | 166.6 [134.9–198.3] | 5/6 | together |

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
| Highest card sample, kit 9S with the ten lines, the same rehearsal | 30,832 MiB (30,594 / 30,832 / 28,328 by stage); over the final bench and the prefix measure, 2 h 15 min, 30,418 | c2a, c3b, c7a, final bench |

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
| Seven cold long prompts against the capture of that boot (`quality-TC3-WOA4K.json`) | identical 7 of 7 on every boot since (the reference and candidate boots of c3b and c7a) | c3b, c7a |

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
  Predicted about 2.7 %: wrong, low. In the final bench its cold code reads were 4.8 to 5.0 % faster at 32K and longer.
- **The decode round is repeatable boot to boot** within 0.6 % (windows p3b and p3c), which is what makes a 0.5 ms
  line decidable; the reference's round at 128K moved by about 0.4 ms between p6 and p6b, the A/A band for that
  cell.
- **Two boots without the drafter give the same text** (window q1): eight forced cells at 4.2K to 252K tokens and
  after a turn, temperature zero, equal 8 of 8.
- **A kit's new log lines are collected before a window judges by them** (window p3l): one boot with every switch
  on, nothing judged, whose product is the real log; every later pattern met its real line first.
- **The final bench** (2026-11-19, the tables above, on the set every optimization left): one stream decodes 160 to
  178 tok/s on code and 80 to 100 on legal and prose from 1K to 256K of prompt; prefill 4,360 to 4,657 tok/s at 64K
  and longer on code, 3,809 to 4,550 on legal and prose; the first token of a 256K prompt after 60 to 64 s; three
  streams together 2.2 to 2.5 times one stream's decode once every prompt is in; on a cached document the first token
  in 0.42 to 0.43 s at 128K and three streams at 400 / 242 / 222 tok/s (code / legal / prose) over their window.

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
  library's switch that fixes the split count (`TM_GEMM_TUNE=max_splits=1`) was tried as arm G and ended
  INCONCLUSIVE at its quality line (window t1). It stays out.
- **A cached or restored turn is not token-identical to a cold read of the same text** (window st4). The rule line
  that asked for it was the wrong line and stays FAILED as written.
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
  about 2.7 was predicted; the packed experts' rounds were longer where shorter was predicted; the final bench's
  decode cells moved up to 8.6 % with the generated text where "within the band" was predicted. Each stays recorded
  as written.
- **Several streams do not read their prompts together.** With two or three requests released at once on prompts of
  8K and longer the reads interleave: three cold 128K code requests got their first tokens after 77.0, 87.4 and
  92.5 s where one alone takes 28.15 s, and a read that shares the cards with a decoding stream runs at roughly half
  its lone rate (both benches; the cause is not established). They decode together once every prompt is in.
- **The first cold read of a legal or prose document is slower than its second repeat**, by 1 to 11 s at 32K and
  longer, in both benches; code's repeats agree within 1.1 s. The one-stream legal and prose first-token cells are
  the mean of the two; the cause is not established.
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

- **Nothing of the plan's optimization list is being built.** Every item has ended: the batched projection adopted
  (window c2a); the conversation store, the lean pipeline hand-off and the packed expert records failed their lines
  (windows st7, c3b and c7a); the one-launch expert layer and the prompt-chunk gaps closed at their timing gates
  without a window. Their kits are in `kit/` as default-off switches (9Q, 9T, 9R, 9U); the store's kit stays private.
- **The Engram table's precision.** The build loads a 4-bit table of 103.0 GiB; the official one is 188.8 GiB and
  fits in host memory. INCONCLUSIVE in window q1 by construction; a reader for the official rows is needed first.
- **The restart after a power trip** serving what the record names.

## Next

The order of the remaining work: the Engram reader and its quality window; then tier B of
[documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md](documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md) as ranked (batched
decode on the pipeline, the split count and exact large sums, the snapshot hit tied to its own read; then tier D), each
window ending adopted or failed, with a rule written before its data and one change per boot; a closeness measure of
prefix hits; then the bench again and this page updated again. Power and clocks stay the owner's: the figures of the
day are 175 W figures; 200 W tripped the host once under four long chats.

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
| `docs/` | [architecture](docs/ARCHITECTURE.md), [operations](docs/OPERATIONS.md), [measurements](docs/measurements.md) (the [2026-11-18 bench](docs/measurements-2026-11-18.md) beside it), [history](docs/history.md), [upstream contributions](docs/upstream-contributions.md); [index](docs/README.md) |
| `documents/plans/` | the [kernel and optimization plan](documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md) |
| `documents/reviews/` | the [upstream review](documents/reviews/UPSTREAM-REVIEW-20261117.md), the [chunk-boundary reading](documents/reviews/CHUNK-BOUNDARY-READING.md) |
| `research/` | this build's own findings ([TWELVE-CARD-RESEARCH.md](research/TWELVE-CARD-RESEARCH.md)) and its [time budget](research/TWELVE-CARD-TIME-BUDGET.md) |
| `engine/` | [README.md](engine/README.md): every kit with its engine commit, what it is and its state |
| `kit/` | the engine changes as install kits (`apply.sh`, `rollback.sh`, `MANIFEST`, `files/`). `PP3KIT9S-over-cut` is the serving kit; `9Q`, `9T`, `9R` and `9U` are the kits of the two failed windows, default-off switches over 9S |
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
