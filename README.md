# Volta 12X AmPLIfy

DeepSeek-V4.1-Flash-1017 served on twelve Tesla V100-SXM2 32 GB cards from 2018, with every routed expert at
NVIDIA's NVFP4 precision on the cards. The model runs as three pipeline stages of four tensor-parallel cards. All
15,360 routed experts are held in card memory: none is compressed and none is read from host memory.

This is the build's record as of the evening of 2026-11-18 (its second release): the engine changes as install kits
through kit 9N, the measurements with their conditions, the closing bench of the serving set (code, legal and prose
prompts of 1K to 256K tokens, one to three streams), the architecture, the history of every test window, the kernel
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

**161 tokens a second** on a code request with 128,000 tokens of context, a 128,000-token file read in **30 seconds**
(4,444 tok/s), **four conversations** kept warm with **524,288 tokens** of context each. MEASURED 2026-11-18 on the
serving set (kit 9N, nine adopted switches), 175 W per card, 1,452 MHz clock cap, four slots, cold reads, thinking off,
1,024 tokens out, mean of two runs. Prose figures, min-max and every stream are in [Measured](#measured).

**Code** (a source file as the prompt)

| Context (prompt tokens) | Decode tok/s, one request | Prompt read tok/s | First token, cold | Three requests together, tok/s summed | First token, document already cached |
|---|---:|---:|---:|---:|---:|
| **1K** | 178 | 1,491 | 0.7 s | 432 | 0.2 s* |
| **8K** | 170 | 3,147 | 2.6 s | 413 | 0.2 s |
| **32K** | 173 | 4,178 | 7.8 s | 398 | 0.2 s |
| **64K** | 166 | 4,414 | 14.9 s | 379 | 0.3 s |
| **128K** | 161 | 4,444 | 29.5 s | 364 | 0.4 s |
| **256K** | 168 | 4,152 | 63.3 s | 368 | 0.7 s |

**Legal** (a brief as the prompt)

| Context (prompt tokens) | Decode tok/s, one request | Prompt read tok/s | First token, cold | Three requests together, tok/s summed | First token, document already cached |
|---|---:|---:|---:|---:|---:|
| **1K** | 99 | 1,383 | 0.8 s | 248 | 0.2 s* |
| **8K** | 102 | 2,920 | 2.8 s | 259 | 0.2 s |
| **32K** | 89 | 3,086 | 12.3 s | 209 | 0.3 s |
| **64K** | 94 | 3,746 | 18.1 s | 218 | 0.3 s |
| **128K** | 87 | 3,794 | 35.4 s | 205 | 0.5 s |
| **256K** | 94 | 4,076 | 64.4 s | 209 | 0.7 s |

\* At 1K the request's fixed cost is most of the read, so the cache cannot show.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/decode-by-context-dark.svg">
  <img alt="Decode speed by context length: one request and three together, code and legal" src="docs/img/decode-by-context-light.svg" width="880">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/first-token-dark.svg">
  <img alt="First token on a cold read against a document already cached, log scale, code and legal" src="docs/img/first-token-light.svg" width="880">
</picture>

- **Three requests together** is each stream's own decode rate while all three stream, summed: the throughput once every
  prompt is in. The pipeline carries one request per step, so long prompts are read one after another and the later
  streams wait for the earlier reads; counted over the whole joint window, three cold 128K requests give 139 tok/s on
  code and 109 on legal. With the document already in the cache the wait disappears: three code streams on one cached
  128K document decode 390 tok/s over the joint window (legal 254), within 2 % of the sum.
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
  four conversations kept warm and a context of 524,288 tokens. One request decodes at 161 to 178 tok/s on code and
  79 to 102 tok/s on legal and prose prompts of 1K to 256K tokens (MEASURED 2026-11-18 at 175 W, the closing bench
  below); the pipeline carries one request per step, so prompts are read one after another and three requests
  together decode at about 2.2 to 2.4 times one stream's rate once every prompt is in. It is not a batched-decode
  engine.
- **What it is tuned for.** Long-context agent work: one request at a time, prompts of 60,000 to 200,000 tokens that
  grow by small turns. Since 2026-11-17 it is the owner's default DeepSeek server.

## What runs

The standing configuration: the serving set since window p6b (2026-11-18 20:19Z), re-recorded after window q1 (20:48Z).

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
| Engine | 1Cat-vLLM's DeepSeek-V4.1 package as carried in PLI Labs' engine (commit `f20d155`) with kit 9N over it (engine `9733024`, 17 files): [`kit/PP3KIT9N-over-cut`](kit/PP3KIT9N-over-cut); every kit in [engine/README.md](engine/README.md) |
| Adopted switches | nine lines: `VLLM_V41_DRAFT_GEMV2=1 VLLM_V41_EMPTY_CACHE_TOKENS=8192 VLLM_V41_INDEX_TC=1 VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD=128 VLLM_V41_PP_ANSWER_NO_SLEEP=1 VLLM_V41_PP_EMPTY_AHEAD=1 VLLM_V41_PP_MID_PREFILL_SKIP=1 VLLM_V41_PP_TURN_CHUNKS=3 VLLM_V41_PREFIX_TOKEN_CACHE=1` (the window of each is named under What worked); the serving record carries their digest |
| Accepted capture | `accepted/quality-TC3.json` (window p5, turn chunks 3) while the turn chunks are adopted; window o2's capture otherwise; decided in one place from the adopted lines |
| Power and clock | 175 W per card and a 1,452 MHz clock ceiling (cards at 1,447 MHz) since 2026-11-18 12:09:23Z. Before: 150 W and 1,477 MHz to 2026-11-17 18:58Z; 200 W to 2026-11-18 11:15:48Z (the host tripped under four long chats); 150 W from 11:45Z |
| Serving | every start check runs on a test port and is recorded; the serving boot on the public port sends no request and needs a record that matches the installed bytes ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), section 13) |

Not in what serves: the conversation store (its whole server test list passed in window st4; its window on this set
has not yet run), the dense library's split switch (inconclusive in window t1), the drafter
off (the quality window's one-slot boots only), and the loop statistics `VLLM_V41_PP_LOOP_STATS` (a diagnostic of
the windows, never served).

## Measured

MEASURED on 2026-11-17 and 2026-11-18; each row names its window, and its power where it is not the window's standing
power (150 W through window sv1; 200 W from 2026-11-17 18:58Z; 175 W and the 1,452 MHz ceiling since 2026-11-18
12:09Z). The full tables with dates and times are in [docs/measurements.md](docs/measurements.md).

### The serving set since window p6b (2026-11-18, 175 W, clock ceiling 1,452 MHz)

Kit 9N with the nine adopted lines (table above). Every row MEASURED on 2026-11-18 at 175 W per card and the 1,452
MHz ceiling unless it says otherwise; each switch's own window is named.

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

**The closing bench, 2026-11-18 (the serving set of window q1).** MEASURED on 2026-11-18, 21:01:56Z to 22:36:32Z, on the serving set of window q1 (kit `PP3KIT9N-over-cut`, the nine adopted lines, four
slots), 175 W per card and the 1,452 MHz clock ceiling: code, legal and prose prompts of 1,024 / 8,192 / 32,768 / 65,536 / 131,072 /
262,144 tokens, read cold, one, two and three streams released together, two repeats each (108 groups, 216 measured requests; greedy,
1,024 tokens out, thinking off). "Prefill tok/s" is the prompt's tokens over the time to its first token. The two aggregate columns:
**Total decode tok/s** is all streams' output tokens over the joint window, from the first request's start to the last request's end;
the prompts are read one after another, so with long prompts it stays near one stream's rate. **Sum of decode rates** adds each
stream's own decode rate while all of them stream: the decode throughput once every prompt is in.

#### Code

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | cold, thinking off, 1,024 out. Mean [min–max], two runs.

| Prompt | Streams | TTFT s by stream | Prefill tok/s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates |
|---|---:|---|---|---|---:|---:|
| 1K | 1 | 0.69 [0.64–0.74] | 1491 [1386–1597] | 178.4 [178.4–178.5] | 178.4 [178.4–178.5] | 178.4 [178.4–178.5] |
| 1K | 2 | 1.02 [0.66–1.38] / 0.82 [0.66–0.98] | 1142 [743–1540] / 1300 [1044–1555] | 165.9 [162.0–169.7] / 161.3 [152.5–170.1] | 313.3 [303.3–323.3] | 327.1 [322.2–332.1] |
| 1K | 3 | 0.94 [0.66–1.22] / 0.95 [0.94–0.95] / 0.95 [0.67–1.23] | 1193 [837–1549] / 1083 [1079–1087] / 1187 [836–1539] | 144.0 [138.4–149.6] / 143.8 [143.7–144.0] / 144.0 [138.6–149.4] | 414.5 [414.1–414.8] | 431.8 [431.5–432.2] |
| 8K | 1 | 2.61 [2.50–2.71] | 3147 [3023–3270] | 170.3 [167.7–173.0] | 170.3 [167.7–173.0] | 170.3 [167.7–173.0] |
| 8K | 2 | 8.05 [8.05–8.05] / 2.53 [2.53–2.53] | 1018 [1018–1018] / 3237 [3236–3239] | 168.5 [168.3–168.8] / 121.4 [121.3–121.4] | 176.6 [176.5–176.7] | 289.9 [289.6–290.2] |
| 8K | 3 | 8.83 [8.83–8.83] / 2.52 [2.52–2.53] / 13.62 [13.61–13.62] | 928 [928–928] / 3247 [3238–3255] / 602 [601–602] | 132.9 [132.8–132.9] / 111.1 [111.1–111.1] / 168.9 [168.6–169.1] | 178.9 [178.8–179.0] | 412.8 [412.5–413.1] |
| 32K | 1 | 7.85 [7.55–8.16] | 4178 [4015–4341] | 172.8 [169.6–176.0] | 172.8 [169.6–176.0] | 172.8 [169.6–176.0] |
| 32K | 2 | 15.77 [12.49–19.06] / 15.77 [12.47–19.08] | 2171 [1719–2623] / 2173 [1718–2628] | 144.2 [115.8–172.6] / 144.2 [115.9–172.5] | 163.5 [163.5–163.5] | 288.4 [288.3–288.5] |
| 32K | 3 | 28.68 [26.28–31.08] / 16.58 [16.58–16.59] / 28.68 [26.27–31.09] | 1151 [1054–1247] / 1976 [1975–1977] / 1151 [1054–1247] | 153.0 [134.9–171.1] / 92.0 [92.0–92.0] / 153.2 [134.7–171.6] | 149.9 [149.8–149.9] | 398.2 [397.9–398.5] |
| 64K | 1 | 14.86 [14.46–15.25] | 4414 [4296–4532] | 166.1 [164.1–168.1] | 166.1 [164.1–168.1] | 166.1 [164.1–168.1] |
| 64K | 2 | 32.95 [32.91–32.99] / 26.21 [26.17–26.24] | 1989 [1987–1992] / 2501 [2497–2504] | 164.2 [163.7–164.8] / 110.6 [110.5–110.8] | 157.7 [157.5–157.9] | 274.9 [274.2–275.6] |
| 64K | 3 | 44.51 [37.04–51.97] / 44.51 [37.08–51.94] / 47.03 [47.02–47.04] | 1515 [1261–1769] / 1514 [1262–1767] / 1394 [1393–1394] | 125.6 [88.0–163.2] / 125.7 [88.0–163.3] / 128.1 [128.1–128.1] | 144.9 [144.9–145.0] | 379.4 [379.3–379.4] |
| 128K | 1 | 29.50 [29.04–29.96] | 4444 [4375–4513] | 161.1 [159.4–162.8] | 161.1 [159.4–162.8] | 161.1 [159.4–162.8] |
| 128K | 2 | 55.12 [55.08–55.15] / 62.21 [62.18–62.23] | 2378 [2377–2380] / 2107 [2106–2108] | 105.5 [105.3–105.6] / 158.4 [158.4–158.5] | 151.2 [151.1–151.3] | 263.9 [263.8–264.0] |
| 128K | 3 | 88.11 [80.26–95.96] / 93.38 [90.79–95.97] / 85.53 [80.29–90.78] | 1499 [1366–1633] / 1405 [1366–1444] / 1538 [1444–1633] | 120.9 [83.6–158.2] / 140.1 [122.9–157.4] / 103.1 [83.8–122.4] | 138.6 [138.4–138.8] | 364.2 [363.4–364.9] |
| 256K | 1 | 63.30 [60.05–66.56] | 4152 [3938–4366] | 168.2 [166.2–170.1] | 168.2 [166.2–170.1] | 168.2 [166.2–170.1] |
| 256K | 2 | 116.64 [116.62–116.67] / 124.35 [124.29–124.41] | 2247 [2247–2248] / 2108 [2107–2109] | 104.1 [104.0–104.2] / 166.4 [165.9–166.8] | 147.7 [147.5–148.0] | 270.5 [270.0–271.0] |
| 256K | 3 | 186.58 [183.82–189.35] / 172.37 [172.33–172.41] / 186.55 [183.73–189.37] | 1405 [1384–1426] / 1521 [1520–1521] / 1406 [1384–1427] | 144.1 [123.3–164.9] / 80.8 [80.7–81.0] / 143.6 [122.5–164.7] | 132.0 [131.9–132.1] | 368.5 [368.2–368.9] |

#### Legal

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | cold, thinking off, 1,024 out. Mean [min–max], two runs.

| Prompt | Streams | TTFT s by stream | Prefill tok/s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates |
|---|---:|---|---|---|---:|---:|
| 1K | 1 | 0.75 [0.65–0.85] | 1383 [1202–1564] | 98.7 [97.3–100.1] | 98.7 [97.3–100.1] | 98.7 [97.3–100.1] |
| 1K | 2 | 0.81 [0.67–0.96] / 0.82 [0.68–0.95] | 1303 [1071–1535] / 1292 [1076–1508] | 94.4 [93.4–95.3] / 94.4 [93.0–95.8] | 186.2 [185.8–186.6] | 188.8 [188.3–189.2] |
| 1K | 3 | 0.96 [0.68–1.24] / 0.96 [0.68–1.25] / 0.97 [0.96–0.97] | 1168 [824–1512] / 1166 [818–1513] / 1060 [1054–1067] | 82.5 [80.7–84.3] / 82.5 [80.6–84.4] / 82.5 [82.4–82.5] | 241.6 [241.6–241.7] | 247.5 [247.4–247.6] |
| 8K | 1 | 2.83 [2.56–3.11] | 2920 [2638–3203] | 102.3 [93.7–110.8] | 102.3 [93.7–110.8] | 102.3 [93.7–110.8] |
| 8K | 2 | 5.35 [2.57–8.12] / 5.35 [2.58–8.11] | 2096 [1009–3182] / 2093 [1010–3175] | 96.3 [85.6–106.9] / 96.4 [85.4–107.3] | 135.6 [135.4–135.7] | 192.6 [192.4–192.9] |
| 8K | 3 | 8.96 [8.95–8.97] / 13.78 [13.77–13.79] / 2.59 [2.59–2.59] | 914 [913–915] / 595 [594–595] / 3167 [3166–3168] | 87.4 [87.3–87.5] / 102.3 [102.1–102.6] / 69.2 [69.1–69.2] | 144.8 [144.7–144.9] | 258.9 [258.6–259.2] |
| 32K | 1 | 12.29 [7.75–16.83] | 3086 [1947–4226] | 89.4 [88.3–90.4] | 89.4 [88.3–90.4] | 89.4 [88.3–90.4] |
| 32K | 2 | 16.14 [12.85–19.43] / 16.17 [12.86–19.49] | 2118 [1686–2550] / 2115 [1681–2549] | 78.5 [69.4–87.5] / 78.4 [69.9–87.0] | 111.6 [111.2–112.0] | 156.9 [156.4–157.4] |
| 32K | 3 | 29.30 [26.87–31.74] / 24.44 [17.11–31.77] / 21.98 [17.10–26.85] | 1126 [1032–1220] / 1473 [1031–1915] / 1568 [1220–1916] | 77.9 [72.8–82.9] / 68.4 [53.5–83.3] / 63.0 [53.4–72.7] | 113.8 [113.8–113.8] | 209.3 [209.1–209.5] |
| 64K | 1 | 18.06 [14.87–21.26] | 3746 [3083–4409] | 94.2 [93.5–95.0] | 94.2 [93.5–95.0] | 94.2 [93.5–95.0] |
| 64K | 2 | 30.32 [26.90–33.74] / 30.32 [26.95–33.69] | 2189 [1942–2436] / 2189 [1945–2432] | 81.7 [71.9–91.5] / 81.7 [71.6–91.9] | 114.0 [113.8–114.2] | 163.4 [163.1–163.8] |
| 64K | 3 | 43.16 [38.15–48.18] / 45.71 [38.19–53.23] / 50.70 [48.18–53.22] | 1539 [1360–1718] / 1474 [1231–1716] / 1296 [1231–1360] | 65.1 [54.4–75.8] / 71.2 [54.5–87.8] / 81.7 [75.7–87.7] | 114.9 [114.9–115.0] | 218.0 [217.9–218.1] |
| 128K | 1 | 35.42 [29.85–40.99] | 3794 [3197–4391] | 86.9 [83.6–90.3] | 86.9 [83.6–90.3] | 86.9 [83.6–90.3] |
| 128K | 2 | 60.18 [56.72–63.65] / 60.20 [56.57–63.82] | 2185 [2059–2311] / 2185 [2054–2317] | 77.2 [67.7–86.7] / 77.2 [67.7–86.6] | 108.3 [108.3–108.4] | 154.4 [154.3–154.5] |
| 128K | 3 | 95.81 [93.20–98.42] / 87.91 [82.65–93.18] / 90.55 [82.60–98.50] | 1369 [1332–1406] / 1496 [1407–1586] / 1459 [1331–1587] | 76.8 [71.0–82.5] / 61.0 [50.9–71.1] / 66.8 [50.9–82.6] | 108.7 [108.6–108.7] | 204.6 [204.6–204.6] |
| 256K | 1 | 64.44 [61.60–67.28] | 4076 [3897–4256] | 94.4 [94.2–94.5] | 94.4 [94.2–94.5] | 94.4 [94.2–94.5] |
| 256K | 2 | 119.81 [119.75–119.87] / 127.47 [127.40–127.53] | 2188 [2187–2189] / 2057 [2056–2058] | 68.2 [68.1–68.3] / 90.2 [90.0–90.3] | 107.8 [107.7–107.9] | 158.3 [158.1–158.5] |
| 256K | 3 | 176.53 [176.42–176.64] / 190.80 [187.73–193.88] / 190.76 [188.03–193.49] | 1485 [1484–1486] / 1374 [1352–1396] / 1374 [1355–1394] | 50.2 [49.9–50.5] / 79.5 [72.7–86.2] / 79.2 [71.8–86.6] | 105.9 [105.4–106.4] | 208.9 [207.9–209.9] |

#### Prose

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | cold, thinking off, 1,024 out. Mean [min–max], two runs.

| Prompt | Streams | TTFT s by stream | Prefill tok/s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates |
|---|---:|---|---|---|---:|---:|
| 1K | 1 | 0.66 [0.64–0.67] | 1559 [1520–1598] | 84.3 [83.0–85.7] | 84.3 [83.0–85.7] | 84.3 [83.0–85.7] |
| 1K | 2 | 0.80 [0.65–0.94] / 0.80 [0.66–0.93] | 1331 [1091–1571] / 1324 [1106–1541] | 80.7 [79.9–81.5] / 80.7 [79.8–81.6] | 159.6 [159.4–159.7] | 161.4 [161.2–161.5] |
| 1K | 3 | 1.08 [0.95–1.21] / 0.66 [0.66–0.66] / 1.08 [0.94–1.23] | 962 [843–1080] / 1554 [1549–1558] / 962 [836–1088] | 71.7 [71.2–72.3] / 69.7 [69.6–69.9] / 71.8 [70.9–72.6] | 209.0 [208.6–209.4] | 213.3 [212.8–213.7] |
| 8K | 1 | 2.80 [2.53–3.07] | 2953 [2665–3242] | 94.1 [92.9–95.4] | 94.1 [92.9–95.4] | 94.1 [92.9–95.4] |
| 8K | 2 | 2.53 [2.52–2.54] / 8.04 [8.03–8.06] | 3239 [3231–3247] / 1018 [1016–1021] | 75.6 [75.5–75.6] / 91.9 [91.9–92.0] | 123.0 [122.9–123.0] | 167.5 [167.5–167.5] |
| 8K | 3 | 2.53 [2.53–2.53] / 13.69 [13.69–13.69] / 8.86 [8.86–8.87] | 3237 [3235–3239] / 598 [598–598] / 924 [924–924] | 62.0 [62.0–62.0] / 87.4 [87.3–87.6] / 76.2 [76.1–76.3] | 134.3 [134.1–134.4] | 225.7 [225.4–225.9] |
| 32K | 1 | 8.92 [7.64–10.20] | 3751 [3213–4288] | 90.1 [89.1–91.1] | 90.1 [89.1–91.1] | 90.1 [89.1–91.1] |
| 32K | 2 | 15.92 [12.64–19.20] / 15.92 [12.61–19.23] | 2150 [1707–2593] / 2151 [1704–2598] | 79.1 [70.2–87.9] / 79.1 [70.2–88.0] | 112.3 [112.3–112.3] | 158.2 [158.1–158.2] |
| 32K | 3 | 21.65 [16.78–26.52] / 21.67 [16.79–26.56] / 31.45 [31.44–31.46] | 1594 [1235–1953] / 1593 [1234–1951] / 1042 [1041–1042] | 63.6 [53.8–73.5] / 63.7 [53.9–73.5] / 84.2 [84.2–84.2] | 114.4 [114.3–114.5] | 211.5 [211.5–211.6] |
| 64K | 1 | 18.73 [14.69–22.76] | 3670 [2879–4461] | 84.6 [80.9–88.4] | 84.6 [80.9–88.4] | 84.6 [80.9–88.4] |
| 64K | 2 | 29.95 [26.60–33.31] / 29.95 [26.56–33.34] | 2216 [1967–2464] / 2216 [1966–2467] | 76.8 [68.1–85.6] / 76.8 [68.1–85.5] | 109.4 [109.3–109.4] | 153.6 [153.6–153.7] |
| 64K | 3 | 45.05 [37.59–52.52] / 42.56 [37.54–47.57] / 50.06 [47.53–52.59] | 1496 [1248–1743] / 1562 [1378–1746] / 1313 [1246–1379] | 66.9 [52.2–81.6] / 61.6 [52.2–71.1] / 76.4 [71.3–81.5] | 111.4 [111.4–111.5] | 205.0 [204.8–205.1] |
| 128K | 1 | 32.82 [29.59–36.05] | 4033 [3636–4430] | 87.3 [86.6–87.9] | 87.3 [86.6–87.9] | 87.3 [86.6–87.9] |
| 128K | 2 | 56.18 [56.17–56.19] / 63.30 [63.28–63.33] | 2333 [2332–2333] / 2071 [2070–2071] | 66.4 [66.4–66.4] / 84.6 [84.6–84.7] | 106.5 [106.5–106.6] | 151.0 [151.0–151.1] |
| 128K | 3 | 89.74 [81.78–97.69] / 87.10 [81.82–92.37] / 95.00 [92.40–97.60] | 1472 [1342–1603] / 1510 [1419–1602] / 1381 [1343–1419] | 65.4 [50.4–80.5] / 59.9 [50.1–69.7] / 75.0 [69.4–80.5] | 107.5 [107.3–107.7] | 200.3 [200.0–200.7] |
| 256K | 1 | 63.10 [61.13–65.07] | 4159 [4029–4288] | 78.8 [77.5–80.0] | 78.8 [77.5–80.0] | 78.8 [77.5–80.0] |
| 256K | 2 | 122.51 [118.68–126.34] / 122.56 [118.66–126.46] | 2142 [2075–2209] / 2141 [2073–2209] | 67.9 [59.2–76.6] / 67.8 [59.8–75.8] | 96.7 [96.3–97.2] | 135.7 [135.0–136.5] |
| 256K | 3 | 186.63 [186.59–186.67] / 175.15 [175.12–175.18] / 192.41 [192.39–192.43] | 1405 [1404–1405] / 1497 [1496–1497] / 1362 [1362–1363] | 61.8 [61.8–61.8] / 44.7 [44.6–44.8] / 71.7 [71.6–71.8] | 97.3 [97.2–97.5] | 178.2 [178.1–178.3] |

**With prefix sharing: the same bench with the document already in the cache.** For each length and kind one cold
"warm read" put the document in the prefix cache; then one, two and three streams on that document, released together,
two repeats (126 groups, 216 measured requests; MEASURED 2026-11-18 22:38Z to 23:22Z on the same serving set, 175 W, the
1,452 MHz ceiling). A hit is a first token under a quarter of the warm read's (the server reports no cached-token count).
Hits 150 of 216: the 36 requests at 1K are not hits by that rule (first token 0.19 to 0.61 s against a 0.64 to 0.67 s
read: the fixed cost of a request, not a read), and 30 requests re-read the document: in every two- and three-stream
cell at 8K and longer, one stream of the first repeat read the whole document again (its first token 1.2 to 2.5 times a
lone cold read's, the cards shared with the other streams' decode), after which every stream hit. The document was read
once more the first time a second stream shared it and once more the first time a third did, never after (hypothesis:
the cache is kept per slot; not established by this measure). Mean [min–max] mixes the repeat with the re-read and the
repeat without; the upper figure of the two aggregate columns is the all-hit repeat. No group serialized. On a hit the
first token at 128K came in 0.40 to 0.47 s and at 256K in 0.66 s; three streams on one cached 128K document, all hits
(one run): code 395 tok/s summed and 390 over the joint window, legal 255 and 254, prose 230 and 228. The tokens of a hit
differ from a cold read's: 0 of 150 hits reproduced the warm read's token digest while all 30 re-reads did; a cached read
is another rounding of the same arithmetic, and the quality of answers on hits is not judged by this measure.

#### Code, with prefix sharing

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | document in the prefix cache, thinking off, 1,024 out. Mean [min–max], two runs; the warm read is the one cold read that put the document in the cache.

| Prompt | Streams | TTFT s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates | Hits | Overlap |
|---|---:|---|---|---:|---:|---|---|
| 1K | warm read | 0.67 (1,532 tok/s) | 178.9 | - | - | - | - |
| 1K | 1 | 0.21 [0.18–0.23] | 179.6 [177.4–181.8] | 179.6 [177.4–181.8] | 179.6 [177.4–181.8] | 0/2 | together |
| 1K | 2 | 0.49 [0.27–0.72] / 0.20 [0.19–0.21] | 171.3 [169.8–172.8] / 163.9 [156.5–171.4] | 327.1 [312.3–342.0] | 335.2 [326.3–344.2] | 0/4 | together |
| 1K | 3 | 0.55 [0.29–0.82] / 0.25 [0.21–0.30] / 0.30 [0.25–0.35] | 149.9 [149.6–150.2] / 143.9 [139.3–148.6] / 144.8 [138.5–151.2] | 429.3 [414.1–444.4] | 438.7 [427.3–450.0] | 0/6 | together |
| 8K | warm read | 2.55 (3,217 tok/s) | 172.7 | - | - | - | - |
| 8K | 1 | 0.21 [0.20–0.22] | 183.5 [181.2–185.9] | 183.5 [181.2–185.9] | 183.5 [181.2–185.9] | 2/2 | together |
| 8K | 2 | 0.25 [0.22–0.28] / 3.19 [0.22–6.17] | 149.8 [120.9–178.8] / 173.0 [169.0–177.1] | 262.0 [170.5–353.5] | 322.9 [289.9–355.9] | 3/4 | together |
| 8K | 3 | 3.40 [0.40–6.40] / 0.30 [0.26–0.34] / 0.29 [0.26–0.31] | 155.0 [151.6–158.5] / 131.7 [113.2–150.3] / 131.3 [113.8–148.8] | 344.3 [243.6–445.1] | 418.1 [385.5–450.7] | 5/6 | together |
| 32K | warm read | 7.61 (4,304 tok/s) | 176.4 | - | - | - | - |
| 32K | 1 | 0.25 [0.25–0.25] | 178.9 [176.6–181.2] | 178.9 [176.6–181.2] | 178.9 [176.6–181.2] | 2/2 | together |
| 32K | 2 | 0.28 [0.28–0.29] / 7.98 [0.35–15.61] | 118.0 [65.9–170.1] / 173.5 [171.6–175.4] | 218.1 [96.6–339.6] | 291.5 [241.3–341.6] | 3/4 | together |
| 32K | 3 | 8.23 [0.32–16.15] / 0.40 [0.32–0.47] / 0.39 [0.38–0.41] | 160.5 [146.2–174.7] / 106.5 [63.8–149.2] / 105.9 [64.1–147.8] | 289.4 [141.5–437.3] | 372.9 [302.6–443.2] | 5/6 | together |
| 64K | warm read | 14.49 (4,524 tok/s) | 168.9 | - | - | - | - |
| 64K | 1 | 0.31 [0.29–0.33] | 166.8 [164.8–168.7] | 166.8 [164.8–168.7] | 166.8 [164.8–168.7] | 2/2 | together |
| 64K | 2 | 0.32 [0.32–0.32] / 12.24 [0.44–24.05] | 108.9 [61.1–156.8] / 163.5 [159.2–167.8] | 190.9 [68.6–313.1] | 272.5 [228.9–316.0] | 3/4 | together |
| 64K | 3 | 0.41 [0.38–0.43] / 12.69 [0.65–24.73] / 0.39 [0.33–0.45] | 97.1 [59.0–135.3] / 153.6 [139.7–167.5] / 97.4 [58.7–136.1] | 252.2 [100.6–403.9] | 348.2 [285.2–411.1] | 5/6 | together |
| 128K | warm read | 29.12 (4,502 tok/s) | 162.7 | - | - | - | - |
| 128K | 1 | 0.42 [0.39–0.45] | 159.0 [157.3–160.7] | 159.0 [157.3–160.7] | 159.0 [157.3–160.7] | 2/2 | together |
| 128K | 2 | 0.50 [0.49–0.50] / 19.80 [0.58–39.03] | 105.2 [60.1–150.3] / 157.6 [152.1–163.1] | 173.0 [45.6–300.4] | 262.8 [223.1–302.5] | 3/4 | together |
| 128K | 3 | 20.21 [0.72–39.71] / 0.57 [0.53–0.60] / 0.58 [0.52–0.64] | 147.8 [132.7–162.9] / 94.3 [58.3–130.3] / 94.8 [58.1–131.6] | 228.7 [67.5–389.9] | 336.9 [279.2–394.6] | 5/6 | together |
| 256K | warm read | 60.29 (4,348 tok/s) | 170.5 | - | - | - | - |
| 256K | 1 | 0.66 [0.65–0.67] | 164.0 [162.2–165.7] | 164.0 [162.2–165.7] | 164.0 [162.2–165.7] | 2/2 | together |
| 256K | 2 | 35.34 [0.66–70.01] / 0.80 [0.78–0.82] | 162.4 [153.2–171.6] / 108.1 [61.0–155.2] | 166.5 [27.2–305.9] | 270.5 [232.5–308.4] | 3/4 | together |
| 256K | 3 | 0.97 [0.79–1.16] / 0.93 [0.71–1.15] / 36.14 [1.25–71.03] | 94.0 [59.0–129.1] / 95.6 [57.8–133.3] / 153.1 [135.2–171.1] | 212.0 [40.2–383.7] | 342.7 [287.9–397.5] | 5/6 | together |

#### Legal, with prefix sharing

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | document in the prefix cache, thinking off, 1,024 out. Mean [min–max], two runs; the warm read is the one cold read that put the document in the cache.

| Prompt | Streams | TTFT s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates | Hits | Overlap |
|---|---:|---|---|---:|---:|---|---|
| 1K | warm read | 0.66 (1,553 tok/s) | 100.5 | - | - | - | - |
| 1K | 1 | 0.19 [0.19–0.19] | 102.7 [102.6–102.7] | 102.7 [102.6–102.7] | 102.7 [102.6–102.7] | 0/2 | together |
| 1K | 2 | 0.51 [0.29–0.73] / 0.20 [0.19–0.22] | 96.4 [95.1–97.6] / 94.7 [92.4–97.1] | 187.6 [181.2–193.9] | 191.1 [187.5–194.7] | 0/4 | together |
| 1K | 3 | 0.21 [0.20–0.22] / 0.37 [0.33–0.40] / 0.61 [0.28–0.94] | 83.1 [80.9–85.3] / 84.0 [82.0–86.0] / 84.9 [84.1–85.7] | 246.8 [238.0–255.6] | 252.0 [247.0–257.1] | 0/6 | together |
| 8K | warm read | 2.59 (3,159 tok/s) | 110.9 | - | - | - | - |
| 8K | 1 | 0.22 [0.21–0.22] | 98.1 [97.7–98.5] | 98.1 [97.7–98.5] | 98.1 [97.7–98.5] | 2/2 | together |
| 8K | 2 | 0.27 [0.23–0.31] / 3.20 [0.24–6.17] | 84.0 [74.7–93.2] / 99.5 [92.7–106.2] | 158.3 [131.4–185.3] | 183.5 [181.0–186.0] | 3/4 | together |
| 8K | 3 | 0.28 [0.25–0.30] / 3.41 [0.36–6.46] / 0.28 [0.24–0.32] | 71.3 [65.7–76.9] / 84.0 [77.1–90.8] / 71.3 [65.9–76.6] | 202.6 [175.7–229.5] | 226.5 [222.4–230.6] | 5/6 | together |
| 32K | warm read | 7.81 (4,195 tok/s) | 90.4 | - | - | - | - |
| 32K | 1 | 0.26 [0.25–0.26] | 98.2 [97.9–98.5] | 98.2 [97.9–98.5] | 98.2 [97.9–98.5] | 2/2 | together |
| 32K | 2 | 0.31 [0.28–0.35] / 8.02 [0.27–15.77] | 71.1 [49.2–93.1] / 90.2 [88.0–92.5] | 130.1 [75.4–184.8] | 161.4 [137.2–185.5] | 3/4 | together |
| 32K | 3 | 0.33 [0.30–0.37] / 0.33 [0.31–0.35] / 8.32 [0.42–16.23] | 63.9 [46.8–81.0] / 64.0 [46.7–81.4] / 82.3 [81.7–83.0] | 175.7 [108.6–242.7] | 210.2 [176.4–244.1] | 5/6 | together |
| 64K | warm read | 14.87 (4,407 tok/s) | 95.3 | - | - | - | - |
| 64K | 1 | 0.31 [0.30–0.32] | 101.7 [101.3–102.1] | 101.7 [101.3–102.1] | 101.7 [101.3–102.1] | 2/2 | together |
| 64K | 2 | 0.36 [0.35–0.37] / 14.64 [0.43–28.86] | 65.0 [34.1–96.0] / 95.6 [94.5–96.6] | 121.9 [52.1–191.8] | 160.6 [128.6–192.5] | 3/4 | together |
| 64K | 3 | 0.43 [0.40–0.46] / 0.47 [0.41–0.54] / 15.09 [0.49–29.68] | 58.4 [33.1–83.6] / 58.7 [33.0–84.5] / 88.4 [84.1–92.8] | 163.3 [76.1–250.5] | 205.5 [158.8–252.2] | 5/6 | together |
| 128K | warm read | 29.90 (4,383 tok/s) | 89.5 | - | - | - | - |
| 128K | 1 | 0.47 [0.45–0.48] | 102.9 [102.4–103.3] | 102.9 [102.4–103.3] | 102.9 [102.4–103.3] | 2/2 | together |
| 128K | 2 | 0.50 [0.48–0.51] / 23.81 [0.58–47.03] | 63.7 [30.8–96.7] / 93.6 [89.8–97.3] | 114.3 [35.3–193.2] | 157.3 [120.6–194.0] | 3/4 | together |
| 128K | 3 | 24.18 [0.53–47.84] / 0.66 [0.61–0.71] / 0.56 [0.52–0.61] | 87.2 [84.7–89.7] / 57.9 [30.1–85.6] / 57.6 [30.0–85.1] | 153.0 [52.3–253.7] | 202.7 [149.9–255.4] | 5/6 | together |
| 256K | warm read | 61.63 (4,254 tok/s) | 94.4 | - | - | - | - |
| 256K | 1 | 0.66 [0.64–0.69] | 87.9 [84.3–91.5] | 87.9 [84.3–91.5] | 87.9 [84.3–91.5] | 2/2 | together |
| 256K | 2 | 0.80 [0.72–0.88] / 40.77 [0.81–80.72] | 56.9 [27.7–86.1] / 90.1 [85.4–94.7] | 96.6 [22.5–170.7] | 147.0 [122.4–171.5] | 3/4 | together |
| 256K | 3 | 1.06 [0.98–1.15] / 41.55 [0.87–82.22] / 1.10 [0.90–1.31] | 50.7 [27.0–74.5] / 83.8 [73.0–94.5] / 51.0 [26.9–75.1] | 125.7 [33.3–218.0] | 185.5 [148.3–222.7] | 5/6 | together |

#### Prose, with prefix sharing

MEASURED | 175 W per card | 1452 MHz clock cap | four slots | document in the prefix cache, thinking off, 1,024 out. Mean [min–max], two runs; the warm read is the one cold read that put the document in the cache.

| Prompt | Streams | TTFT s by stream | Decode tok/s by stream | Total decode tok/s | Sum of decode rates | Hits | Overlap |
|---|---:|---|---|---:|---:|---|---|
| 1K | warm read | 0.64 (1,591 tok/s) | 86.3 | - | - | - | - |
| 1K | 1 | 0.19 [0.19–0.19] | 87.7 [87.5–87.8] | 87.7 [87.5–87.8] | 87.7 [87.5–87.8] | 0/2 | together |
| 1K | 2 | 0.20 [0.19–0.20] / 0.61 [0.39–0.83] | 80.3 [78.6–82.0] / 82.3 [81.6–83.0] | 159.4 [155.3–163.5] | 162.6 [160.2–165.0] | 0/4 | together |
| 1K | 3 | 0.27 [0.27–0.27] / 0.21 [0.21–0.21] / 0.56 [0.33–0.78] | 70.1 [70.0–70.2] / 69.8 [69.7–69.9] / 71.1 [70.2–72.1] | 208.3 [207.7–208.9] | 211.0 [209.9–212.1] | 0/6 | together |
| 8K | warm read | 2.55 (3,208 tok/s) | 95.2 | - | - | - | - |
| 8K | 1 | 0.22 [0.20–0.23] | 94.7 [94.3–95.2] | 94.7 [94.3–95.2] | 94.7 [94.3–95.2] | 2/2 | together |
| 8K | 2 | 3.19 [0.24–6.13] / 0.26 [0.23–0.30] | 90.7 [89.6–91.8] / 81.4 [72.8–90.0] | 149.6 [120.0–179.1] | 172.1 [164.6–179.6] | 3/4 | together |
| 8K | 3 | 0.27 [0.23–0.30] / 3.40 [0.38–6.42] / 0.28 [0.24–0.32] | 69.3 [65.2–73.3] / 77.7 [74.0–81.4] / 69.3 [65.0–73.7] | 191.7 [163.7–219.7] | 216.3 [211.5–221.0] | 5/6 | together |
| 32K | warm read | 7.68 (4,268 tok/s) | 91.3 | - | - | - | - |
| 32K | 1 | 0.26 [0.26–0.26] | 88.3 [88.1–88.6] | 88.3 [88.1–88.6] | 88.3 [88.1–88.6] | 2/2 | together |
| 32K | 2 | 0.32 [0.31–0.33] / 7.99 [0.28–15.70] | 65.3 [46.8–83.7] / 85.7 [83.4–88.0] | 121.2 [75.8–166.7] | 151.0 [134.8–167.2] | 3/4 | together |
| 32K | 3 | 0.38 [0.37–0.39] / 8.25 [0.31–16.19] / 0.38 [0.33–0.43] | 58.9 [44.3–73.4] / 77.8 [73.2–82.4] / 59.0 [44.2–73.7] | 163.9 [108.5–219.2] | 195.6 [170.9–220.3] | 5/6 | together |
| 64K | warm read | 14.70 (4,457 tok/s) | 88.9 | - | - | - | - |
| 64K | 1 | 0.33 [0.33–0.33] | 86.2 [85.8–86.5] | 86.2 [85.8–86.5] | 86.2 [85.8–86.5] | 2/2 | together |
| 64K | 2 | 14.55 [0.34–28.76] / 0.40 [0.38–0.42] | 83.7 [79.8–87.5] / 56.2 [32.2–80.1] | 105.3 [51.1–159.5] | 139.9 [119.8–160.0] | 3/4 | together |
| 64K | 3 | 14.95 [0.36–29.55] / 0.42 [0.41–0.44] / 0.49 [0.48–0.50] | 77.9 [71.3–84.5] / 51.3 [31.1–71.6] / 51.5 [31.1–71.9] | 144.0 [74.4–213.6] | 180.7 [146.7–214.8] | 5/6 | together |
| 128K | warm read | 29.66 (4,419 tok/s) | 87.9 | - | - | - | - |
| 128K | 1 | 0.40 [0.40–0.41] | 92.4 [92.0–92.8] | 92.4 [92.0–92.8] | 92.4 [92.0–92.8] | 2/2 | together |
| 128K | 2 | 24.44 [0.47–48.41] / 0.53 [0.50–0.56] | 87.7 [87.2–88.1] / 57.8 [28.0–87.6] | 104.3 [34.4–174.2] | 145.4 [116.1–174.8] | 3/4 | together |
| 128K | 3 | 0.70 [0.65–0.75] / 25.02 [0.52–49.51] / 0.60 [0.57–0.63] | 52.2 [27.3–77.0] / 82.0 [76.1–87.9] / 51.9 [27.2–76.5] | 139.2 [50.7–227.7] | 186.0 [142.4–229.6] | 5/6 | together |
| 256K | warm read | 61.07 (4,293 tok/s) | 79.6 | - | - | - | - |
| 256K | 1 | 0.67 [0.63–0.71] | 88.2 [88.2–88.2] | 88.2 [88.2–88.2] | 88.2 [88.2–88.2] | 2/2 | together |
| 256K | 2 | 40.71 [0.82–80.60] / 0.92 [0.89–0.95] | 81.2 [79.7–82.6] / 55.4 [27.6–83.2] | 93.5 [22.1–165.0] | 136.5 [107.3–165.7] | 3/4 | together |
| 256K | 3 | 1.02 [1.00–1.05] / 0.85 [0.85–0.85] / 41.43 [0.97–81.89] | 49.7 [26.8–72.6] / 49.2 [26.8–71.6] / 75.9 [72.0–79.8] | 123.6 [32.7–214.4] | 174.8 [133.4–216.2] | 5/6 | together |


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
- **The decode round is repeatable boot to boot** within 0.6 % (windows p3b and p3c), which is what makes a 0.5 ms
  line decidable; the reference's round at 128K moved by about 0.4 ms between p6 and p6b, the A/A band for that
  cell.
- **Two boots without the drafter give the same text** (window q1): eight forced cells at 4.2K to 252K tokens and
  after a turn, temperature zero, equal 8 of 8.
- **A kit's new log lines are collected before a window judges by them** (window p3l): one boot with every switch
  on, nothing judged, whose product is the real log; every later pattern met its real line first.
- **The closing bench** (2026-11-18, the tables above): one stream decodes 161 to 178 tok/s on code and 79 to 102 on
  legal and prose from 1K to 256K of prompt; prefill 4,100 to 4,400 tok/s at 64K and longer on code, 3,700 to 4,200
  on legal and prose; the first token of a 256K prompt after 63 to 64 s; three streams together 2.2 to 2.4 times one
  stream's decode once every prompt is in.

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
  its legal and prose prefill fell under a band drawn from code cells. Each stays recorded as written.
- **Several streams do not read their prompts together.** With two or three requests released at once on prompts of
  8K and longer, the later request's first token comes after its own read plus the earlier one's, and a read that
  shares the cards with a decoding stream runs at roughly half its lone rate (closing bench; the cause is not
  established). The repeat-to-repeat spread of a cold read reached 2x on three legal and prose cells where code's
  were within 3 %: unexplained, recorded.
- **The Engram precision item is still open.** The quality window could not decide it (no verified local extraction
  of the official host-memory FP8 Engram table, revision 1bae0e10, exists in the trees): INCONCLUSIVE by
  construction, not a pass.
- **The store is still not in what serves.** Its kit over kit 9M applies unchanged over 9N; its window has not run.

## In progress

- **The conversation store** (a kit over kit 9M that applies unchanged over 9N): being built for its
  window on this set; the window has not yet run (ESTIMATE about 92 minutes without a server).
- **The lean pipeline hand-off** (`VLLM_V41_PP_LEAN_HANDOFF`, item C3 of the kernel plan; a kit over 9N): built and
  tested on CPU; window not yet run.
- **The batched `wo_a` projection at 4,092 rows** (item C2; a kit over 9N): built and tested on CPU; window not yet
  run.
- **The zero padding out of each native expert record** (item C7; a kit over 9N and a re-cut of the expert packs):
  built and tested on CPU on a slice; the re-cut and the window not yet run.
- **The native expert layer in one launch** (item C4): sized against the engine's own fused kernels on CPU and not
  built: no window.
- **The Engram table's precision.** The build loads a 4-bit table of 103.0 GiB; the official one is 188.8 GiB and
  fits in host memory. INCONCLUSIVE in window q1 by construction; a reader for the official rows is needed first.
- **The restart after a power trip** serving what the record names.

## Next

The order of the remaining work: every optimization left, each window ending adopted or failed, with a rule written
before its data and one change per boot (the store, then C3, C2 and C7 above, then tier B of
[documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md](documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md) as ranked:
batched decode on the pipeline, the split count and exact large sums, the snapshot hit tied to its own read; then
tier D); then the closing bench again on the final serving set and this page updated again. Power and clocks stay
the owner's: the figures of the day are 175 W figures; 200 W tripped the host once under four long chats.

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
| `docs/` | [architecture](docs/ARCHITECTURE.md), [operations](docs/OPERATIONS.md), [measurements](docs/measurements.md), [history](docs/history.md), [upstream contributions](docs/upstream-contributions.md); [index](docs/README.md) |
| `documents/plans/` | the [kernel and optimization plan](documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md) |
| `documents/reviews/` | the [upstream review](documents/reviews/UPSTREAM-REVIEW-20261117.md), the [chunk-boundary reading](documents/reviews/CHUNK-BOUNDARY-READING.md) |
| `research/` | this build's own findings ([TWELVE-CARD-RESEARCH.md](research/TWELVE-CARD-RESEARCH.md)) and its [time budget](research/TWELVE-CARD-TIME-BUDGET.md) |
| `engine/` | [README.md](engine/README.md): every kit with its engine commit, what it is and its state |
| `kit/` | the engine changes as install kits (`apply.sh`, `rollback.sh`, `MANIFEST`, `files/`). `PP3KIT9N-over-cut` is the serving kit |
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
