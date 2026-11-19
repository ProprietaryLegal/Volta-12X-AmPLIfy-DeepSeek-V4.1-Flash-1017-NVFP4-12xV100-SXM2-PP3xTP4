# Measurements

Every measured figure of the twelve-card build, with its window and conditions. The source of each figure is the
window's result entry in the build's plan (section 9; "Amendment N" names an entry; the plan is not in this
release) or the build's working log. The receipts are on the server under `<build>/logs/window/<TAG>/` and
`run/state/`. They are not in this repository: they hold generated text.

Labels: **MEASURED** = a log or probe on this hardware. **ARITHMETIC** = computed here from measured values.
**PREDICTED** = written in the plan before the window, shown beside the result. Nothing on this page is an
estimate unless it says so.

## Conditions

| Item | Value | Source |
|---|---|---|
| Host and cards | one host, twelve Tesla V100-SXM2 32 GB, 32,510 MiB per card | PLAN section 2 |
| Boards | X and Y behind a PEX88096 switch, P behind a PLX 8749; order `xyp` in every window | `run/swap_arm12.sh`, PLAN section 5 |
| Power and clock | 150 W and 1,477 MHz per card, MEASURED 2026-11-16 23:06Z to 23:14Z, through window sv1 (2026-11-17 18:16Z). **200 W** and 1,477 MHz from 2026-11-17 18:58Z (the limits service's log showed all twelve V100s "power limit set to 200W") to 2026-11-18 11:15:48Z, when the host ended under four long chats with no shutdown sequence and no kernel line (read as a power trip). **150 W** from 11:45Z. **175 W and a 1,452 MHz clock ceiling** (cards at 1,447 MHz) since 2026-11-18 12:09:23Z: every figure of windows p4 to c7a and of both benches | PLAN section 2; Amendments 28, 58 and 62 |
| Layout | three stages of four cards, layers 0-13 / 14-27 / 28-39 | PLAN section 1 |
| Dates | windows w1 to t1 ran on 2026-11-17 (UTC); p1 crossed midnight; p2s to q1 and the closing bench ran on 2026-11-18; st5 to c7a and the final bench on 2026-11-19; b4a, b3a and the second bench of that day (14:09Z to 16:24Z) on 2026-11-19 | PLAN section 9 |
| "The default" | in the window names below, the server that was restored at the end of a window before 2026-11-17 18:14Z; a window stopped it, ran its arms and restored it | PLAN section 9 |
| The line | 31,998 MiB: the owner's limit for the highest sample on any card | PLAN section 9, line 6 |

Every figure of a window up to and including sv1 was taken at 150 W (PLAN Amendment 28 says so of "every speed
figure of this build before 18:58Z"). The 200 W figures are marked "200 W" on this page: the serving build's
baseline after window g1, windows mv1, g1, t1, p1, p2s, p2, p3, p3l, p3b and p3c. Windows p4 to c7a and both
benches are 175 W figures; p3r's are 150 W. TODO(owner of the repo): the
clock cap of 1,477 MHz is recorded for 2026-11-16 and for 18:57Z on 2026-11-17; that it held in between is
inferred, not read.

Terms. **Fresh text**: the requests F1, F2, F3 of the owner's cells and prompts made the same way. **Repeated
text**: the copy tasks RA, RB, RC and prompts made the same way. **Arm X**: three stages with a quantized expert mix
(5,651 experts in NVFP4, the rest compressed), used to test the three-stage engine against a two-stage reference
boot of the same experts. **Arm F2**: every routed expert
NVFP4, two slots, 8,192-row prompt chunks. **Arm F3c**: every routed expert NVFP4, three slots, 4,096-row chunks:
the configuration of windows o2 and k1. **Arm F4c**: the same with four slots: the standing configuration since
window s4. **Arm G**: the standing configuration with the dense library's split switch `TM_GEMM_TUNE=max_splits=1`.
**Arm T**: the standing configuration with the engine's timers and the torch profiler (the time budget). **Boots
KA, KD1, KD2, KN, KN2**: the store windows' boots (store off; on with its dump on an empty root; the restart; on as
it would serve; its restart). **Reference N2**: an all-native reference computed with experts fetched from host memory (two stages on eight
cards; every expert native). **Closeness**: `tools/closeness.py` on twelve short prompts; "first divergence" is
the first token where two captures differ; "mean |dlogprob|" is the mean absolute difference of log-probabilities.
A first-run speed figure carries 2 to 3 ms of noise per round; the second pass is the reading (PLAN section 9,
line 5).

## Summary: the standing configuration (four slots)

Three stages, every routed expert NVFP4, four slots, 4,092-row prompt chunks (`max-num-batched-tokens 4096`), kit
6c, card limit 31,998 MiB, profile `run/serve-nvfp4-pp3-full-c4096-a4.yaml` (PLAN Amendment 19). Since window st3
the store's kit is in the trees with the store off; since 2026-11-17 18:14Z the configuration is the owner's default and
serves on the serving port (Amendment 27). The table below is window s4, arm F4c, one boot, at 150 W. The 200 W figures
follow it.

| What | Figure | Window |
|---|---|---|
| Experts | 15,360 hot, 0 cold on every rank | s4 |
| Load, launch to ready | 134 s | s4 |
| Card limit 31,998 MiB | on 12 of 12 workers; headroom under the allocator limits at the end of warm-up 3,203 / 2,985 / 4,503 MiB | s4 |
| Memory after warm-up, by stage | 28,795 / 29,013 / 27,495 MiB | s4 |
| Against the accepted long-prompt capture (window o2, arm F3c) | identical 7 of 7 | s4 |
| Each chat alone against four together | PASS | s4 |
| One request | 128.1 tok/s fresh text, 176.6 repeated text | s4 |
| Four requests together, sum | 306.6 tok/s fresh (60.5 to 85.5 each; first token median 0.35 s), 453.4 repeated (110.9 to 117.8 each). Three together in o2: 288.0 and 444.6 | s4 |
| Prompt read, one 62,263-token prompt | 3,284 tok/s (three slots: 3,286) | s4 |
| Highest card sample, four grown chats at once (4 chats, 30 requests, largest prompt 153,597 tokens) | 30,398 MiB (29,838 / 30,398 / 28,892); 0 out-of-memory lines | s4 |
| Room under the line, fullest (middle) stage | 1,600 MiB (1,300 MiB under the rule's ceiling of 31,698) | s4 |
| Four documents of 110,000 to 140,000 tokens read at once | first token after 129 to 166 s each (three at once in k1: 105 to 121 s) | s4 |
| First token of a turn with four grown chats open | under 2,048 new tokens: median 4.12 s; larger turns 9.33 s (three chats in k1: 4.73 and 8.16 s) | s4 |
| Short prompts against reference N2 | not run again at four slots; the three-slot figures below stand for the same engine and experts | o2 |

Not measured at four slots: a serving run of hours under a rule. The boots since s4 with the store's kit in the trees and the
store off (st3b KA, st4 KA, sv1) repeated the capture 7 of 7 and the four-chat peak within 34 MiB (30,364 to 30,398
MiB).

### The standing configuration at 200 W (one run each)

| What | 200 W | 150 W (s4) | Source |
|---|---|---|---|
| One request, fresh / repeated text | 129.7 / 179.2 tok/s | 128.1 / 176.6 | the serving arm P3g1-S, 19:27:45Z (PLAN Amendment 29) |
| Four together, fresh / repeated, sum | 312.1 / 447.5 tok/s | 306.6 / 453.4 | same |
| Prompt read, one 62,263-token prompt | 3,733 tok/s, first token 16.68 s | 3,284 tok/s, 18.96 s | same |
| Prompt read, the untimed 62,260-token read of arm T | 3,712 tok/s (99.4 % of 3,733) | | window t1 (Amendment 31) |
| One-request decode round, 300 tokens in 129 rounds | 25.1 ms (26.5 with the profiler) | not measured | window t1, arm T |

Reading +13.7 %, decode within 2 % (ARITHMETIC). Amendment 28 records the reasoning before the data: a decode round
keeps each card busy about a third of the time, a prompt read more.

## Summary: the serving set since window b4a (kit 9S, eleven switches; 175 W, 1,452 MHz ceiling)

The set since window c2a (below) plus the split count, `TM_GEMM_TUNE=max_splits=1`, the eleventh line of
`run/state/ADOPTED.env` (window b4a, 2026-11-19 12:30:04Z to 13:13:21Z): the dense multiplies' library takes one split
instead of searching. Its accepted capture is `accepted/quality-TC3-WOA4K-S1.json` (the ten-line capture kept beside it).
Serves on the serving port as P3b4a-S since 2026-11-19 13:13:21Z and, after window b3a (the exact four-card sum, failed its
closeness line, not adopted), as P3b3a-S since 14:08:19Z on the record SERVE-CHECKED of 13:10:50Z. Every figure below is
MEASURED on 2026-11-19 at 175 W per card and the 1,452 MHz clock ceiling. The bench and the prefix-sharing measure ran
on it from 14:09:20Z to 16:24:31Z without a restart; their tables are in the section Closing bench, 2026-11-19, second,
below (Amendment 92). The split count moves rounding: against the ten-line capture the served answers are identical 1 of 7,
median first divergence 45 to 48 tokens, median of the mean |dlogprob| 0.035 to 0.037 nats (window b4a; the yardstick
of window o2 is 50 and 0.0307), and every cell of the bench generated different text from the 05:53Z run's, so the
decode columns moved with the text (-10.1 to +9.0 %, mean -0.9 % over 54 cells) while the round itself did not (window
b4a: one fresh request +1.3 %).

| What | Figure | Window |
|---|---|---|
| Prefill, cold reads of 60,706 / 93,028 / 247,953 / 425,783 tokens, the only request, against the ten lines on the same boot kind | 4,659 / 4,756 / 4,552 / 4,314 tok/s against 4,664 / 4,747 / 4,515 / 4,327 (-0.1 / +0.2 / +0.8 / -0.3 %); first token 13.03 / 19.56 / 54.47 / 98.70 s | b4a |
| Decode at four slots, against the ten lines | one request fresh 137.3 against 135.6 tok/s (+1.3 %); one repeated and four together +4.7 and +4.3 % (the text moved; report only) | b4a |
| Closeness of the served answers to the ten-line capture | identical 1 of 7; median first divergence 45 (summary) / 48 (scaled); median mean \|dlogprob\| 0.0366 / 0.0350 nats; CLEAN (early wrong-token divergences 0 of 7) | b4a |
| A cache hit against its own cold read (nine prompts of the bench at 32K to 128K, greedy, 32 tokens; D11) | identical 4 of 9 (prose 3 of 3, code 64K); median first divergence 25, median mean \|dlogprob\| 0.0275; every resend a hit by timing (0.25 to 0.45 s against 7.2 to 28.6 s cold) | b3a (reference arm) |
| Highest card sample, four-chat rehearsal | 30,832 MiB; by stage 30,594 / 30,832 / 28,328 | b4a, b3a |
| The bench, one stream decode at 128K (two repeats of one text; the spread across texts of a kind is about ±9 %) | code 157.4, legal 93.1, prose 85.6 tok/s (1K: 183.8 / 101.5 / 86.1; 256K: 153.4 / 92.4 / 84.5) | bench 2026-11-19, second |
| The bench, one cold read, prefill at 128K and 256K | code 4,670 and 4,518 tok/s (first token 28.07 and 58.03 s); legal 4,538 and 4,422 (28.88 and 59.28 s); prose 4,586 and 4,456 (28.58 and 58.83 s); first and second repeats within 0.15 s in every cell (the slow first legal and prose read of the 05:53Z run, 1 to 11 s, did not recur) | bench 2026-11-19, second |
| The bench, three streams released together at 128K, sum of the streams' decode rates | code 360.0, legal 212.6, prose 198.3 tok/s (the earliest-first-token-to-last-end aggregate 138.0 / 110.8 / 106.9) | bench 2026-11-19, second |
| The prefix measure: first token on a hit at 128K; three streams on one cached 128K document, the all-hit repeat | 0.43 / 0.44 / 0.44 s (code / legal / prose); 432.7 / 232.0 / 222.5 tok/s summed (426.6 / 230.1 / 221.0 over the window) | measure 2026-11-19, second |
| Highest card sample over the bench and the measure (2 h 15 min, 1,513 samples) | 30,418 MiB | bench 2026-11-19, second |

The set before it (kit 9S, ten switches, 2026-11-19 03:55Z to 12:30Z) follows.

## Summary: the serving set since window c2a (kit 9S, ten switches; 175 W, 1,452 MHz ceiling)

Three stages, every routed expert NVFP4, four slots, 4,092-row prompt chunks, card limit 31,998 MiB, kit
`PP3KIT9S-over-cut` (kit 9N plus the batched 4,092-row weight-only projection in prefill), no store kit, and the ten
adopted lines of `run/state/ADOPTED.env`: `VLLM_V41_DRAFT_GEMV2=1 VLLM_V41_EMPTY_CACHE_TOKENS=8192 VLLM_V41_INDEX_TC=1
VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD=128 VLLM_V41_PP_ANSWER_NO_SLEEP=1 VLLM_V41_PP_EMPTY_AHEAD=1
VLLM_V41_PP_MID_PREFILL_SKIP=1 VLLM_V41_PP_TURN_CHUNKS=3 VLLM_V41_PREFIX_TOKEN_CACHE=1 VLLM_V41_WO_A_BATCHED_4K=1`; the
accepted capture `accepted/quality-TC3-WOA4K.json` (window c2a). Served on the serving port as P3c2a-S from 2026-11-19
03:55:51Z, then as the way back of each later window (P3c3b-S 05:02:08Z, P3c7a-S 05:39:07Z; record SERVE-CHECKED of
03:53:11Z), until window b4a replaced it with the eleven-line set at 12:30:04Z. Every figure below is MEASURED on 2026-11-19 at 175 W per card and the 1,452 MHz clock ceiling unless the
row says otherwise; the windows that measured each switch alone are in the section By window. This is the set every
optimization of Amendment 74 step 4 left: the store, the lean hand-off (C3) and the packed experts (C7) failed their
lines on it; C1 and C4 closed without a window. The final bench and the prefix-sharing measure ran on it from
05:53:12Z to 08:08:28Z; their tables are in the section Closing bench, 2026-11-19, below (Amendment 87).

| What | Figure | Window |
|---|---|---|
| Prefill, cold reads of 60,705 / 93,025 / 247,948 / 425,779 tokens, the only request | 4,663 / 4,741 / 4,531 / 4,309 tok/s (first token 13.02 / 19.62 / 54.72 / 98.81 s); the same boot without the batched projection 4,476 / 4,559 / 4,358 / 4,028 | c2a |
| Decode round at about 128K of context, one request (medians of the loop's steady lines) | L1 (a one-step turn of 880 on a 128,585-token chat) 25.895 ms; LM (after a cold read of 127,663 tokens) 25.588 ms (c7a's reference boot, kit 9U with its switch off, the same engine path as 9S); c3b's reference boot 26.253 / 25.954 | c7a, c3b |
| Decode round on a short chat, one request | S1 24.388 ms, SM 24.284 ms (c7a); c3b's reference boot 30.810 / 27.555 with lines spread 24.7 to 44.1 ms | c7a, c3b |
| Closeness of the served answers to the accepted capture of kit 9N (the batched projection moves rounding) | identical 1 of 7, median first divergence 58 tokens, median of the mean \|dlogprob\| 0.0354 nats | c2a |
| Highest card sample, four-chat rehearsal (30 requests, largest prompt about 153,600 tokens) | 30,832 MiB (c2a, c7a), 30,830 (c3b, servecheck); by stage 30,594 / 30,832 / 28,328 | c2a, c3b, c7a |
| Load, launch to ready | 164 to 197 s | st7, c3a |
| The conversation store on this set's predecessor (kit 9N, nine lines): first token of the larger turns in a four-chat rehearsal | 4.718 and 3.722 s with the store on against 3.596 and 3.590 off (limit 3.696): FAIL; every state check PASS | st7 |
| The final bench, one stream decode at 128K (two repeats of one text; the spread across texts of a kind is about ±9 %) | code 175.0, legal 94.2, prose 79.9 tok/s (1K: 178.3 / 97.3 / 85.6; 256K: 168.5 / 90.0 / 84.4) | bench 2026-11-19 |
| The final bench, one cold read, prefill at 128K and 256K | code 4,657 and 4,360 tok/s (first token 28.15 and 60.23 s); legal 3,944 and 4,094 (34.09 and 64.46 s, means of a slow first repeat and a fast second: 28.70 and 59.23 s on the second); prose 4,331 and 4,247 (30.40 and 61.89 s; second repeats 28.37 and 58.79) | bench 2026-11-19 |
| The final bench, three streams released together at 128K, sum of the streams' decode rates | code 391.5, legal 216.1, prose 192.3 tok/s (the earliest-first-token-to-last-end aggregate 142.9 / 112.3 / 105.7) | bench 2026-11-19 |
| The prefix measure again: first token on a hit at 128K; three streams on one cached 128K document, the all-hit repeat | 0.42 to 0.43 s; code 399.8, legal 242.2, prose 221.5 tok/s over the window (405.1 / 244.0 / 223.2 summed) | measure 2026-11-19 |
| Highest card sample over the bench and the measure (2 h 15 min, 1,522 samples) | 30,418 MiB | bench 2026-11-19 |

The set before it (kit 9N, nine switches, 2026-11-18 20:19Z to 2026-11-19 03:55Z) follows.


## Summary: the serving set since window p6b (kit 9N, nine switches; 175 W, 1,452 MHz ceiling)

Three stages, every routed expert NVFP4, four slots, 4,092-row prompt chunks, card limit 31,998 MiB, kit
`PP3KIT9N-over-cut`, no store kit, and the nine adopted lines of `run/state/ADOPTED.env`: `VLLM_V41_DRAFT_GEMV2=1
VLLM_V41_EMPTY_CACHE_TOKENS=8192 VLLM_V41_INDEX_TC=1 VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD=128
VLLM_V41_PP_ANSWER_NO_SLEEP=1 VLLM_V41_PP_EMPTY_AHEAD=1 VLLM_V41_PP_MID_PREFILL_SKIP=1 VLLM_V41_PP_TURN_CHUNKS=3
VLLM_V41_PREFIX_TOKEN_CACHE=1`; the accepted capture `accepted/quality-TC3.json` (window p5, turn chunks 3). Serves
on the serving port as P3p6b-S since 2026-11-18 20:19:16Z and as P3q1-S since 20:48:45Z (record SERVE-CHECKED of
20:16:46Z). Every figure below is MEASURED on 2026-11-18 at 175 W per card and the 1,452 MHz clock ceiling (cards
at 1,447 MHz) unless the row says otherwise; the windows that measured each switch alone are in the section By window.
The closing bench of Amendment 74 (code, legal, prose; one, two and three streams; 1K to 256K) ran on this set from
21:01:56Z to 22:36:32Z; its tables are in "Closing bench, 2026-11-18" below (Amendment 77).

| What | Figure | Window |
|---|---|---|
| Decode round at about 128K of context, one request (medians of the loop's own steady lines) | L1 (a one-step turn of 880 on a 128,585-token chat) 25.829 ms; LM (after a cold read of 127,663 tokens) 25.719 ms; the reference boot without the tensor-core scorer 26.633 / 26.677 | p6b |
| Decode round on a short chat, one request | S1 24.375 ms (one-step read), SM 24.381 (many-step read) | p6b |
| Per token at 128K, forced 1,000-token sequences (not an answer speed) | L1 7.528 ms, LM 6.862 ms (the reference boot 7.882 / 7.090; -4.50 % / -3.17 % repeat for repeat) | p6b |
| Per token at 128K, the TC boot of p6 (same kit, the window that went back) | 6.88 ms after a cold read (145.4 tok/s), 7.63 after a one-step turn (131.1 tok/s); the reference 7.27 / 7.99 | p6 |
| Prefill, one cold read of about 127,663 tokens | 4,587 tok/s (TC boot) against 4,582 (reference) | p6b |
| Prefill, one cold read of 425,784 tokens on the serving port, the only request | first token 102.73 s, 4,145 tok/s; high 30,392 MiB on p4b's set (104.46 s, 4,076) | p5u, p4b |
| Four requests together, fresh text, sum | 337.9 tok/s (reference boot 316.6); repeated text 451.6 (449.1); none starved | p6b |
| A turn alone on a 128,000-token chat, first token (medians of two) with the turn chunks | +880 new tokens 1.131 s, +2,048 1.471 s, +4,200 2.319 s, +10,000 3.623 s (without the chunks 1.184 / 2.061 / 3.687 / 4.963) | p5 |
| A turn of 880 tokens beside another chat's 248,000-token read, with the hold (medians of three) | the turn's first token 3.26 s, whole turn 5.12 s (11.99 without the hold); the read's first token 60.99 s (62.66) | p5 |
| The front of a turn on a 128,000-token chat with the token cache (template, tokenizing, hashing) | 47.2 ms against 308.0 (median of 20 turns, cell for cell -256.8 ms) | p4b |
| Two boots without the drafter, eight forced cells at 4.2K / 94K / 128K / 252K and after a turn, temperature zero | text equal 8 of 8 | q1 |
| Highest card sample, four-chat rehearsal (30 requests, largest prompt about 153,600 tokens) | 30,814 MiB (p5t, p5u, p6b); the one-slot no-drafter boots 30,020 | p5u, p6b, q1 |
| Load, launch to ready | 134 s | p5 |

The 200 W figures of the day (windows p2, p3, p3b, p3c, before the power trip of 11:15:48Z) and the 150 W figures
of p3r are in the section By window with their power. The decode switches were judged at 200 W; the scheduler switches, the
token cache and the tensor-core scorer at 175 W.


## Summary: three slots (the configuration of windows o2 and k1)

Three stages, every routed expert NVFP4, three slots, 4,096-row prompt chunks, kit 6c, card limit 31,998 MiB,
profile `run/serve-nvfp4-pp3-full-c4096-a3.yaml`. It was the standing configuration from Amendment 13 to
Amendment 19 and is still served by `SERVE_CONFIG=F3c`.

| What | Figure | Window |
|---|---|---|
| Experts | 15,360 hot, 0 cold on every rank | o2 and every boot since |
| Load, launch to ready | 134 to 135 s (about 37 s process start, 86 s model load, the rest graphs and warm-up) | o2, k1 |
| One request | 129.5 tok/s fresh text, 177.3 repeated text | o2, arm F3c |
| Three requests together, sum | 288.0 tok/s fresh, 444.6 repeated | o2, arm F3c |
| Prompt read, one 62,257-token prompt | 3,286 tok/s (96 % of the two-slot arm's 3,410 with 8,188-row chunks) | o2 |
| First token, turn under 2,048 new tokens on a 130,000 to 150,000-token chat | median 1.52 s (arm F3c) and 1.28 s (serving boot) | o2 |
| First token, turn of 4,313 or 5,622 new tokens on such a chat | 3.4 to 9.0 s | o2 |
| First token of a long chat's turn after another chat was read | 2.6 s and 2.1 s at three slots; 46.7 s at two slots (138,824 tokens read again) | o2 |
| Memory after warm-up, by stage | 28,535 / 28,667 / 27,253 MiB; outside the allocator 867 / 881 / 865 | o2 |
| Highest card sample, two grown chats, cells and long reads | 30,052 MiB (29,578 / 30,052 / 28,646); serving boot 30,048 (29,540 / 30,048 / 28,610) | o2 |
| Highest card sample, three grown chats at once | 30,054 MiB under the limit (29,576 / 30,054 / 28,654); 30,052 without it (29,576 / 30,052 / 28,632) | k1 |
| Three chats of 120,000 to 140,000 tokens read at once | first token after 105 to 121 s each | k1 |
| First token of a turn with three such chats open | under 2,048 new tokens: median 4.73 s and 5.40 s; larger turns 8.16 s and 6.67 s (two boots) | k1 |
| Card limit 31,998 MiB | on 12 of 12 workers; allocator limits 31,131 / 31,117 / 31,133 MiB; headroom at the end of warm-up 3,463 / 3,331 / 4,745 MiB | k1 |
| Room under the line, fullest (middle) stage | 1,946 MiB | o2, k1 |
| Serving boot against the accepted long-prompt capture | identical 7 of 7, with the limit and without it; again in st1 with the store's kit in the tree and the store off | k1, st1 |
| Short prompts against reference N2 | CLEAN; median first divergence 208 tokens, median mean \|dlogprob\| 0.0090 | o2 (also w3, w5, o1) |
| Each chat alone against three together | equal, six of six | o2 |
| Same configuration, long prompts repeated | identical 3 of 3 inside one boot and 3 of 3 against another boot (two-slot build) | after o1 |

## By window

### w1: first three-stage boot (00:16Z to 01:01Z; first kit; arm X)

| What | Figure |
|---|---|
| Hand-off probes | eight pairs PASS |
| Arm X boot | tier digests and 5,651 hot as predicted (2,423 / 1,886 / 1,342 by stage) |
| Rule line 1 | FAIL: 15 of 15 answers differ from a two-stage reference boot of the same experts |
| Closeness, arm X against that reference | first divergence at token 1 on 12 of 12 prompts (a gap of 4.6 to 16.7 nat); token 0 right; three to five junk tokens, then the reference's answer for 100 tokens or more |
| KV pool | 1,389,057 tokens at three stages at 0.25 GiB |

### w2: diagnostic window (from 01:19:34Z; kit 2, the faulty cadence still in)

| Arm | Result |
|---|---|
| D0, two stages on the build's engine | CLEAN, 12 identical; alone-against-together check PASS |
| D2, Engram hand-over and middle commit only | JUNK, 12 of 12 |
| D1, mirror only | JUNK, 12 of 12 |
| X1, the build's layout, one slot | passed its boot checks; at 01:49:39Z the four cards of the first stage logged Xid 43 within 48 ms; the chain stopped by the crash rule |

PREDICTED at 01:34Z, before the three-stage results: D0 CLEAN, D1, D2 and X1 JUNK.

### w3: kit 3, the scheduling fix (about 02:1xZ to 03:01Z; then served to 04:27:12Z)

Arm X (three stages with the quantized expert mix of a two-stage reference boot): rule line 1 PASS against that
reference (closeness CLEAN, 0 of 12 early divergences; alone-against-together check PASS); highest
card sample 21,180 MiB. One-slot arm X1: CLEAN, no Xid.

Arm F2 (every routed expert NVFP4, two slots, 8,192-row chunks):

| What | Figure |
|---|---|
| Experts | 5,376 / 5,376 / 4,608 hot = 15,360; 0 cold on every rank |
| Free after load | 4.45 / 4.33 / 7.91 GiB by stage |
| One request alone | fresh 106.5 tok/s; repeated 170.8 |
| Three sent at once (two slots) | 108.5 tok/s per request fresh; 152.7 repeated |
| Long reads, 20k to 364k tokens | 2,265 to 3,327 tok/s |
| Highest card sample | 31,808 MiB on the middle stage (31,010 / 31,808 / 30,018): 190 MiB under the line |
| Closeness | CLEAN against N2 (0 of 12 early) |
| The owner's score line (rule line 3) | INCONCLUSIVE (see w5) |

The serving arm P3w3-S (from 03:01:22Z, two slots):

| What | Figure |
|---|---|
| Boot, driver-free / allocator reserved, by stage | 4,587 / 27,124; 3,773 / 27,918; 5,259 / 26,444 MiB; 799 / 819 / 807 MiB outside the allocator |
| Fullest card under an external client's replay of test prompts | 31,778 MiB at 03:04:55Z; 31,836 at 03:33:36Z (58 MiB in half an hour); level for 38 minutes; 31,988 at 04:11:51Z (10 MiB under the line) |
| The owner's sampler, 04:14:14Z to 04:27:00Z | middle stage low mark flat at 29,758 MiB, highs 31,768 to 31,814; first stage lows 28,406 to 28,914, high 30,778; last stage lows 28,016 to 28,786, high 30,016 to 30,018 |
| At the stop | 0 tracebacks, 0 out-of-memory lines |

The step to 31,988 MiB is unexplained. No stage showed a floor that keeps rising.

### s1: slot counts on the quantized expert mix (about 04:3xZ to 05:05:01Z; kit 4; arm X's experts)

Tool: `tools/slots_cells.py`. Two passes each; tok/s.

| Arm | Fresh text, sum | Repeated text, sum | Each request | Highest card sample |
|---|---|---|---|---|
| Three slots | 276.3 / 281.5 | 425.7 / 424.0 | not in the source | 21,182 MiB |
| Four slots | 281.7 / 281.9 | 425.8 / 427.6 | 60 to 76 fresh, 104 to 110 repeated; first token 0.4 to 1.0 s | 21,692 MiB |
| Six slots | the engine core died on the first requests (exit 30, no Xid): a label table sized for four slots | | | |

Four at once give the same total as three. Memory per added slot on the fullest card: about 510 MiB (ARITHMETIC,
21,182 to 21,692). PREDICTED for four slots: at most 21,612 MiB (431.7 MiB per slot): low by 80 MiB.

### s2: six and eight slots (started 05:35:09Z, stopped 05:48:11Z)

Stopped at the owner's word of 05:45:57Z. No six- or eight-slot verdict exists.

### w4, first run (05:54Z to 06:02Z; kit 6b)

Arm X did not boot: the first stage's four ranks refused at load ("V4.1 stage 0: the Engram of layer 14 was not
loaded"). The server's own line: "PP stage 0/3 loads 358 of 1008". Nothing was measured.

### w5: kit 6c (about 06:2xZ to 07:12:05Z)

Arm X: CLEAN, 0 of 12; median first divergence 115 tokens and median mean |dlogprob| 0.0063 (kit 3's own figures);
alone-against-together check PASS. One-slot arm: CLEAN. The first stage logs the moved Engram projection at 25.9
MiB as loaded (ESTIMATED before: 42). The middle stage holds no embedding.

Arm F2 (two slots, 8,192-row chunks):

| What | Figure |
|---|---|
| Experts | 15,360 hot, 0 cold |
| Load | 134 s |
| After warm-up, by stage | used 28,631 / 28,661 / 27,397 MiB; allocator reserved 27,792 / 27,806 / 26,556; outside the allocator 839 / 855 / 841 |
| Long reads, 20,331 to 364,374 tokens | 2,269 to 3,373 tok/s |
| Highest card sample | 31,336 MiB (30,478 / 31,336 / 30,046). PREDICTED about 31,560. Window w3: 31,808 |
| Rehearsal of two grown chats (17 requests, largest prompt 153,658 tokens) | middle stage high 31,334 MiB: 664 MiB under the line; first-token medians 2.93 s and 5.18 s |
| Arm F2e, cache release at 512 rows | same peak (31,334 against 31,334 MiB); larger turns' first token 10.9 s against 5.2 s |

Closeness on w5's receipts (identical on w3's): arm F2 against N2: median first divergence 208 tokens, median mean
|dlogprob| 0.0090. The two reference captures N2 and N2B: identical on
12 of 12. The owner's score line came out INCONCLUSIVE on both all-NVFP4 arms (index measure 12 for and 0 against;
mean measure 7 for, 0 against, 5 undecided of 12, with 8 needed). PLAN Amendment 10 replaced it.

### o1: first optimization window (07:55Z to 08:26:07Z; kit 6c)

| Arm | What | Figure |
|---|---|---|
| F2 | load; one read of 62,255 tokens; rehearsal | 139 s; 3,453 tok/s; middle stage high 31,334, low 29,440 MiB |
| F2 | closeness against N2 | CLEAN; 208 / 0.0090 |
| F3c | boot | 15,360 hot, 0 cold on the lowered reserve (3,686 MiB); load 135 s; after warm-up 28,535 / 28,667 / 27,253 MiB |
| F3c | short prompts | CLEAN; 208 / 0.0090 |
| F3c | long prompts against arm F2 (rule line c) | CLEAN, but median mean \|dlogprob\| 0.0333 against 0.0126 allowed: the arm ended there (exit 34) |
| F3c against F2, per prompt | mean \|dlogprob\| | 0.0403 at 5,694 tokens; 0.0333 at 11,151; 0.0245 at 17,935 |
| Serving boot P3o1-S (two slots) | rehearsal | highest card sample 31,422 MiB (30,474 / 31,422 / 30,038; 47 samples) |

The same two-slot configuration gave 31,334 MiB in w5's serve phase and in o1's arm F2: boot to boot the peak
moves by about 90 MiB.

Measured by the owner on the serving build after o1 (08:26Z to 08:33Z):

| What | Figure |
|---|---|
| Long prompts, same configuration | two captures of one boot identical 3 of 3; another boot identical 3 of 3: the noise floor is zero |
| Short prompts, arm F3c against arm F2 | identical 12 of 12 |
| Copy task RC, all-NVFP4 build | 1,500 tokens: 5,761 characters of reasoning and no answer, read-first and cold alike; 4,000 tokens: 534 of 534 words read-first, 536 of 534 cold (0.9981) |

One request each for the copy task: an observation. The cells' RC score of 0.0000 for the all-NVFP4 arm in w3 and
w5 compared "reasoning ended just inside 1,500 tokens" with "just outside". It is not a loss of the build.

### o2: second optimization window (09:09:05Z to 10:08:51Z; kit 7 with its switch at the default unless named)

The server was down from about 09:18:45Z to 10:13:07Z: 54 minutes (ESTIMATED before: 49).

| Arm | What | Figure |
|---|---|---|
| F2 (reference) | closeness; load; one read of 62,258 tokens; rehearsal high | CLEAN against N2, 208 / 0.0090; 135 s; 3,410 tok/s; 30,476 / 31,336 / 30,020 MiB |
| F2s (chunks 64 rows shorter) against F2 | long set | the 5,694-token prompt identical; the six prompts over 8,188 tokens differ: median first divergence 50, median mean \|dlogprob\| 0.0307 (per prompt 0.0298, 0.0316, 0.0280, 0.0367, 0.0364, 0.0003). PREDICTED 0.02 to 0.05 |
| F3c | boot | 15,360 hot, 0 cold |
| F3c | short prompts | CLEAN; 208 / 0.0090 |
| F3c | long prompts against F2 (rule line c2) | HOLDS: median first divergence 44 against 50; median mean \|dlogprob\| 0.0331 against 0.0307; no early wrong token; largest gap at a divergence 0.531 nat |
| F3c | alone against together, three chats | PASS (six chats, each equal alone and together) |
| F3c | three together, sum | 288.0 tok/s fresh, 444.6 repeated |
| F3c | one request alone | 129.5 tok/s fresh, 177.3 repeated |
| F3c | prompt read | 3,286 tok/s (96 % of arm F2's) |
| F3c | memory | no out-of-memory line; highest card sample 30,052 MiB (29,578 / 30,052 / 28,646); after warm-up 28,535 / 28,667 / 27,253; outside the allocator 867 / 881 / 865. PREDICTED about 30,600 (30,300 to 31,000) |
| F3o (three slots, 8,188-row chunks, hand-offs awaited) | all lines | every line passed; long prompts identical to arm F2 7 of 7; sum 263.6 / 427.0 tok/s; read 3,463 tok/s; highest 31,664 MiB. PREDICTED 640 MiB under arm F3r |
| F3r (three slots, 8,188-row chunks; memory only) | highest card sample | 31,628 MiB. PREDICTED 31,846 to 31,932 |
| Serving boot P3o2-S (10:04:41Z) | rehearsal high | 29,540 / 30,048 / 28,610 MiB |

Long chats, from the rehearsal's own lines: see the summary table. Cold copy set at 4,000 tokens, one request each:
the build RA 584 of 584 words, RB 467 of 467, RC 536 of 534 (0.9981).

Not measured in o2: three long chats at once (measured in k1); arm F3c's speed and memory lines on kit 6c itself
(they ran on kit 7 with the switch at its default; k1 showed kit 6c token for token equal on the long prompts).

### k1: the standing configuration under the card limit (10:53:38Z to 11:13:38Z; kit 6c)

| Boot | What | Figure |
|---|---|---|
| P3k1-S (limit 31,998 MiB) | load | 134 s |
| | experts | 15,360 hot, 0 cold |
| | card limit | on 12 of 12 workers; allocator limits 31,131 / 31,117 / 31,133 MiB; headroom 3,463 / 3,331 / 4,745 MiB (PREDICTED exactly) |
| | against the accepted capture | identical 7 of 7 (PREDICTED) |
| | rehearsal: 3 chats, 24 requests, largest prompt 153,663 tokens | exit 0; highest card sample 30,054 MiB (29,576 / 30,054 / 28,654). PREDICTED at most 30,350 |
| | three chats of 120,000 to 140,000 tokens read at once | first token after 105 to 121 s each |
| | turns | under 2,048 new tokens: median 4.73 s; larger turns 8.16 s |
| P3k1-SN (no limit) | load; experts; capture | 134 s; 0 cold; identical 7 of 7 |
| | rehearsal | exit 0, 0 out-of-memory lines, highest 30,052 MiB (29,576 / 30,052 / 28,632); turns 5.40 s and 6.67 s |

There was no out-of-memory event in k1. The chain's line "out-of-memory lines 12" came from a pattern that matched
the card limit's own announcement, once per worker. The second boot cost about 9 minutes more without the default.

### Step 0 probe of the four-card sum (11:19:21Z to 11:19:39Z; one board, no model)

Tool `tools/allreduce_probe.py`; receipt on the host, `logs/probe/allreduce-20261117T111921Z.out`.

| Case | Result |
|---|---|
| Paths | the group dispatched `['CUSTOM', 'PYNCCL']` as the serving boot does; custom path limit 8,388,608 B; [8188, 5120] and [8124, 5120] went through pynccl, [819, 5120] and [755, 5120] through the custom path |
| REPEAT (the same call twice) | equal |
| SIZE (the whole against its first 8,124 rows alone) | DIFFERENT: 7.15 % of the elements, 1,402 of 8,124 rows |
| SMALL | EQUAL |
| OFFSET (rows 64 to 8,187 alone against inside the whole) | DIFFERENT: 8.62 %, 1,684 rows |
| Against the four ranks added in FP32 and rounded once | the NCCL sum differs in 32.89 % of all elements, in every row; the custom-path sum in none |

Limits of the probe: the entry is proven, not that every hidden-size sum of a prompt chunk takes it; random
numbers, not hidden states.

### st1: store window 1 (12:42:04Z to about 12:59Z; the store's kit over kit 6c)

| Boot | What | Figure |
|---|---|---|
| KA (store's kit in the tree, store off, under the limit) | load; experts; limit; capture | 139 s; 15,360 hot, 0 cold; 12 of 12 workers; identical 7 of 7 |
| | rehearsal, three grown chats | exit 0, 0 out-of-memory lines, 30,052 MiB (29,574 / 30,052 / 28,718) |
| KD1 (store on, dump) | boot | refused by the store's fingerprint check at 12:54:50Z, exit 30 at 154 s. No record set was made; the store's tools never ran |

### s4: four slots (window from 13:32:14Z; the arm 13:33:59Z to 13:47:00Z; kit 6c alone, card limit 31,998 MiB)

Arm F4c: profile `run/serve-nvfp4-pp3-full-c4096-a4.yaml` (the three-slot profile with `max-num-seqs 4`; reserves
unchanged at 3,686 / 3,686 / 7,361 MiB). The store's kit was out of the trees. Receipts on the host:
`logs/window/s4/` (`chain.out`, `S4-RESULT.txt`), `run/state/P3s4-F4c.log`.

| Rule line | What | Figure |
|---|---|---|
| a | load; experts; card limit | 134 s, rc 0; 15,360 hot, 0 cold; the limit on 12 of 12 workers |
| a | after warm-up, by stage | 28,795 / 29,013 / 27,495 MiB (PREDICTED about 28,860 / 28,995 / 27,560); headroom under the allocator limits 3,203 / 2,985 / 4,503 MiB |
| b | against the accepted capture | identical 7 of 7 (PREDICTED: a slot changes no token) |
| d | each chat alone against four together | PASS |
| e | four requests together, twice | PASS, none starved. Sum 306.6 tok/s fresh (60.5 to 85.5 each; first token median 0.35 s) and 453.4 repeated (110.9 to 117.8 each) |
| e | one request alone | 128.1 tok/s fresh, 176.6 repeated |
| f | prompt read, one 62,263-token prompt | 3,284 tok/s (needed at least 2,793; three slots: 3,286) |
| g | rehearsal with four grown chats (4 chats, 30 requests, largest prompt 153,597 tokens) | exit 0, 0 out-of-memory lines, highest card sample 30,398 MiB (29,838 / 30,398 / 28,892): 1,600 MiB under the line, 1,300 under the ceiling of 31,698 |
| report | four documents of 110,000 to 140,000 tokens read at once | first token after 129 to 166 s each |
| report | turns with four grown chats open | under 2,048 new tokens: median 4.12 s; larger turns 9.33 s |

Against three together in o2 (288.0 and 444.6 tok/s), four together gave 6 % more on fresh text and 2 % more on
repeated text (ARITHMETIC). The chain ended in `final`. Four slots stand (PLAN Amendment 19).

### st3: the store window stopped at its first boot (15:47:43Z to 15:56Z; the store's kit `e6df5ed` over kit 6c)

Boot KA (store off, four slots, card limit): load 134 s, /health 200, 0 tracebacks, 12 of 12 limit lines, 0
out-of-memory lines, twelve sizing lines 5,376 / 5,376 / 4,608 hot and 0 cold, and the twelve two-tier tier-digest
lines of the store's kit with the predicted values (one value for stages 0 and 1, another for stage 2). The
driver's rank check then counted 24 lines where it wanted 12 and the stop rule restored the default. No line of
the build failed (PLAN Amendment 21).

### st3b: the store's first boots on this build (15:56:24Z to 16:20:43Z; default down 15:57:14Z to 16:20:29Z)

| Boot | What | Figure |
|---|---|---|
| KA (store off) | load; experts; limit; capture | 134 s; 0 cold; 12 of 12; identical 7 of 7 |
| | rehearsal, four grown chats | exit 0, 0 out-of-memory lines, 30,390 MiB (29,838 / 30,390 / 28,896); first token 3.92 s and 10.34 s |
| KD1 (store on, dump, empty root) | load; digests; store line | 139 s; the twelve digests as predicted; one new record set, every other count 0 |
| | S7 at 16, 64, 128 new tokens | PASS three times; the state check PASS on all twelve workers (twelve piece manifests per chunk); answers identical over 128 tokens |
| | S2 "before" at 130,000 tokens | as asked |
| | highest card sample of the boot, store and dump on | 30,398 MiB |
| | M12 | INCONCLUSIVE: the chats' restored state equal to the saved state in every piece on twelve workers, but the test's second reference differs from its first; the window stopped by its rule |
| | the dump | layer 0's window pages of positions 25,616 to 25,759 on stage 0, all four ranks: a read with a tail chunk of 2,292 rows and one of 2,308 rows give nine pages of which six differ; on tails of 6,376 and 6,392 rows (one split count) identical |

Not run: M1, M3, S6, boots KD2, KN and KN2 (PLAN Amendment 22).

### The split-count probe (16:49:38Z to 16:50:03Z; one card of the spare board, no model, no server)

Tool `tools/gemm_split_probe.py` with `tools/run_gemm_split_probe.sh`; receipt on the host,
`logs/probe/gemm-split-20261117T164938Z.out`. Layer 0's `wkv` projection (Q8_0, 5,120 to 512) through the engine's
own preparation on real embedding rows (PLAN Amendment 23).

| What | Figure |
|---|---|
| Split counts traced, by row count | 10 at 2,244, 2,260, 2,276, 2,292; 1 at 2,308; 4 at 2,196, 2,212, 2,228 and at 3,148, 3,164; 2 at 3,696, 3,712, 6,376, 6,392; 5 at 4,092 and 8,188 |
| Against the CPU model written before the run | equal at all sixteen |
| 2,292 against 2,308 rows | 6,395 of 1,173,504 projection values differ, in 2,126 of 2,292 rows; after the norm, the rotation and the 8-bit rounding 212 window values in 201 rows |
| Pairs on one split count | bit-identical whatever the row count or tile |
| With `TM_GEMM_TUNE=max_splits=1` | all 120 pairs bit-identical |
| Time of the `wkv` call, default against one split | 0.57 against 0.40 ms at 2,292 rows; 1.87 against 1.51 ms at 8,188; 0.62 against 0.78 ms at 3,696 |

### st4: the store passes its whole test list (17:15:01Z to 18:05:00Z; four slots, card limit, lengths of Amendment 23)

| Boot | What | Figure |
|---|---|---|
| KA (store off) | load; capture; four grown chats | 135 s; identical 7 of 7; 29,836 / 30,398 / 28,896 MiB; first token 2.18 s and 9.34 s |
| KD1 (store on, dump, empty root) | S7 at 16, 64, 128 | PASS, the state check on twelve workers |
| | S2 "before" | 130,000 tokens, as asked |
| | M12 at 22,736 and M1 at 22,704 tokens | PASS and PASS |
| | M3 warm | 438,350,272 B of pinned RAM for its packs |
| | S6 | 0 waits on twelve workers over 1,704 saves |
| KD2 (restart, same root) | the record set | the same set; 5 pinned and 102 conversation entries loaded; 0 damaged |
| | S2 "after" at 130,000 tokens | PASS: the chat back from disk, its state equal to the saved state, its turn's answer identical to the references |
| | M3 check; S6 | PASS; 0 waits over 624 saves |
| KN (store on, no dump, the same set) | the accepted capture read cold past the store | identical 7 of 7 |
| | four grown chats | exit 0, 0 out-of-memory lines, 29,802 / 30,396 / 28,894 MiB against KA's 29,836 / 30,398 / 28,896: no card memory |
| | S7 answers; S6 | identical; 0 waits over 648 saves |
| | first token with four grown chats (report only, one run) | 4.78 s and 12.24 s against KA's 2.18 s and 9.34 s: the larger turns 2 to 3 s above every store-off boot of the day; NOT established as a cost of the store |
| KN / KN2 | the restore pair: a chat of 129,666 tokens read in KN (first token 41.52 s), one turn of 131,466 tokens answered and saved; after the restart (the same set, 5 + 194 entries loaded) the same turn restored from the store (hit, reason store, at 131,456 of a stored 131,466) | first token after **1.23 s**; the first boot's turn had been read cold in 43.84 s. The answer is equal to the first boot's in 26 of 137 positions: the pair FAILS its line as written |

Why the pair's line failed (PLAN Amendment 25): the first boot's turn left the chat's prompt at its last token and
was read cold from 0, so its answer is a cold read's; the restored turn read its last ten rows in a ten-row chunk.
A cached turn and a cold read of one text differ in rounding on this engine. The store's own S2, which compares a
restored turn with turns started at the same position, is identical in state and answer.

### sv1: the build serves as the default (18:05:01Z to 18:16:40Z; on the test port; 150 W)

Arm P3sv1-S (`chain_serve.sh sv1`): 0 cold, the limit on 12 of 12, the accepted capture identical 7 of 7, four
grown chats exit 0 with 30,364 MiB, first token 3.99 s and 10.67 s (PLAN Amendment 27).

### mv1: the move to the serving port (19:17:01Z to 19:20:34Z)

`chain_move.sh mv1 sv1`: `stoparm S`, check, probes, `servestate` (the record matched), `servedefault` exit 0; no
request was sent before the build was declared serving. 3 minutes without a server (ESTIMATED before: about 4).
The first run on the cards of the two serving phases, the port copy and `stoparm S` on the serving port. One gap:
`chain_split.sh` refused the window `chain_move.sh` made, which had no record of the fallback's digests; copied by
hand; the chains now carry it forward (PLAN Amendment 29).

### g1: arm G's first run (19:20:51Z to 19:27:13Z; 200 W)

Arm G booted with `TM_GEMM_TUNE=max_splits=1`: 0 cold, 15,360 hot, the limit on 12 of 12. Line a's witness: every
field in the log `max_splits=1` (API server, engine core, the four last-stage workers), no parse-failure line,
twelve titled workers, and the variable unset in `/proc/<pid>/environ` of all twelve: FAIL by the rule as written,
exit 33; lines b to g did not run. The library's own line "max_splits 1" stood 16 times in the arm's log and 0 times
in the logs of s4, sv1 and mv1, so the switch was delivered; a titled worker's environment is blank by
construction. NO VERDICT on the switch (PLAN Amendment 29). The build served on the serving port again from 19:27:13Z.

### t1: the time budget and arm G's rerun (20:24:04Z to 20:43:34Z; 200 W)

`chain_budget.sh t1 g1 G`: the build stopped (cards 0 MiB 20:24:23Z), check, probes, arm T 20:25:06Z to 20:36:31Z,
arm G to 20:40:45Z, the build on the serving port again at 20:43:34Z by `servedefault` on the matching record (PLAN
Amendment 31).

Arm T, CAPTURED (T1: 120 trace files, 10 of 10 cycles, 0 profiler-failure lines, attention-timer lines from all
three stages 282 / 282 / 22, 0 tracebacks, 0 out-of-memory lines; highest card sample 30,394 MiB). From the log
(the trace analysis is to be `research/TWELVE-CARD-TIME-BUDGET.md`):

| One 4,092-row prompt chunk, median forward span | Stage 0 | Stage 1 | Stage 2 |
|---|---|---|---|
| Whole chunk | 793 ms (p90 882) | 891 ms (p90 1,310) | 844 ms |
| Experts, shared layer, router and their sums | 318 ms | 334 ms | 271 ms |
| Attention | 119 ms | 130 ms | 112 ms |
| Output projection with its sum | 104 ms | 113 ms | 90 ms |
| Query and key projections | 65 ms | 65 ms | 64 ms |
| Index (p90; it grows with context) | 53 ms (114) | 142 ms (362) | 222 ms (437) |
| hc norms | about 92 ms | about 89 ms | about 80 ms |
| Engram | 13 ms | | |

| What | Figure |
|---|---|
| Wall time per chunk of the untimed read (62,260 tokens at 3,712 tok/s) | 1,102 ms; the slowest stage's span is 0.81 of it: about 200 ms per chunk outside any stage's forward |
| The expert path's share of a chunk | 40 % on stage 0 (318 of 793 ms) and 37 % on stage 1 (334 of 891); 32 % on stage 2 (271 of 844) (ARITHMETIC from the rows above) |
| One-request decode | 25.1 ms per round without the profiler, 26.5 ms with it (+5.6 %); the same 300 tokens with and without |
| First read of 93,028 tokens | first token after 25.1 s (3,710 tok/s) |
| Turn of 880 new tokens on 128,617 tokens (a hit) | first token 1.24 s |
| Turn of 880 new tokens on 249,276 tokens (a hit) | first token 2.22 s |
| Four other turns (both 4,200-token turns among them) | read COLD: 36.0, 36.0, 37.9 s, and 88.4 s beside a second chat; by rule T2 those cells do not count. The probe replays answers without reasoning, so its turns leave the stored prompt at the last token |
| The arm's prefix log | 8 hits; cold: 7 "snapshot_pool_short", 5 "past_processed", 3 "partial_prompt" |

Arm G (line a as corrected in Amendment 29): line a PASS (fields `max_splits=1` in every logging process, 16
library lines, no other value, twelve workers). Line b: against capture P CLEAN; against the all-native N2 CLEAN (0
early wrong tokens of 12) and NEARER than capture P (first divergence 96; mean |dlogprob| 0.0079),
but KEEPS: WORSE, the median first divergence 96 tokens where at least 104 are asked (half the standing build's
208), with the mean |dlogprob| 0.0079 inside its bound of 0.018 (the standing build: 0.0090). WORSE without JUNK is
INCONCLUSIVE as coded: exit 34; lines c to g did not run. The switch stays out (Amendment 31).

### p1: kit 8 meets the cards (2026-11-17 23:30:01Z to 2026-11-18 00:06:33Z; 200 W; one boot per variant, every figure one run)

Kit 8 (twelve files over the eight-card base; three switches, all off by default) was put in both trees for the
window and kit 6c with the store's kit put back for serving; 36.5 minutes without a server. Each variant is judged
against this window's own `off` boot, with the strict equality judge (Amendments 37 to 39).

| Variant | What | Figure |
|---|---|---|
| `off` (kit 8, every switch off) | load; experts; limit | 144 s; 0 cold; 12 of 12 |
| | the accepted long-prompt capture | identical 7 of 7 (PREDICTED identical) |
| | reads, tokens: tok/s (first token) | 60,706: 3,850 (15.77 s); 93,025: 3,833 (24.27 s); 247,953: 3,316 (74.79 s); 425,777: 2,454 (173.49 s). PREDICTED within 2 % of 3,733 at 62,000: +3.1 % |
| | turns on a chat of about 128,600 tokens, all prefix hits, median of two (server prefill / client first token) | +880 new tokens 0.746 / 1.243 s; +2,048: 1.744 / 2.230 s; +4,200: 3.307 / 3.752 s; +10,000: 4.724 / 5.131 s |
| | a turn of 205 tokens beside another chat's 250,000-token read | first token 4.29 s, whole turn 14.15 s (alone 1.18 s and 1.72 s); the read 98.5 s (alone 74.8 s) |
| | the same cold read five times at 4,200 and at 10,000 new rows | identical token ids, 5 of 5 |
| | highest card sample | 30,446 MiB (29,838 / 30,446 / 28,894) |
| `rel` (`VLLM_V41_EMPTY_CACHE_TOKENS=8192`) | the accepted capture | identical 7 of 7 |
| | reads against `off` | 60,706 tokens 4,334 tok/s (+12.6 %); 93,027 tokens 4,212 (+9.9 %; the line asked +8 %); 247,953 tokens 3,409 (+2.8 %) |
| | turns (report only) | +2,048 prefill 1.589 s against 1.744 (-8.9 %); +4,200 3.191 against 3.307 (-3.5 %); +880 unchanged |
| | four-chat rehearsal (38 requests, largest prompt 160,288 tokens) | highest card sample 30,394 MiB (29,788 / 30,394 / 28,932), 0 out-of-memory lines. PREDICTED a higher mark than `off`: WRONG |
| | verdict | PASS on every line; adoptable by the next window's adopt boot |
| `mid` (`VLLM_V41_PP_MID_PREFILL_SKIP=1`) | line a | FAIL, nothing measured: the switch was active on all four middle-stage workers, but the arm's witness failed the boot on vLLM's warning "Unknown vLLM environment variable detected" for the unregistered switch |

The `rel` gain fades at 250,000 tokens, where the index dominates. Nothing was adopted in p1: the build served again
in its standing configuration (kit 6c with the store's kit) by the serving boot on the matching record.

### p2s: the release setting serves (04:17:00Z to 04:29:52Z; 200 W; kit 8 + `rel`)

The first `servecheck` on a card: the API server's environment holds 8192, the release line from 0 of 12 workers
(12 of 12 in p1's boots without it), identical 7 of 7, the rehearsal high 30,394 MiB. 12 min 27 s without a server.
The owner's cold read on the serving port: 425,782 tokens, first token 128.04 s, 3,325 tok/s, high 30,472 MiB (PLAN
Amendment 44).

### p2: the middle-stage skip (04:56:03Z to 05:57:17Z; 200 W; kit 8)

| What | `off` | `mid` | `adopt` (`rel` + `mid`) |
|---|---|---|---|
| Accepted capture; the same read five times at 4,200 and 10,000 rows | 7 of 7; 5 of 5 identical ids | 7 of 7; 5 of 5 | 7 of 7; 5 of 5 |
| Reads of 60,704 / 93,027 / 247,952 / 425,776 tokens | 3,810 / 3,696 / 3,284 / 2,757 tok/s | 4,206 / 4,322 / 4,301 / 4,125 (+10.4 / +16.9 / +31.0 / +49.6 %) | 4,562 / 4,515 / 4,405 / 4,223 (+19.7 / +22.1 / +34.1 / +53.2 %); first token 13.31 / 20.61 / 56.29 / 100.81 s (off 15.93 / 25.17 / 75.51 / 154.41) |
| The turn beside a 250,000-token read | first token 3.61 s, whole turn 12.75 s; the read's first token 86.96 s | 3.01 s, 10.43 s; 61.86 s | |
| Turns alone (+880 / +2,048 / +4,200 / +10,000) | reference | within 1 % but +10,000 -5.2 % | 0.750 s (0.749); -7.4 %; -3.0 %; -7.1 % |
| Published figures, one request alone, about 128,000 tokens of public text (report lines) | prefill 3,893 tok/s (127,679 tokens in 32.80 s); decode after the cold read 128.4 tok/s (7.79 ms a token, 529 out); after a one-step turn 103.7 tok/s (9.64 ms, 158 out) | | prefill 4,426 tok/s (28.85 s); 124.3 tok/s (8.04 ms, 398 out); 112.0 tok/s (8.93 ms, 209 out). Different lengths and text: not a pair |
| Highest card sample | 30,446 MiB | 30,410 | 30,394 (rehearsal) |

PREDICTED for `mid` +3 to +8 % at 94,000 tokens: wrong, low by half and more. Adopted; 61 minutes without a server
(Amendment 47).

### p3: decode first, stopped at `base` (06:30:04Z to 06:46:52Z; 200 W; kit 9L; the loop statistics at 64)

`base` (the receive-buffer wait 0) passed every measuring line and was failed by the witness (Amendment 50).

| What | Figure |
|---|---|
| Accepted capture | identical 7 of 7 (kit 9L's bytes with every switch off) |
| The loop's own counts after a ONE-step read (S1: 24 steady lines; L1: 7) | no empty step in 0 of 64 answers, 128 sleeps, 64 blocking draft takes, 2.83 ms from the answer to the next step's submission |
| After a MANY-step read (SM: 21 lines; LM: 11) | 64 of 64, 64 sleeps, 0 takes, 1.41 ms |
| Ms a token, forced 1,000-token sequences (medians; repeats differ with the text by up to 26 %) | S1 9.53, SM 8.20, L1 6.35, LM 7.47 |
| Prefill, median of 4 cold reads of about 127,663 tokens | 4,570 tok/s (27.9 s each) |
| Four together, two passes | 316.6 and 323.6 tok/s in sum on fresh text (247.1 / 253.0 by wall clock), one alone 130.5 / 131.0; repeated text 476.5 / 480.7, alone 179.9 / 179.1 |

### p3l: line discovery and the index capture (07:54:01Z to 08:04:43Z; 200 W; kit 9M; nothing judged)

One boot with `ahead`, `nosleep`, `gemv2` and the statistics at 64, four short requests: every line in a known form;
the wait's line (middle stage) and the drafter kernel's line (last stage) print at boot. The round's first printing
by a server, short prompts: medians 23.976, 24.183, 24.160, 24.237 ms; answer to submission 0.552 ms. The capture
boot: one cold read of 127,677 tokens, index operands written on 12 of 12 workers, 20 files, 133 MiB; high 30,476
MiB. 10 min 22 s without a server (Amendment 52).

### p3b: the decode window, stopped at `off` by a two-sided no-harm line (08:58:04Z to 09:25:29Z; 200 W; kit 9M)

| Cell | `base` round (ms) | `off` round | `off` against `base` (round / paired ms a token) |
|---|---|---|---|
| S1 (one-step read, short chat) | 25.457 | 25.335 | -0.48 % / -0.69 % |
| SM (many-step read, short) | 24.307 | 24.264 | -0.18 % / -0.14 % |
| L1 (one-step turn of 880 on 128,585 tokens) | 28.231 | 28.066 | -0.58 % / -2.02 % |
| LM (cold read of 127,663 tokens) | 26.309 | 26.302 | -0.03 % / -1.25 % |

The judge's "within 2 %" failed L1 for being faster by 2.02 %: INCONCLUSIVE, the list stopped (Amendment 54). What
it measured: two boots of this engine repeat the decode round within 0.6 % and the paired ms a token within about
2 %. One-step against many-step: 1.15 ms a round on the short chat, 1.92 ms at 128K. Prefill 4,580 tok/s at
127,663 tokens. 27 minutes without a server.

### p3c and p3r: five decode boots PASS; the power trip; the adopted set serves at 150 W (10:00:04Z to 11:58:19Z; kit 9M)

All five boots at 200 W (Amendment 58):

| Cell | `off` round (ms) | `ahead` | `ahead` against `off` | `ang` against `off` |
|---|---|---|---|---|
| S1 | 25.392 | 24.235 | -1.157 ms (-4.55 %); paired ms a token -4.25 % | -1.279 ms (-5.04 %) |
| L1 | 28.164 | 26.985 | -1.179 ms (-4.19 %); -4.00 % | -1.338 ms (-4.75 %) |
| SM | 24.311 | | +0.33 % (round), +0.20 % (ms a token) | |
| LM | 26.180 | | +0.61 %, -0.30 % | |

`base` against `off` (the wait): S1 25.239 / 25.392, SM 24.284 / 24.311, L1 28.144 / 28.164, LM 26.240 / 26.180 ms
(+0.32, -0.43, +0.04, +0.05 % paired): the wait costs nothing measurable. `ahead`'s counts on the one-step cells 64
of 64, 64 sleeps, 0 takes, answer to submission 1.40 ms (was 2.84). `an` and `ang`: no loss against the boot
before. `ang` at 128K on forced 1,000-token sequences: 5.88 ms a token after a one-step turn (170.0 tok/s), 137.9
tok/s after a cold read; prefill 4,564 tok/s at 127,663 tokens (median of 4). Highs 30,394 and 30,714 MiB.
PREDICTED for `ahead` 1.14 to 1.27 ms: met.

THE POWER TRIP: the host ended at 11:15:48Z during the adopted set's servecheck rehearsal (four long chats) with no
shutdown sequence and no kernel line; back at 11:18:37Z with twelve cards, 0 Xid. Caps to 150 W at 11:45Z (the owner's rule of
2026-11-17). p3r: servecheck PASS 11:55:49Z (7 of 7; 30 requests, largest prompt 153,624 tokens, high
30,714 MiB), servedefault 11:58:19Z: P3p3r-S with rel, mid, ahead, nosleep, gemv2 at 150 W. The owner's read:
425,781 tokens, first token 152.03 s (2,801 tok/s), high 30,356 MiB (at 200 W in p2: 100.8 s, 4,223 tok/s).

### p4: stopped at its first boot by the probe's option parser (13:30:03Z to 13:37:06Z; 175 W, 1,452 MHz ceiling; kit 9M)

`verify` passed its witness and 7 of 7; `tok_probe.py run ... --warm 0` exited 9 on its own usage line; nothing
about the cache measured; 7 minutes without a server (Amendment 60). The first window at the caps the owner set at
12:05Z (175 W; clock ceiling 1,452 MHz, cards at 1,447), in force since 12:09:23Z.

### p5: the prompt-end hold and the turn chunks (13:37:27Z to 14:32:24Z; 175 W; kit 9M, five adopted lines)

| What | `off` | `hold` (`VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD=128`) | `turn` (`VLLM_V41_PP_TURN_CHUNKS=3`) |
|---|---|---|---|
| Accepted capture | 7 of 7 | 7 of 7 | its own capture, kept as the TC3 candidate (another rounding) |
| Turns alone on a 128,000-token chat, first token (medians of 2): +880 / +2,048 / +4,200 / +10,000 | 1.184 / 2.061 / 3.687 / 4.963 s | | 1.131 / 1.471 / 2.319 / 3.623 s (the line: +4,200 at most 75 %: 63 %; +880 at most 103 %) |
| A turn of 880 (205 out) beside a 248,000-token read (medians of 3) | first token 3.33 s, whole turn 11.99 s; the read's first token 62.66 s | 3.26 s, 5.12 s (-57 %; the line asked -50 %); 60.99 s | |
| The same cold read five times | 5 of 5 at 4,200 and 10,000 | | 5 of 5 |
| Highest card sample | 30,386 MiB | 30,714 | 30,874 |

The gate boot (both switches on) met its readers first: each lone request's turn line is the rule's cut (head, then
1,024 + 1,024 + 1,024); the turn beside the read holds one read, 128 of 128 opening steps left; high 30,360 MiB.
Adopted: the hold; servecheck high 30,714 MiB; 54 min 37 s without a server (Amendment 62).

### p4b: the token cache (14:32:55Z to 15:06:52Z; 175 W; kit 9M, six adopted lines)

| What | Figure |
|---|---|
| `verify`: verify lines / answered chat-route requests | 32 / 32, every one "identical"; 20 of 20 turns hits reusing 127,659 to 135,958 ids of about 137,000 compared; the three edits (E1 inside an earlier message, E2 and E3 at a delimiter) hits with 135,127 reused, identical; exit summary hits 23, misses 9, ids reused 3,041,989, tokenized 373,353, evictions 4, different 0 |
| `one` against `off`, 27 cells of 27 | prompt tokens equal 27 of 27; the reply's text digest and tokens out equal 27 of 27 (the pieces' digest equal 27 of 27 too) |
| The turns' front (first token less queue less prefill), cell for cell over 20 turns | -256.8 ms at the median (-375.1 to -204.2); `off` 308.0 ms, `one` 47.2 ms; 0 of 20 not faster (the line: 150 ms, at most two slower). PREDICTED about 250 ms: met |
| The misses' front, 4 cold reads | -49.8 ms at the median (-90.9 to +18.5); the line allowed +30 |
| Line S at the 100th call | hits 23, misses 77, ids reused 3,041,989, different 0: exact |
| Highest card sample | 30,680 MiB (all three boots); servecheck 30,714 |

Adopted; 33 min 37 s without a server (Amendment 62). The owner's read on the set: 425,780 tokens, first token
104.46 s, 4,076 tok/s, high 30,392 MiB (the p3r set at the same caps: 103.24 s, 4,124).

### p5t and p5u: the turn chunks serve (15:46:51Z to 15:58:27Z; 16:40:00Z to 16:51:49Z; 175 W; kit 9M)

p5t: servecheck P3p5t-C passed 0 cold, the card limit, the turn chunks at boot, identical 7 of 7 against
`accepted/quality-TC3.json` (the hold and the token cache on), the rehearsal high 30,814 MiB; REFUSED by the
after-requests witness of `VLLM_V41_PP_EMPTY_AHEAD` (its first-use line needs a lone one-step request, which the
four-chat rehearsal reaches by chance only); went back to the seven lines (Amendment 64). p5u, with the lone
requests after the rehearsal (Amendment 64's rule): identical 7 of 7 against TC3 (median first divergence 200, mean
|dlogprob| 0.0000), rehearsal high 30,814 MiB (4 chats, 30 requests, largest prompt 153,627 tokens), the lone
334-token request printed the line in the same second; servedefault 16:51:49Z with EIGHT lines; 11 min 29 s without
the serving port. Reported from the lone requests (one repeat each): 8.157 ms a token after a one-step read (S1) against
9.505 after a many-step read (SM); L1 8.386 against LM 9.342. The owner's read: 425,784 tokens, first token
102.73 s, 4,145 tok/s (Amendment 65).

### p6: the tensor-core index scorer, went back on a wrong line (18:25:00Z to 18:57:54Z; 175 W; kit 9N)

| What | `ref` (the serving scorer) | `tc` (`VLLM_V41_INDEX_TC=1`) |
|---|---|---|
| Accepted capture | 7 of 7 | 7 of 7 |
| Prefill, about 127,663 tokens | 4,593 tok/s | 4,575 |
| Decode, forced 1,000-token sequences at 128K: after a cold read / after a one-step turn | 137.6 tok/s (7.27 ms a token) / 125.2 (7.99) | 145.4 (6.88) / 131.1 (7.63) |
| Round (medians of steady lines): S1 / SM / L1 / LM | 24.638 / 24.741 / 27.033 / 26.823 ms | 24.359 / 24.344 / 25.784 / 25.597 (read from the log by an independent reviewer; L1 -1.250 ms, LM -1.226) |
| Highest card sample | 30,814 MiB | 30,780 |

The lines boot's gate: 6 decode steps between two readings on every worker, 2 calls a step on stage 0 and 3 on
stages 1 and 2; high 29,204 MiB. Line C as written failed the TC boot (6,637 scorer steps against a floor of 6,848
from 108 statistics lines); the rule was wrong (Amendment 69): the counters count full-width verification steps of
the drafter, the statistics count every token-bearing batch, prompt chunks included. The corrected line: 6,637 =
6,637 on twelve workers (29 SpecDecoding reports, 33,185 drafted tokens at five a verification). 32 min 34 s
without the serving port (Amendment 67).

### p6b: the tensor-core index scorer adopted (19:35:00Z to 20:19:16Z; 175 W; kit 9N, nine lines after)

| What | `ref` | `tc` | Line |
|---|---|---|---|
| Accepted capture; the same cold read repeated | 7 of 7; same ids | 7 of 7; same ids | PASS |
| Line C (corrected) | gate 3 = 3 | 6,641 scorer steps on every worker = 6,641 verification steps | PASS |
| Round S1 / SM / L1 / LM (ms) | 24.445 / 24.437 / 26.633 / 26.677 | 24.375 / 24.381 / 25.829 / 25.719 (-0.070 / -0.056 / -0.804 / -0.958 ms; L1 -3.02 %, LM -3.59 %) | PASS (L1, LM at least 0.3 ms shorter, apart) |
| Per token (reported), L1 / LM / S1 / SM | 7.882 / 7.090 / 7.904 / 8.005 ms | 7.528 / 6.862 / 7.881 / 7.864 (-4.50 % / -3.17 % repeat for repeat) | reported |
| Streamed text equal to the reference's | | 22 of 22 repeats | reported |
| Prefill, the 128K cold read | 4,582 tok/s | 4,587 (+0.1 %) | PASS (at least 97 %) |
| Four together, fresh / repeated, sum | 316.6 / 449.1 tok/s | 337.9 (+6.7 %) / 451.6; none starved | PASS |
| Cells over the first look's limit; heads recomputed | | 15,796 of 47,575,695,264 (1 in 3,011,882); 8,544 | reported |
| Highest card sample | | 30,814 MiB; lines boot 29,204 | PASS (ceiling 31,800) |

Adopted: servecheck P3p6b-C (identical 7 of 7, the rehearsal, the lone requests), record 20:16:46Z, servedefault
P3p6b-S 20:19:16Z. 44 min 16 s without the serving port. The reference boots of p6 and p6b differ by about 0.4 ms a round
(27.033 / 26.823 against 26.633 / 26.677) while the TC boots agree within 0.05 to 0.12 ms: the gain stands at
ESTIMATE 0.8 to 1.2 ms a round at about 128K, MEASURED twice (Amendment 72).

### q1: the quality window (20:31:01Z to 20:48:45Z; 175 W; kit 9N, the nine lines; two one-slot boots without the drafter)

| What | Qa | Qb |
|---|---|---|
| speculative_config=None; out-of-memory and traceback lines; cold experts; the card limit | yes; 0; 0 on 12 of 12; on 12 of 12 | the same |
| Memory samples; highest card | 51; 30,020 MiB | 48; 30,020 MiB |
| The eight forced cells {4200, 94000, 128000, 252000} x {read, turn}, 205 tokens each, temperature zero | complete | complete |

Judge: QUALITY PASS, 8 of 8 equal text digests between the two boots. The Engram precision item: INCONCLUSIVE by
construction (no extracted reference). Way back by servestate on the matching record, servedefault P3q1-S
20:48:45Z; 17 min 44 s without the serving port (Amendment 75).


### The closing bench (21:01:56Z to 22:36:32Z; P3q1-S; 175 W; kit 9N, the nine lines; Amendments 74 and 77)

108 groups, 216 measured requests, judge PASS completeness (token digest matched the same prompt alone 198 of 198);
94 min 36 s on the serving port with no other traffic (ESTIMATE was 47 to 111 minutes). The tables are in
[measurements-2026-11-18.md](measurements-2026-11-18.md).

### st5, st6 and st7: the conversation store's window on the serving set (2026-11-18 23:31Z to 2026-11-19 02:21Z; 175 W; kit 9N, the nine lines; the store's overlay on in KD1 and KD2, rolled back after)

| What | Store off (KA, KN) | Store on (KD1 with dump, KD2 restarted on the saved set) | Line |
|---|---|---|---|
| Accepted capture | KA 7 of 7 identical; KN 7 of 7 cold past the store (148 entries loaded) | | PASS |
| The store's state checks | | S7 follow at 16 / 64 / 128 answer tokens with the state check; S2 at a history of 130,000 tokens before and, exact, after the restart; M12, M1, M3 (warm); 5 pinned and 107 conversation entries loaded, 0 damaged | PASS, every line |
| Waits on the save path (S6) | | 0 waits on 12 workers over 1,704 save lines (KD1), 0 over 624 (KD2) | PASS |
| Memory by stage, four-chat rehearsal | 30,578 / 30,814 / 28,304 MiB (KA, KN alike) | equal to KA's | PASS (ceiling 31,800) |
| First token of the larger turns, medians, four boots OFF / ON / ON / OFF | 3.596 s; 3.590 s | 4.718 s; 3.722 s (limit 3.696) | FAIL (both ON boots over) |
| First token of the small turns (under 2,048 requested tokens) | 2.586 s; 2.302 s | 3.029 s; 0.999 s (limit 2.686) | INCONCLUSIVE (the ON boots straddle it) |

Every state line of the store passed and its timing line failed: THE STORE IS A FAILED OPTIMIZATION on this build as the
rule stands (Amendments 79 and 81). The overlay rolled back (26 files in each tree), servestate matched the record of
p6b, servedefault P3st7-S at 02:21:15Z (164 s to ready). st7 took 69 minutes without the serving port (ESTIMATE 92). Before
it, st5 stopped at its hand-off probe on a port race in a borrowed rendezvous probe (replaced by the build's own
`window/nccl_probe.py`, Amendment 80), and st6 served the unchanged configuration again at 01:11:43Z after 59 minutes
lost to a one-off runner that ended on a `diff | grep -c` under `pipefail`: nothing served from 00:00:01Z to 01:11:43Z.
Observed, no cause established: the two ON boots differ from each other more than the OFF boots do, and the first ON
boot after a cold set load (197 entries) is the slow one.

### c3a: the lean pipeline hand-off (C3) on kit 9P did not boot (02:32:07Z to 02:51:45Z; 175 W; kit 9N, the nine lines)

The reference boot (switch off) passed every line: identical 7 of 7, highest card sample 30,814 MiB, Xid 0. The
candidate's middle-stage workers refused in the hand-off's constructor: kit 9P assumed a hidden size of 4,096 where the
model's is 5,120 (the GGUF's own metadata, receipt `accepted/model-geometry.json`); its CPU test had taken
4,096 from inside the test, not from the model. C3 INCONCLUSIVE (Amendment 82); fixed as kit 9Q (geometry from the
model, each refused condition its own error, the test reading the receipt), run again as c3b. Way back: kit 9N, the
nine lines, servedefault P3c3a-S at 02:51:45Z (197 s to ready); 20 minutes without the serving port.

### c2a: the batched 4,092-row weight-only projection adopted (03:19:25Z to 03:55:51Z; 175 W; kit 9S, ten lines after)

| What | `Wr` (switch off) | `Wc` (`VLLM_V41_WO_A_BATCHED_4K=1`) | Line |
|---|---|---|---|
| Cold reads, tok/s, for 60,705 / 93,025 / 247,948 / 425,779 tokens delivered | 4,476 / 4,559 / 4,358 / 4,028 | 4,663 / 4,741 / 4,531 / 4,309 (+4.2 / +4.0 / +4.0 / +7.0 %) | PASS (each at least 97 %; the 93K cell at least 102 %: 104.0) |
| First token of those reads | 13.56 / 20.41 / 56.90 / 105.70 s | 13.02 / 19.62 / 54.72 / 98.81 s | reported |
| Closeness against the accepted capture (the change moves rounding) | | identical 1 of 7, none diverging at tokens 0 to 3; median first divergence 58 (line at least 25); median of the mean \|dlogprob\| 0.0354 nats (line at most 0.0614); largest gap at a divergence 0.469 nat | HOLDS |
| Highest card sample; by stage | 30,816 MiB; 30,578 / 30,816 / 28,304 | 30,832 MiB; 30,594 / 30,832 / 28,328; 0 out-of-memory lines | PASS (ceiling 31,800) |

Adopted: ten lines in `run/state/ADOPTED.env`; the reference capture from here is `accepted/quality-TC3-WOA4K.json`.
The SERVE-CHECKED record was stale (p6b), so servecheck ran on the test port: PASS at 03:53:11Z (highest card sample 30,830
MiB), record written for window c2a; servedefault P3c2a-S on the serving port at 03:55:51Z. 36 min 26 s without the serving port
(ESTIMATE 35 to 46). The prediction (about 2.7 % off the chunk, by the code read) was WRONG, low: the measure is 4.0 to
7.0 %, largest at the longest read (Amendment 83).

### c3b: the lean pipeline hand-off (C3) on kit 9T ran and failed its line (04:32:45Z to 05:02:08Z; 175 W; kit 9S, the ten lines; C3 closed)

| What | `Cr` (switch off; 0 active hand-off markers) | `C3` (switch on; 12 active and 12 first-forward markers) | Line |
|---|---|---|---|
| Accepted capture; the same cold read repeated | 7 of 7; 5 of 5 identical | 7 of 7 | PASS |
| Round S1 / SM / L1 / LM (ms; medians of 16 / 14 / 11 / 11 steady lines) | 30.810 / 27.555 / 26.253 / 25.954 (S1 and SM spread 24.7 to 44.1 and 24.4 to 38.3) | 24.320 / 24.285 / 25.947 / 25.822 | LM 0.133 ms shorter, under the 0.3 ms line: FAIL; L1 0.306 ms shorter with the two boots' lines overlapping: UNREADABLE; S1 and SM within "not more than 2 % longer" |
| Highest card sample | 30,830 MiB; 0 out-of-memory lines | 30,830 MiB | PASS |

The short cells' differences (S1 21.1 % and SM 11.9 % shorter) are the reference boot's wide spread, not a measured gain:
no line claims them. The prediction (0.15 to 0.45 ms shorter a round) was met at its low end and the line was not.
THE LEAN HAND-OFF IS A FAILED OPTIMIZATION on this build (Amendments 84 and 85). Way back by the chain: kit 9S in both
trees, the ten lines, SERVE-CHECKED MATCH (c2a), servedefault P3c3b-S at 05:02:08Z; 29 min 23 s without the serving port
(ESTIMATE 36 to 42). Observation, no rule event: the health watch on P3c2a-S exited at 04:30:50Z on one health probe
answering 000 (curl, 5 s); the gate read health 200 twenty seconds later, no Xid line, no reset; the arm served until
its planned stop at 04:32:52Z.

### c7a: the packed-576 experts (C7) on kit 9U ran and failed their line (05:06:41Z to 05:39:07Z; 175 W; kit 9S, the ten lines; C7 closed)

Qualification only (the window's rule): kit 9U = kit 9S plus the three packed-expert files; the re-cut packs (320 files, 61,440
experts, 288,777,830,400 bytes) and the sidecar library staged outside the serving trees.

| What | `Jr` (kit 9U, switch off) | `Jp` (`VLLM_V41_EXPERT_PACKED576=1`, twelve workers packed; ready in 184 s) | Line |
|---|---|---|---|
| Accepted capture | 7 of 7 identical | 7 of 7 identical | PASS |
| Round S1 / SM / L1 / LM (ms) | 24.388 / 24.284 / 25.895 / 25.588 | 24.453 / 24.503 / 25.990 / 25.743 (S1 +0.27 %, SM +0.90 %; L1 +0.095 ms, LM +0.155 ms) | L1 and LM asked at least 0.3 ms shorter: FAIL; S1 and SM within 2 %: PASS |
| Highest card sample; by stage | 30,832 MiB; 30,596 / 30,832 / 28,328 | 29,952 MiB; 27,572 / 29,952 / 29,952 | reported |
| Memory like for like over the boot, the seven answers and the decode cells, by stage | 28,328 / 30,830 / 30,830 MiB | 27,572 / 29,952 / 29,952 MiB (756 / 878 / 878 MiB less a card) | reported |

The packed boot answered the accepted prompts token for token and used less card memory, but its decode rounds were
longer, not shorter: the prediction (0.33 to 0.41 ms shorter) was WRONG in sign, the memory size was met and the stage
pattern was not. The later lines of the rule were not run (INCONCLUSIVE). THE PACKED EXPERTS ARE A FAILED OPTIMIZATION on
this build as the rule stands (Amendments 85 and 86). Way back by the chain: sidecars out, kit 9S, the ten lines,
SERVE-CHECKED MATCH, servedefault P3c7a-S at 05:39:07Z; 32 min 26 s without the serving port (ESTIMATE 43 to 48).


### The final bench and the prefix-sharing measure again (2026-11-19 05:53:12Z to 08:08:28Z; P3c7a-S, not restarted; 175 W; kit 9S, the ten lines; Amendment 87)

The same two measures as 2026-11-18 on the set every optimization left: the bench 05:53:12Z to 07:24:46Z (108 groups, 216 measured
requests, judge PASS completeness, token digest matched the same prompt alone 198 of 198; 91 min 34 s) and the prefix-sharing
measure 07:24:46Z to 08:08:28Z (108 groups, 216 requests, hits 150 of 216, judge PASS; 43 min 42 s); 2 h 15 min 16 s with the
serving port up and its four users holding their traffic; highest card sample 30,418 MiB (1,522 samples). The tables are the
section Closing bench, 2026-11-19, below; the predictions of Amendment 87 are checked in the section Predictions against results.

### b4a: the split count (B4) adopted (12:30:04Z to 13:13:21Z; 175 W; kit 9S, eleven lines after)

The window's rule and chain are not in this release; the serving port was down 43 minutes; Xid 0; peak card sample 30,832 MiB on both arms.

| What | `Wr` (kit 9S, the ten lines) | `Wc` (the ten lines + `TM_GEMM_TUNE=max_splits=1`) | Line |
|---|---|---|---|
| The split witness | no `TM_GEMM_TUNE` field, no parse line | every field `max_splits=1`; 16 library parse lines "max_splits 1", none other | PASS |
| Accepted capture (`quality-TC3-WOA4K.json`) | 7 of 7 identical | identical 1 of 7; CLEAN (early wrong-token divergences 0 of 7); median first divergence 45 / 48 (asked at least half of 50), median mean \|dlogprob\| 0.0366 / 0.0350 (asked at most twice 0.0307) | HOLDS |
| Cold reads of 60,706 / 93,028 / 247,953 / 425,783 tokens (tok/s; first token s) | 4,664 / 4,747 / 4,515 / 4,327; 13.02 / 19.60 / 54.92 / 98.41 | 4,659 / 4,756 / 4,552 / 4,314 (-0.1 / +0.2 / +0.8 / -0.3 %); 13.03 / 19.56 / 54.47 / 98.70 | within 3 % either way: PASS |
| Decode at four slots, no request starved | one fresh 135.6 tok/s | one fresh 137.3 (+1.3 %); one repeated and four together +4.7 and +4.3 % (report only) | PASS |
| Hits against the cold read (line e, report only) | 9 of 9 hits by timing; the tool's parser rejected 7 of 9 receipts | 9 of 9 hits; 9 of 9 rejected | a defect of the tool's float32 mass bound, fixed; re-run in b3a |

Adopted: the eleventh line written, `Wc`'s long capture bound as `accepted/quality-TC3-WOA4K-S1.json`, the full check on
the test port 13:02:43Z to 13:10:50Z, the serving boot 13:10:51Z; P3b4a-S serves the serving port since 13:13:21Z (Amendment 89).

### b3a: the exact four-card sum (B3) on kit 9V ran and failed its closeness line (13:45:04Z to 14:08:19Z; 175 W; kit 9S, the eleven lines; B3 closed)

The window's rule and chain are not in this release; the serving port was down 23 min 15 s; Xid 0; peak card sample 30,832 MiB on `Wr`.

| What | Step 0 probe (stage 0's four cards, idle, no model) | `Wr` (kit 9V, switch unset, the eleven lines) | `Wc` (`VLLM_V41_TP_EXACT_SUM=1`) | Line |
|---|---|---|---|---|
| The exact sum's cost on the board, BIG tile, median of 20 timed calls | off 1.180 ms; on 2.632 ms (2.2 times; the four ranks bit-identical) | | | reported |
| The witness | | first uses 0, the switch not set | 12 of 12 ranks on the exact path, tile 8,388,608 bytes | PASS / PASS |
| Accepted capture (`quality-TC3-WOA4K-S1.json`) | | 7 of 7 identical (kit 9V unset is today's arithmetic) | identical 0 of 7; CLEAN; median first divergence 13 (at least 25 needed), median mean \|dlogprob\| 0.0348 | FAIL |
| Cold reads of 60,704 / 93,025 / 247,953 / 425,780 tokens (tok/s; first token s) | | 4,675 / 4,768 / 4,532 / 4,331; 12.99 / 19.51 / 54.71 / 98.31 | not run | |
| Decode at four slots | | one fresh 137.6, one repeated 188.5, four together 328.5 tok/s summed | not run | |
| A hit against its own cold read (nine bench prompts, 32K to 128K; the fixed tool) | | identical 4 of 9; median first divergence 25, median mean \|dlogprob\| 0.0275; hits by timing 9 of 9 (0.25 to 0.45 s against 7.2 to 28.6 s cold); the server sends no cached-token count | | reported (D11) |

The rule's prediction named the speed line as the one that fails; the closeness line failed first, so the read cost in
percent was not measured: WRONG on the deciding line. Returned to kit 9S on a matching record; P3b3a-S serves the serving port
since 14:08:19Z (Amendment 90).

### The bench of Amendment 89 (14:09:20Z to 16:24:31Z; P3b3a-S; 175 W; kit 9S, the eleven lines; Amendment 92)

The closing bench 14:09:20Z to 15:40:33Z (91 min 13 s) and the prefix-sharing measure 15:40:33Z to 16:24:31Z (43 min
58 s) on the serving port without a restart, its users holding their traffic; both judges PASS; highest card sample 30,418 MiB
(1,513 samples). The tables are the section Closing bench, 2026-11-19, second, below.

## Closing bench, 2026-11-19, second (the eleven-line set; the serving set of window b4a, kit 9S, eleven lines)

MEASURED on 2026-11-19, 14:09:20Z to 15:40:33Z, on the serving set since window b4a (kit `PP3KIT9S-over-cut`, the eleven
adopted lines, four slots; P3b3a-S, the server not restarted for the bench), 175 W per card and the 1,452 MHz clock
ceiling: the same bench as the two before it, unchanged (code, legal and prose prompts of 1,024 / 8,192 / 32,768 / 65,536 /
131,072 / 262,144 tokens, read cold, one, two and three streams released together, two repeats each; 108 groups, 216
measured requests; greedy, 1,024 tokens out, thinking off); judge PASS completeness (token digest matched the same prompt
alone 198 of 198). "Prefill tok/s" is the prompt's tokens over the time to its first token. **Total decode tok/s** is all
streams' output tokens over the window from the earliest first token to the last request's end (the bench tool's
arithmetic; the tool is not in this release); **Sum of decode rates** adds each stream's own decode rate while all of them
stream. The users of the port held their traffic (the hold line of 14:09Z, the free line of 16:26Z). What this set changes
against the 05:53Z run is one line, the split count (window b4a), which moves rounding: every cell generated different text
from the 05:53Z run's (0 of 54 token digests shared), so the decode columns moved with the text, -10.1 to +9.0 % by cell and
-0.9 % on the mean of the 54 aggregate cells, the round itself unchanged (window b4a); the cold reads are within -0.7 to
+0.7 % of the 05:53Z run's second repeats at every length; and the slow first cold read of a legal or prose document seen
in the two runs before (1 to 11 s slower than its second) did not recur, the first and the second repeat agreeing within
0.15 s in every one-stream cell (HYPOTHESIS, from the one line that differs: the library's split search on a shape not yet
seen, which the fixed split count removes; not established by a boot built for it). The 05:53Z run's tables are kept
unchanged in [measurements-2026-11-19-morning.md](measurements-2026-11-19-morning.md). Amendment 92.

### Code

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

### Legal

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

### Prose

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

The prefix-sharing measure that followed, 15:40:33Z to 16:24:31Z (108 groups, 216 requests, 18 warm reads, hits 150 of
216, the server sending a cached-token count for 0; judge PASS; a hit is a first token under a quarter of the warm read's;
the 36 requests at 1K are not hits by that rule, and 30 requests re-read the document, one per new slot the first time a
second or a third stream shared it, never after): hits at 128K first token 0.43 / 0.44 / 0.44 s (code / legal / prose; 0.42
to 0.43 in the 05:53Z run), at 256K 0.63 to 0.73 s; the warm reads within -0.4 to +0.3 % of the 05:53Z run's at 32K and
longer; the all-hit repeat of three streams on one cached 128K document 432.7 / 232.0 / 222.5 tok/s summed (426.6 / 230.1 /
221.0 over the window; 405.5 / 244.2 / 223.4 summed in the 05:53Z run, the text having moved). 0 of 150 hits reproduced the
warm read's token digest and all 30 re-reads did.

### Code, with prefix sharing

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

### Legal, with prefix sharing

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

### Prose, with prefix sharing

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

## Predictions against results

| Prediction (where written) | Predicted | Measured | Window |
|---|---|---|---|
| Diagnostic arms with the faulty cadence (01:34Z) | D0 CLEAN; D1, D2, X1 JUNK | D0 CLEAN; D1, D2 JUNK; X1 faulted with Xid 43 | w2 |
| Highest card sample at four slots, quantized mix (Amendment 4) | at most 21,612 MiB | 21,692 MiB | s1 |
| Middle-stage highest sample, two slots, memory kit (Amendment 5) | about 31,560 MiB | 31,336 MiB | w5 |
| Moved Engram projection (Amendment 5) | about 42 MiB | 25.9 MiB as loaded | w5 |
| Arm F3c highest card sample (after w5) | about 30,600 MiB (30,300 to 31,000) | 30,052 MiB | o2 |
| A 64-row shift of the chunk cut, median mean \|dlogprob\| (Amendment 12) | 0.02 to 0.05 | 0.0307 | o2 |
| Arm F3r highest card sample (Amendment 11) | 31,846 MiB (31,846 to 31,932) | 31,628 MiB | o2 |
| Arm F3o against arm F3r (Amendment 11) | up to 640 MiB lower | 36 MiB higher | o2 |
| Allocator limits and headroom under the card limit (Amendment 14) | 31,131 / 31,117 / 31,133; 3,463 / 3,331 / 4,745 MiB | exactly those | k1 |
| Serving boot on kit 6c against o2's capture on kit 7 (Amendment 14) | identical 7 of 7 | identical 7 of 7 | k1 |
| Rehearsal with three grown chats (Amendment 13) | at most 30,350 MiB | 30,054 MiB | k1 |
| Boot KA against the accepted capture (Amendment 15) | identical | identical 7 of 7 | st1 |
| Store window 1: every line passes (Amendment 15) | pass | boot KD1 refused | st1 |
| Four slots, highest card sample (Amendment 18) | about 30,400 MiB, at most 30,800 | 30,398 MiB | s4 |
| Four slots, after warm-up by stage (Amendment 18) | about 28,860 / 28,995 / 27,560 MiB | 28,795 / 29,013 / 27,495 MiB | s4 |
| Four together against three together, sum on fresh text (Amendment 18) | the same sum as three (288.0 tok/s), about 70 tok/s each | 306.6 tok/s against 288.0: 6 % more; 60.5 to 85.5 each | s4 |
| Prompt read at four slots (Amendment 18) | within 3 % of 3,286 tok/s | 3,284 tok/s | s4 |
| First token with four grown chats open (Amendment 18) | about 6 s and 10 s | 4.12 s and 9.33 s | s4 |
| Four slots against the accepted capture (Amendment 18) | identical | identical 7 of 7 | s4 |
| Store window st3 / st3b: every line passes (Amendments 15, 16, 19, 20) | pass | st3 stopped at the driver's rank count; st3b stopped at M12, INCONCLUSIVE | st3, st3b |
| The twelve two-tier tier digests with the store's kit (Amendment 20, read in the kit) | two values, by stage | as predicted | st3, st3b |
| Split counts at sixteen row counts (CPU model `tools/tm_dispatch_emulation.py`, before the probe) | the model's count at each | all sixteen as modelled | probe 16:50Z |
| Store window st4 (Amendment 23) | every line passes; the restore pair's first token in seconds | every store line PASS; the pair restored in 1.23 s and FAILED its token-identity line | st4 |
| Store on, card memory against off (Amendment 16) | within 100 MiB | 2 to 34 MiB lower | st4 |
| The restored turn's first token (Amendment 16) | 2 to 10 s | 1.23 s | st4 |
| Arm G line a (Amendment 24) | holds | FAIL (the witness's environment clause); then PASS under the corrected line | g1, t1 |
| Arm G line b (Amendment 24) | CLEAN, KEEPS, NEARER, close to 208 tokens and 0.0090 | CLEAN and NEARER, 0.0079; first divergence 96 tokens where 104 are asked: INCONCLUSIVE | t1 |
| Arm T captures; no out-of-memory line under the limit (Amendment 30) | holds | holds | t1 |
| A 4,092-row chunk on stages 0 and 1 (Amendment 30) | about 0.97 s each, within 10 %, stage 1 the slower | 793 and 891 ms: stage 1 the slower by 12 %, both faster than predicted | t1 |
| A turn of about 880 new tokens (Amendment 30) | 0.7 to 0.9 s to the first token | 1.24 s (a hit) | t1 |
| A one-request decode round (Amendment 30) | about 26 ms | 25.1 ms | t1 |
| `rel` reads at 62,000 tokens within 2 % of 3,733 tok/s (Amendment 35) | within 2 % | +3.1 % at 60,706 | p1 |
| `rel` memory high-water mark higher than `off` (Amendment 35) | higher | 30,394 against 30,446 MiB: not higher | p1 |
| `rel`'s gain at 94,000 tokens up to about +9 % (Amendment 35, corrected) | up to +9 % | +9.9 % | p1 |
| The owner's 420,000-token read on the `rel` set (Amendment 41) | 150 to 175 s | 128.04 s | p2s |
| `mid` at 94,000 tokens (Amendment 40) | +3 to +8 % | +16.9 % (the gain grows with length: +49.6 % at 426K) | p2 |
| `adopt` repeats `rel`'s +9.9 % within 2 points at 93K, floor 108 % (Amendment 40) | about +10 % | +22.1 % with `mid` in the set | p2 |
| The one-step-turn decode line about 1.1 to 1.3 ms a token slower than the cold-read line (Amendment 43) | 4 % at 128K | 1.85 ms (`off`) and 0.89 (`adopt`), on different texts: direction only | p2 |
| `off` within 1 % of `base` on every decode cell; `ahead` 3.9 to 4.5 % fewer ms a token (Amendment 46) | as stated | p3 stopped at `base`'s witness; measured in p3c: `ahead` -4.25 % (S1), -4.00 % (L1) | p3, p3c |
| `ahead` 1.14 to 1.27 ms a round shorter on S1 and L1 (Amendment 53) | 1.14 to 1.27 ms | -1.157 and -1.179 ms | p3c |
| `an` about equal to `ahead`; `ang` unknown (Amendment 53) | 0 to 0.1 ms | no loss; `ang` against `off` -1.279 / -1.338 ms | p3c |
| The token cache: about 250 ms shorter at the front of a turn; a miss within 30 ms (Amendment 49) | 250 ms; +30 ms | -256.8 ms; the misses -49.8 ms | p4b |
| The hold passes; the turn chunks measured (Amendment 59) | pass | the whole turn beside a read -57 % (asked -50 %); +4,200 at 63 % (asked at most 75 %) | p5 |
| The turn chunks serve: every step passes; 7 of 7 against TC3 with the hold and the cache on (Amendment 63) | pass | 7 of 7 held; refused by the after-requests witness (a line that needs a lone request) | p5t |
| The lone requests print the line; the rehearsal high 30,600 to 31,000 MiB (Amendment 64) | as stated | printed in the same second; 30,814 MiB | p5u |
| The round at about 128K 0.85 to 0.95 ms shorter with the tensor-core scorer (Amendment 61) | 0.85 to 0.95 ms | p6 (read from the logs after the wrong line): -1.250 / -1.226 ms; p6b: -0.804 / -0.958 | p6, p6b |
| Line C: 2n / 3n calls between two readings and n > 64(m-1) (Amendment 61) | holds | 2n / 3n held; the floor mixed units and was wrong (Amendment 69) | p6 |
| L1 and LM about 1.2 ms shorter (Amendment 69) | 1.2 ms | 0.80 and 0.96 ms (the reference boot moved by about 0.4 ms) | p6b |
| The quality window: text equal 8 of 8; the Engram line INCONCLUSIVE; 15 to 23 min (Amendment 73) | as stated | 8 of 8; INCONCLUSIVE; 17 min 44 s | q1 |
| Minutes without a server: p2s 14; p2 60 to 74; p3b/p3c 79; p4 32 to 36; p5 67; p5u 13 to 17; p6 about 50; p6b about 50 | | 12.5; 61; (p3c crossed the power trip); 33.6; 54.6; 11.8; 32.6; 44.3 | |
| Code one stream at 128K about 133 tok/s after a cold read (Amendment 74) | about 133 tok/s | 161.1 tok/s (6.2 ms a token): wrong, low | closing bench |
| Prefill 4,400 to 4,600 tok/s at 61K to 128K (Amendment 74) | 4,400 to 4,600 | code 4,414 and 4,444: met; legal 3,746 / 3,794 and prose 3,670 / 4,033: not met (the band was drawn from code cells) | closing bench |
| Prefill about 4,100 tok/s at 256K (Amendment 74) | about 4,100 | 4,152 / 4,076 / 4,159 (code / legal / prose): met | closing bench |
| Three streams at 1K about 290 tok/s in sum (Amendment 74) | about 290 | code 431.8 (414.5 over the joint window): wrong, low; legal 247.5 and prose 213.3 (no kind was named) | closing bench |
| The closing bench's reserved interval (Amendment 74) | 47 to 111 minutes, reserve 130 | 94 min 36 s | closing bench |
| One-stream decode at every length within the A/A band of 2026-11-18 (Amendment 87) | within the band | cells moved -8.5 to +8.6 % because the generated text moved with the rounding change (ids differ in every cell of 8K and longer); the six cells with the same text within 1.1 to 3.4 %; mean over the moved cells +0.9 %: wrong as written, the round unchanged | final bench |
| Cold-read prefill +4 to 7 % at 32K and longer; code 128K 4,620 to 4,750 tok/s; 256K first token 59 to 61 s (Amendment 87) | as stated | code +4.8 / +5.0 / +4.8 / +5.0 %, 4,657 at 128K, 60.23 s at 256K: met; legal and prose met at the low end on their second repeats only (+3.5 to +4.3 %), their first cold read of a document being 1 to 11 s slower in both benches | final bench |
| Three streams: the summed decode within the band; the aggregate column +4 to 7 % (Amendment 87) | as stated | the sum moved with the text (-4.0 to +7.5 %); the aggregate +0.0 to +4.5 % and -1.7 at worst: wrong, the column starts at the earliest first token, not the first request's start (the 2026-11-18 description was wrong) | final bench |
| Prefix hits unchanged (128K 0.40 to 0.47 s; 390 / 254 / 228 tok/s on one cached 128K document); warm reads +4 to 7 % (Amendment 87) | as stated | 0.42 to 0.43 s; 399.8 / 242.2 / 221.5; warm reads 4.0 to 4.2 % faster: met | prefix measure |
| The bench 85 to 100 minutes, the measure 40 to 50, about 2 h 15 in all (Amendment 87) | as stated | 91 min 34 s; 43 min 42 s; 2 h 15 min 16 s: met | final bench |
| B4: not identical, CLEAN and HOLDS against the o2 yardstick; reads within 3 % either way; decode under 2 % either way (Amendment 88) | as stated | identical 1 of 7, CLEAN, HOLDS (45 to 48 against 50; 0.0350 to 0.0366 against 0.0307): met; reads -0.3 to +0.8 %: met; decode one fresh +1.3 % met, the repeated and four-together cells +4.7 and +4.3 % (the text moved): exceeded upward | b4a |
| B3: line c (speed) is the line that fails, by a cost estimate of 6 to 12 % of a 128K read (the window's rule, Amendment 88) | fails on c | line b (closeness) failed first, median first divergence 13 against 50; c never ran: wrong on the deciding line; the probe's board cost 2.2 times a large call stands | b3a |
| A hit IDENTICAL to its cold read on every prompt under B3 and B4 (the B3 rule's line e, ESTIMATE) | 9 of 9 | B3 did not serve; under B4 alone identical 4 of 9, median first divergence 25: not met (D11) | b3a |
| The bench and the measure about 2 h 15 min on the eleven-line set, the users holding once (Amendment 89) | 2 h 15 | 91 min 13 s; 43 min 58 s; 2 h 15 min 11 s: met | bench 2026-11-19, second |
