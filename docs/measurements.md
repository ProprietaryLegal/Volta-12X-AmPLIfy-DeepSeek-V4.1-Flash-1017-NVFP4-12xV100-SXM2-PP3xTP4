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
| Power and clock | 150 W and 1,477 MHz per card, MEASURED 2026-11-16 23:06Z to 23:14Z, through window sv1 (2026-11-17 18:16Z). **200 W** and 1,477 MHz from 2026-11-17 18:58Z (the limits service's log showed all twelve V100s "power limit set to 200W") to 2026-11-18 11:15:48Z, when the host ended under four long chats with no shutdown sequence and no kernel line (read as a power trip). **150 W** from 11:45Z. **175 W and a 1,452 MHz clock ceiling** (cards at 1,447 MHz) since 2026-11-18 12:09:23Z: every figure of windows p4 to q1 and of the closing bench | PLAN section 2; Amendments 28, 58 and 62 |
| Layout | three stages of four cards, layers 0-13 / 14-27 / 28-39 | PLAN section 1 |
| Dates | windows w1 to t1 ran on 2026-11-17 (UTC); p1 crossed midnight; p2s to q1 and the closing bench ran on 2026-11-18 | PLAN section 9 |
| "The default" | in the window names below, the server that was restored at the end of a window before 2026-11-17 18:14Z; a window stopped it, ran its arms and restored it | PLAN section 9 |
| The line | 31,998 MiB: the owner's limit for the highest sample on any card | PLAN section 9, line 6 |

Every figure of a window up to and including sv1 was taken at 150 W (PLAN Amendment 28 says so of "every speed
figure of this build before 18:58Z"). The 200 W figures are marked "200 W" on this page: the serving build's
baseline after window g1, windows mv1, g1, t1, p1, p2s, p2, p3, p3l, p3b and p3c. Windows p4 to q1 and the closing
bench are 175 W figures; p3r's are 150 W. TODO(owner of the repo): the
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
94 min 36 s on the serving port with no other traffic (ESTIMATE was 47 to 111 minutes). The tables are the section
"Closing bench, 2026-11-18" below.

## Closing bench, 2026-11-18 (window q1 serving set)

MEASURED on 2026-11-18, 21:01:56Z to 22:36:32Z, on the serving set of window q1 (kit `PP3KIT9N-over-cut`, the nine adopted lines, four
slots), 175 W per card and the 1,452 MHz clock ceiling: code, legal and prose prompts of 1,024 / 8,192 / 32,768 / 65,536 / 131,072 /
262,144 tokens, read cold, one, two and three streams released together, two repeats each (108 groups, 216 measured requests; greedy,
1,024 tokens out, thinking off). "Prefill tok/s" is the prompt's tokens over the time to its first token. The two aggregate columns:
**Total decode tok/s** is all streams' output tokens over the joint window, from the first request's start to the last request's end;
the prompts are read one after another, so with long prompts it stays near one stream's rate. **Sum of decode rates** adds each
stream's own decode rate while all of them stream: the decode throughput once every prompt is in.

### Code

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

### Legal

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

### Prose

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

### Code, with prefix sharing

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

### Legal, with prefix sharing

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

### Prose, with prefix sharing

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
