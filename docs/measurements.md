# Measurements

Every measured figure of the twelve-card build, with its window and conditions. The source of each figure is the
window's result entry in the build's plan (section 9; "Amendment N" names an entry; the plan is not in this first
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
| Power and clock | 150 W and 1477 MHz per card, MEASURED 2026-11-16 23:06Z to 23:14Z, through window sv1 (2026-11-17 18:16Z). **200 W** and 1,477 MHz since 2026-11-17 18:58Z: the limits service's log showed all twelve V100s "power limit set to 200W" | PLAN section 2; Amendment 28 |
| Layout | three stages of four cards, layers 0-13 / 14-27 / 28-39 | PLAN section 1 |
| Dates | every window ran on 2026-11-17 (UTC) | PLAN section 9 |
| The default | the eight-card build Volta Max uPLIft, two stages, three slots, 5,651 of 15,360 routed experts in NVFP4; measured in the same window with the same tool where a row compares. It was the serving default until 2026-11-17 about 18:14Z and is the fallback since; this page keeps the windows' word "the default" for it | PLAN section 1; Amendment 27 |
| The line | 31,998 MiB: the owner's limit for the highest sample on any card | PLAN section 9, line 6 |

Every figure of a window up to and including sv1 was taken at 150 W (PLAN Amendment 28 says so of "every speed
figure of this build before 18:58Z"). The 200 W figures are marked "200 W" on this page: the serving build's
baseline after window g1, and windows mv1, g1, t1 and p1. TODO(owner of the repo): the
clock cap of 1,477 MHz is recorded for 2026-11-16 and for 18:57Z on 2026-11-17; that it held in between is
inferred, not read.

Terms. **Fresh text**: the requests F1, F2, F3 of the owner's cells and prompts made the same way. **Repeated
text**: the copy tasks RA, RB, RC and prompts made the same way. **Arm X**: three stages with the default's own
expert mix (5,651 hot), used to compare the three-stage engine with the default. **Arm F2**: every routed expert
NVFP4, two slots, 8,192-row prompt chunks. **Arm F3c**: every routed expert NVFP4, three slots, 4,096-row chunks:
the configuration of windows o2 and k1. **Arm F4c**: the same with four slots: the standing configuration since
window s4. **Arm G**: the standing configuration with the dense library's split switch `TM_GEMM_TUNE=max_splits=1`.
**Arm T**: the standing configuration with the engine's timers and the torch profiler (the time budget). **Boots
KA, KD1, KD2, KN, KN2**: the store windows' boots (store off; on with its dump on an empty root; the restart; on as
it would serve; its restart). **Reference N2**: the all-native two-stage arm on eight cards (what does not fit a
card is read from host memory). **Closeness**: `tools/closeness.py` on twelve short prompts; "first divergence" is
the first token where two captures differ; "mean |dlogprob|" is the mean absolute difference of log-probabilities.
A first-run speed figure carries 2 to 3 ms of noise per round; the second pass is the reading (PLAN section 9,
line 5).

## Summary: the standing configuration (four slots)

Three stages, every routed expert NVFP4, four slots, 4,092-row prompt chunks (`max-num-batched-tokens 4096`), kit
6c, card limit 31,998 MiB, profile `run/serve-nvfp4-pp3-full-c4096-a4.yaml` (PLAN Amendment 19). Since window st3
the store's kit is in the trees with the store off; since 2026-11-17 18:14Z the configuration is the owner's default and
serves on `:18042` (Amendment 27). The table below is window s4, arm F4c, one boot, at 150 W. The 200 W figures
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

Not measured at four slots: the default by the same tool in the same window (the three-slot comparison of o2 is
the last one) and a serving run of hours under a rule. The boots since s4 with the store's kit in the trees and the
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

## Summary: three slots (the configuration of windows o2 and k1)

Three stages, every routed expert NVFP4, three slots, 4,096-row prompt chunks, kit 6c, card limit 31,998 MiB,
profile `run/serve-nvfp4-pp3-full-c4096-a3.yaml`. It was the standing configuration from Amendment 13 to
Amendment 19 and is still served by `SERVE_CONFIG=F3c`.

| What | Figure | Window |
|---|---|---|
| Experts | 15,360 hot, 0 cold on every rank | o2 and every boot since |
| Load, launch to ready | 134 to 135 s (about 37 s process start, 86 s model load, the rest graphs and warm-up) | o2, k1 |
| One request | 129.5 tok/s fresh text, 177.3 repeated text | o2, arm F3c |
| Three requests together, sum | 288.0 tok/s fresh, 444.6 repeated; the default by the same tool 198.2 and 299.8 (x1.45, x1.48) | o2, arm F3c |
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
| Short prompts against reference N2 | CLEAN; median first divergence 208 tokens, median mean \|dlogprob\| 0.0090. The default: 7 tokens, 0.0656 | o2 (also w3, w5, o1) |
| Each chat alone against three together | equal, six of six | o2 |
| Same configuration, long prompts repeated | identical 3 of 3 inside one boot and 3 of 3 against another boot (two-slot build) | after o1 |

## By window

### Baseline: the default on 2026-11-16 (PLAN sections 1, 2 and 7)

| What | Figure |
|---|---|
| One request, fresh text | 87.5 to 129.4 tok/s (cells F1 129.4, F2 87.5, F3 125.5); copy tasks 159 to 187 |
| Three requests together | 59 to 72 tok/s each on fresh text, 96 to 105 on copy tasks |
| Prompt read | 20K tokens in 7.97 s, 90K in 32.05 s, 181K in 65.3 s |
| Load | 183 s |
| Boot lines, per card, stage 0 / stage 1 | free before experts 28.76 / 28.73 GiB; reserve 4.48 / 7.19; expert budget 23.95 / 21.20; NVFP4 experts 3,529 / 2,122; compressed 4,151 / 5,558; free after load 3.95 / 6.88 |

### w1: first three-stage boot (00:16Z to 01:01Z; first kit; arm X)

| What | Figure |
|---|---|
| Hand-off probes | eight pairs PASS |
| Arm X boot | tier digests and 5,651 hot as predicted (2,423 / 1,886 / 1,342 by stage) |
| Rule line 1 | FAIL: 15 of 15 answers differ from the default's |
| Closeness, arm X against the default | first divergence at token 1 on 12 of 12 prompts, the default's gap 4.6 to 16.7 nat; token 0 right; three to five junk tokens, then the default's answer for 100 tokens or more |
| Speed | per-round time equal to the default's; three together +35 to +45 % |
| KV pool | 1,389,057 tokens at three stages against the default's 2,076,870 at the same 0.25 GiB |

### w2: diagnostic window (from 01:19:34Z; kit 2, the faulty cadence still in)

| Arm | Result |
|---|---|
| D0, two stages on the build's engine | CLEAN, 12 identical; alone-against-together check PASS |
| D2, Engram hand-over and middle commit only | JUNK, 12 of 12 |
| D1, mirror only | JUNK, 12 of 12 |
| X1, the build's layout, one slot | passed its boot checks; at 01:49:39Z the four cards of the first stage logged Xid 43 within 48 ms; the chain stopped by the crash rule |

PREDICTED at 01:34Z, before the three-stage results: D0 CLEAN, D1, D2 and X1 JUNK.

### w3: kit 3, the scheduling fix (about 02:1xZ to 03:01Z; then served to 04:27:12Z)

Arm X (three stages, the default's experts), pass 2, tok/s, against the default's pass 2 of the same window:

| Cell | Arm X | The default |
|---|---|---|
| Alone, fresh (F1 / F2 / F3) | 115.5 / 86.3 / 122.6 | 130.4 / 87.8 / 126.7 |
| Alone, repeated | 169.5 / 158.3 / 163.0 | 186.7 / 180.3 / 159.9 |
| Three together, fresh | 97.0 / 77.1 / 105.6 | 69.6 / 56.8 / 74.0 |
| Three together, repeated | 145.8 / 137.5 / 142.5 | 103.0 / 102.1 / 97.5 |

Arm X: rule line 1 PASS (closeness CLEAN, 0 of 12 early divergences; alone-against-together check PASS); highest
card sample 21,180 MiB. One-slot arm X1: CLEAN, no Xid.

Arm F2 (every routed expert NVFP4, two slots, 8,192-row chunks):

| What | Figure |
|---|---|
| Experts | 5,376 / 5,376 / 4,608 hot = 15,360; 0 cold on every rank |
| Free after load | 4.45 / 4.33 / 7.91 GiB by stage |
| One request alone | fresh 106.5 tok/s against the default's 109.3; repeated 170.8 against 172.5 ("no measurable change") |
| Three sent at once (two slots) | fresh +64 % (108.5 against 66.1 tok/s per request); repeated +51 % (152.7 against 101.0); mixed +56 % |
| Long reads, 20k to 364k tokens | 2,265 to 3,327 tok/s |
| Highest card sample | 31,808 MiB on the middle stage (31,010 / 31,808 / 30,018): 190 MiB under the line |
| Closeness | CLEAN against the default's capture (1 of 12 early) and against N2 (0 of 12) |
| The owner's score line (rule line 3) | INCONCLUSIVE (see w5) |

The serving arm P3w3-S (from 03:01:22Z, two slots):

| What | Figure |
|---|---|
| Boot, driver-free / allocator reserved, by stage | 4,587 / 27,124; 3,773 / 27,918; 5,259 / 26,444 MiB; 799 / 819 / 807 MiB outside the allocator |
| Fullest card under an external client's replay of test prompts | 31,778 MiB at 03:04:55Z; 31,836 at 03:33:36Z (58 MiB in half an hour); level for 38 minutes; 31,988 at 04:11:51Z (10 MiB under the line) |
| The owner's sampler, 04:14:14Z to 04:27:00Z | middle stage low mark flat at 29,758 MiB, highs 31,768 to 31,814; first stage lows 28,406 to 28,914, high 30,778; last stage lows 28,016 to 28,786, high 30,016 to 30,018 |
| At the stop | 0 tracebacks, 0 out-of-memory lines |

The step to 31,988 MiB is unexplained. No stage showed a floor that keeps rising.

### s1: slot counts on the quantized twelve-card mix (about 04:3xZ to 05:05:01Z; kit 4; arm X's experts)

Tool: `tools/slots_cells.py`. Two passes each; tok/s.

| Arm | Fresh text, sum | Repeated text, sum | Each request | Highest card sample |
|---|---|---|---|---|
| The default, three together | 198.1 / 198.4 | 300.3 / 298.9 | not in the source | not in the source |
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
| One request alone against the default | -3.4 % / -1.2 % (inside the night's spread) |
| Three sent at once against the default | +63 % / +51 % / +52 % |
| Highest card sample | 31,336 MiB (30,478 / 31,336 / 30,046). PREDICTED about 31,560. Window w3: 31,808 |
| Rehearsal of two grown chats (17 requests, largest prompt 153,658 tokens) | middle stage high 31,334 MiB: 664 MiB under the line; first-token medians 2.93 s and 5.18 s |
| Arm F2e, cache release at 512 rows | same peak (31,334 against 31,334 MiB); larger turns' first token 10.9 s against 5.2 s |

Closeness on w5's receipts (identical on w3's): arm F2 against N2: median first divergence 208 tokens, median mean
|dlogprob| 0.0090. The default against N2: 7 tokens and 0.0656. The two reference captures N2 and N2B: identical on
12 of 12. The owner's score line came out INCONCLUSIVE on both all-NVFP4 arms (index measure 12 for and 0 against;
mean measure 7 for, 0 against, 5 undecided of 12, with 8 needed). PLAN Amendment 10 replaced it.

### o1: first optimization window (07:55Z to 08:26:07Z; kit 6c)

| Arm | What | Figure |
|---|---|---|
| The default | three together, sum, by `slots_cells.py` | fresh 200.8 / 200.6 tok/s; repeated 301.8 / 300.7 |
| F2 | load; one read of 62,255 tokens; rehearsal | 139 s; 3,453 tok/s; middle stage high 31,334, low 29,440 MiB |
| F2 | closeness against N2; against the default | CLEAN; 208 / 0.0090 against the default's 7 / 0.0656 |
| F3c | boot | 15,360 hot, 0 cold on the lowered reserve (3,686 MiB); load 135 s; after warm-up 28,535 / 28,667 / 27,253 MiB |
| F3c | short prompts | CLEAN; 208 / 0.0090 |
| F3c | long prompts against arm F2 (rule line c) | CLEAN, but median mean \|dlogprob\| 0.0333 against 0.0126 allowed: the arm ended there (exit 34) |
| F3c against F2, per prompt | mean \|dlogprob\| | 0.0403 at 5,694 tokens; 0.0333 at 11,151; 0.0245 at 17,935 |
| F2 against the default, per prompt | mean \|dlogprob\| | 0.0184 / 0.0702 / 0.0117 |
| Serving boot P3o1-S (two slots) | rehearsal | highest card sample 31,422 MiB (30,474 / 31,422 / 30,038; 47 samples) |

The same two-slot configuration gave 31,334 MiB in w5's serve phase and in o1's arm F2: boot to boot the peak
moves by about 90 MiB.

Measured by the owner on the serving build after o1 (08:26Z to 08:33Z):

| What | Figure |
|---|---|
| Long prompts, same configuration | two captures of one boot identical 3 of 3; another boot identical 3 of 3: the noise floor is zero |
| Short prompts, arm F3c against arm F2 | identical 12 of 12 |
| Copy task RC, all-NVFP4 build | 1,500 tokens: 5,761 characters of reasoning and no answer, read-first and cold alike; 4,000 tokens: 534 of 534 words read-first, 536 of 534 cold (0.9981) |
| Copy task RC, the default | 1,500 tokens read-first: 5,286 characters of reasoning, then 52 words (the cells' 0.9808); 4,000 tokens read-first: 0.9981; cold: no answer at 1,500 and none at 4,000 (13,075 characters of reasoning) |

One request each for the copy task: an observation. The cells' RC score of 0.0000 for the all-NVFP4 arm in w3 and
w5 compared "reasoning ended just inside 1,500 tokens" with "just outside". It is not a loss of the build.

### o2: second optimization window (09:09:05Z to 10:08:51Z; kit 7 with its switch at the default unless named)

The default was down from about 09:18:45Z to 10:13:07Z: 54 minutes (ESTIMATED before: 49).

| Arm | What | Figure |
|---|---|---|
| F2 (reference) | closeness; load; one read of 62,258 tokens; rehearsal high | CLEAN against N2, 208 / 0.0090 against the default's 7 / 0.0656; 135 s; 3,410 tok/s; 30,476 / 31,336 / 30,020 MiB |
| F2s (chunks 64 rows shorter) against F2 | long set | the 5,694-token prompt identical; the six prompts over 8,188 tokens differ: median first divergence 50, median mean \|dlogprob\| 0.0307 (per prompt 0.0298, 0.0316, 0.0280, 0.0367, 0.0364, 0.0003). PREDICTED 0.02 to 0.05 |
| F3c | boot | 15,360 hot, 0 cold |
| F3c | short prompts | CLEAN; 208 / 0.0090 |
| F3c | long prompts against F2 (rule line c2) | HOLDS: median first divergence 44 against 50; median mean \|dlogprob\| 0.0331 against 0.0307; no early wrong token; largest gap at a divergence 0.531 nat |
| F3c | alone against together, three chats | PASS (six chats, each equal alone and together) |
| F3c | three together, sum | 288.0 tok/s fresh, 444.6 repeated; the default 198.2 and 299.8 |
| F3c | one request alone | 129.5 tok/s fresh, 177.3 repeated |
| F3c | prompt read | 3,286 tok/s (96 % of arm F2's) |
| F3c | memory | no out-of-memory line; highest card sample 30,052 MiB (29,578 / 30,052 / 28,646); after warm-up 28,535 / 28,667 / 27,253; outside the allocator 867 / 881 / 865. PREDICTED about 30,600 (30,300 to 31,000) |
| F3o (three slots, 8,188-row chunks, hand-offs awaited) | all lines | every line passed; long prompts identical to arm F2 7 of 7; sum 263.6 / 427.0 tok/s; read 3,463 tok/s; highest 31,664 MiB. PREDICTED 640 MiB under arm F3r |
| F3r (three slots, 8,188-row chunks; memory only) | highest card sample | 31,628 MiB. PREDICTED 31,846 to 31,932 |
| Serving boot P3o2-S (10:04:41Z) | rehearsal high | 29,540 / 30,048 / 28,610 MiB |

Long chats, from the rehearsal's own lines: see the summary table. Cold copy set at 4,000 tokens, one request each:
the build RA 584 of 584 words, RB 467 of 467, RC 536 of 534 (0.9981); the default RA and RB the same, RC no answer
inside 4,000 tokens.

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
| | the dump | layer 0's window pages of positions 25,616 to 25,759 on stage 0, all four ranks: a read with a tail chunk of 2,292 rows and one of 2,308 rows give nine pages of which six differ; on the eight-card default (tails 6,376 and 6,392) identical |

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

### sv1: the build serves as the default (18:05:01Z to 18:16:40Z; on `:18043`; 150 W)

Arm P3sv1-S (`chain_serve.sh sv1`): 0 cold, the limit on 12 of 12, the accepted capture identical 7 of 7, four
grown chats exit 0 with 30,364 MiB, first token 3.99 s and 10.67 s (PLAN Amendment 27).

### mv1: the move to `:18042` (19:17:01Z to 19:20:34Z)

`chain_move.sh mv1 sv1`: `stoparm S`, check, probes, `servestate` (the record matched), `servedefault` exit 0; no
request was sent before the build was declared serving. 3 minutes without a server (ESTIMATED before: about 4).
The first run on the cards of the two serving phases, the port copy and `stoparm S` on `:18042`. One gap:
`chain_split.sh` refused the window `chain_move.sh` made, which had no record of the fallback's digests; copied by
hand; the chains now carry it forward (PLAN Amendment 29).

### g1: arm G's first run (19:20:51Z to 19:27:13Z; 200 W)

Arm G booted with `TM_GEMM_TUNE=max_splits=1`: 0 cold, 15,360 hot, the limit on 12 of 12. Line a's witness: every
field in the log `max_splits=1` (API server, engine core, the four last-stage workers), no parse-failure line,
twelve titled workers, and the variable unset in `/proc/<pid>/environ` of all twelve: FAIL by the rule as written,
exit 33; lines b to g did not run. The library's own line "max_splits 1" stood 16 times in the arm's log and 0 times
in the logs of s4, sv1 and mv1, so the switch was delivered; a titled worker's environment is blank by
construction. NO VERDICT on the switch (PLAN Amendment 29). The build served on `:18042` again from 19:27:13Z.

### t1: the time budget and arm G's rerun (20:24:04Z to 20:43:34Z; 200 W)

`chain_budget.sh t1 g1 G`: the build stopped (cards 0 MiB 20:24:23Z), check, probes, arm T 20:25:06Z to 20:36:31Z,
arm G to 20:40:45Z, the build on `:18042` again at 20:43:34Z by `servedefault` on the matching record (PLAN
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
early wrong tokens of 12) and NEARER than P (first divergence 96 against 7; mean |dlogprob| 0.0079 against 0.0656),
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
