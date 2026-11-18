# Changelog

What changed in the build, newest first. One entry per window and per decision. Times are UTC. The figures are in
[docs/measurements.md](docs/measurements.md); the story is in [docs/history.md](docs/history.md).

## 2026-11-18 (evening): second release

- **The first release's kits were published without their `files/models/` directory.** The repository's ignore
  rule for `models/` (meant for model weights) swallowed every kit's `files/models/deepseek_v41/` tree, so kits 6c, 7
  and 8 as published could not apply: their manifests name those files. This release re-includes `kit/*/files/models/`
  and restores them for every kit.

The record as it stands after window q1 and the closing bench: kits 9L, 9M and 9N (the serving kit), the nine adopted
lines, the windows p2s to q1 on this page and in [docs/measurements.md](docs/measurements.md), and the closing bench's
three tables (code, legal, prose; one to three streams; 1K to 256K; 175 W) in [README.md](README.md) and the
measurements page, with the prefix-sharing measure (22:38Z to 23:22Z; the same cells with the document already in the
cache: first token under a second at every length on a hit, one re-read per new stream, and a hit's tokens are not
the cold read's; Amendments 76 and 78) beside them. Kits whose windows have not run (the lean hand-off, the batched `wo_a` projection, the packed
expert records, the store's rebased kit) are not in this release.

## 2026-11-18, 21:01Z to 22:36Z: the closing bench on the serving set (Amendments 74 and 77)

On the set window q1 left serving (kit 9N, the nine lines, four slots, 175 W, the 1,452 MHz ceiling): 1,024 / 8,192 /
32,768 / 65,536 / 131,072 / 262,144 prompt tokens x code, legal, prose x one, two and three streams released
together x two repeats = 108 groups, 216 measured requests plus one warm-up; greedy, 1,024 tokens out, a fresh cache
salt per stream; the tool re-checks the serving record, the kit, the adopted digest and the caps between groups; a
missing, extra or truncated cell is INCONCLUSIVE, never a figure. Judge: PASS completeness; 94 min 36 s. One stream at
128K: code 161.1 tok/s (prefill 4,444 tok/s), legal 86.9 (3,794), prose 87.3 (4,033); at 256K the first token comes
after 63 to 64 s on every kind. Three streams together, the sum of their decode rates once every prompt is in: code
364 to 432 tok/s, legal 205 to 248, prose 178 to 213 by length; over the joint window the aggregate stays near one
stream's rate at 8K and longer because prompts are read one after another. PREDICTED: code one stream at 128K about
133 tok/s (wrong, low: 161.1); prefill 4,400 to 4,600 at 61K to 128K (met on code, not on legal and prose); about
4,100 at 256K (met); three streams at 1K about 290 in sum (wrong, low on code; no kind was named). The tables are in
the measurements page and README.md.

## 2026-11-18, 20:31Z to 20:48Z: window q1, the quality window: two boots without the drafter give the same text (Amendments 73 and 75)

From the serving state, two independent one-slot boots without the drafter (Qa, then Qb; the recorded profile's
derivative with the speculative block removed, no hold, no turn chunks) each took the same eight temperature-zero
forced cells (four fresh public prompts near 4.2K, 94K, 128K and 252K tokens, each followed by the same 900-token
turn, 205 tokens out). Both boots clean (speculative_config=None, zero out-of-memory and traceback lines, the card
limit and zero cold experts on twelve workers, highest card sample 30,020 MiB in both). QUALITY PASS: 8 of 8 complete
forced replies have equal text digests. The Engram precision item (Amendment 35) is INCONCLUSIVE by construction, not
a pass: no verified local extraction of the official host-memory FP8 Engram table (revision 1bae0e10) exists in the
trees. The way back on the matching record: the serving set unchanged since 20:48:45Z. 17 min 44 s without the
serving port (ESTIMATE 15 to 23).

## 2026-11-18, 19:35Z to 20:19Z: window p6b, THE TENSOR-CORE INDEX SCORER ADOPTED; the build serves kit 9N with nine switches (Amendments 69 and 72)

The same window as p6 under the corrected line C. The reference boot (the serving scorer) identical 7 of 7 to the
accepted capture; the TC boot identical 7 of 7; line C under the corrected judge 6,641 scorer steps on every worker
= 6,641 full-width verification steps in the drafter's own metrics; the round at about 128K: L1 25.829 against
26.633 ms (-0.804 ms, -3.02 %), LM 25.719 against 26.677 (-0.958 ms, -3.59 %); the short cells within 0.07 ms;
per token L1 -4.50 % and LM -3.17 % repeat for repeat; the streamed text equal to the reference's in 22 of 22
repeats; the 128K cold read's prefill 4,587 against 4,582 tok/s; four together on fresh text 337.9 against 316.6
tok/s in sum (+6.7 %); highest card sample 30,814 MiB; Xid 0. Adopted: the full check on the test port (identical
7 of 7, the rehearsal, the lone requests of Amendment 64), the record written 20:16:46Z (kit `PP3KIT9N-over-cut`),
the serving boot at 20:19:16Z with `VLLM_V41_INDEX_TC=1` beside the eight lines of window p5u: NINE adopted lines.
44 min 16 s without the serving port (ESTIMATE about 50). Amendment 69's round prediction ("about 1.2 ms shorter")
was too large: 0.80 and 0.96 ms; the two windows' reference boots differ by about 0.4 ms a round, the A/A band to
carry for that cell.

## 2026-11-18, 18:25Z to 18:58Z: window p6, the tensor-core index scorer RAN and WENT BACK on a wrong line (Amendments 61, 66, 67 and 69)

Kit 9N (kit 9M plus `index_tiled.py` and `attention_ops.py`; the v41idx3 library built on the host in 49 s). The
lines boot passed its gate (twelve "library loaded", twelve "active"); the reference boot passed every line (prefill
4,593 tok/s at about 127,663 tokens, decode after a cold read 137.6 tok/s, after a one-step turn 125.2); the TC boot
was identical 7 of 7 and faster per token (6.88 against 7.27 ms after a cold read, 7.63 against 7.99 after a turn)
and was then FAILED by line C, which compared scorer counter steps against the engine core's statistics answers.
The way back held (kit 9M, the eight lines); 32 min 34 s without the serving port. An independent reviewer read the
real logs: RULE WRONG. A counter unit is one scorer execution per full-width verification step of the drafter; an
engine-core "answer" is any completed token-bearing batch, prompt chunks included; "steps = answers" had been a
coincidence of one log's shape. The corrected line requires the counters' step count on every worker to EQUAL the
independent verification count from the SpecDecoding reports: on p6's own logs 6,637 = 6,637 on twelve workers,
PASS, and the round cells read from those logs passed the window's line R (L1 -1.250 ms, LM -1.226 ms). The judge's
patch carries tests with values computed by hand from the real lines; the bound was not moved to fit. The window was
rerun as p6b.

## 2026-11-18, 15:46Z to 16:52Z: windows p5t and p5u, THE TURN CHUNKS SERVE (Amendments 63, 64 and 65)

The rule: `VLLM_V41_PP_TURN_CHUNKS` adoptable at 3 only; the accepted capture a boot is held to is decided in one
place from the adopted lines: the turn chunks adopted means `accepted/quality-TC3.json` (window p5's seven long
prompts read cold on the `turn` boot, another rounding of the same arithmetic), otherwise window o2's capture. p5t's
check boot passed 0 cold, the card limit, the turn chunks at boot, 7 of 7 against TC3 (the first boot held to it,
with the hold and the token cache on), the rehearsal (30,814 MiB), and was refused by the after-requests witness of
`VLLM_V41_PP_EMPTY_AHEAD`: its first-use line prints once per boot at the first decode after a one-step prompt with
nothing else in the pipeline, which the four-chat rehearsal reaches only by the chance of its chats' ending order
(p4b's had passed by that chance). Not the turn chunks' fault; the prediction stands recorded as wrong in that line.
The rule changed in the check only: after the rehearsal it sends the lines phase's lone requests (a 334-token cold
prompt among them), and the witness judges after them. Under that rule (p5u): the check boot identical 7 of 7 to
TC3, rehearsal high 30,814 MiB, the lone request printed the line in the same second, the record written, the
serving boot at 16:51:49Z with EIGHT lines. 11 min 29 s without the serving port. The owner's cold 425,784-token read
after: first token 102.73 s, 4,145 tok/s (p4b's set: 104.46 s, 4,076): the same within the band. Lesson recorded: a
witness that waits for a one-time line must send the request that produces it.

## 2026-11-18, 14:33Z to 15:07Z: window p4b, THE TOKEN CACHE SERVES (Amendments 49, 56, 57, 60 and 62)

Window p4 rerun on kit 9M with the six adopted lines. `verify`: 32 verify lines for 32 answered chat-route requests,
every one "identical", 20 of 20 turns hits reusing 127,659 to 135,958 ids; the three edits (inside an earlier
message, at a delimiter either side) hits with 135,127 reused, identical. `one` against `off`, 27 cells of 27:
prompt tokens equal 27 of 27; the reply's text digest and tokens out equal 27 of 27; the turns' front cell for cell
-256.8 ms at the median (308.0 ms against 47.2; 0 of 20 turns not faster; the line asked 150 ms); the misses'
front -49.8 ms at the median (the line allowed +30); line S exact (hits 23, misses 77, ids reused 3,041,989,
different 0 at the 100th call). Adopted: the check 15:04:22Z, the serving boot 15:06:52Z with SEVEN lines
(`VLLM_V41_PREFIX_TOKEN_CACHE=1` new). 33 min 37 s without a server. PREDICTED about 250 ms at the front: met.

## 2026-11-18, 13:37Z to 14:32Z: window p5, THE PROMPT-END HOLD SERVES; the turn chunks measured (Amendments 59 and 62)

Kit 9M, 175 W and the 1,452 MHz ceiling (in force since 12:09:23Z). The gate boot collected both switches' real
lines first. `off`: turns alone on a 128,000-token chat, first token (medians of two) +880 1.184 s, +2,048 2.061 s,
+4,200 3.687 s, +10,000 4.963 s; beside a 248,000-token read (medians of three) the turn's first token 3.33 s,
whole turn 11.99 s, the read's first token 62.66 s. `hold` (`VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD=128`): 7 of 7,
the hold line 3 of 3, beside the read the WHOLE TURN 5.12 s against 11.99 (-57 %; the line asked -50 %), the
read's first token 60.99 s (not slower). `turn` (`VLLM_V41_PP_TURN_CHUNKS=3`, measured only): each request's cut
the rule's cut; first token of a turn alone +880 1.131 s, +2,048 1.471 s, +4,200 2.319 s (63 % of `off`; the line
asked at most 75 %), +10,000 3.623 s; its capture of the seven long prompts kept as the candidate TC3. Adopted: the
hold, by the check and the serving boot at 14:32:24Z (six lines). 54 min 37 s without a server (ESTIMATE 67).

## 2026-11-18, 13:30Z to 13:37Z: window p4 stopped at its first boot on a defect of the probe (Amendment 60)

The `verify` boot passed its witness and 7 of 7; then the token probe printed its usage and exited 9: it parsed its
options with another probe's parser, which refuses a value of 0. The dry runs stub the probe, so no simulation met
it. Fixed with a test from the usage line; the rule unchanged; rerun as p4b. 7 minutes without a server. The first
window at 175 W and the 1,452 MHz clock ceiling (the owner's setting of 12:05Z, in force since 12:09:23Z).

## 2026-11-18, 10:00Z to 11:58Z: window p3c, five decode boots PASS at 200 W; THE POWER TRIP; p3r at 150 W: kit 9M with `ahead`, `nosleep` and `gemv2` SERVES (Amendments 54, 55 and 58)

p3b (08:58Z to 09:25Z) had stopped at `off`: the wait's line "within 2 %" failed a boot for being 2.02 % FASTER
on one cell; a no-harm line must be one-sided (`notlonger:2`, `noloss:2`; the owner's defect of form, Amendment
54), and `ahead`'s "lines apart" clause was changed from two extremes to medians against extremes (Amendment 55).
p3b measured what the build lacked: two boots of one engine repeat the decode round within 0.6 %. p3c, all five
boots at 200 W: `base` and `off` within 0.5 % (the receive-buffer wait costs nothing); `ahead`
(`VLLM_V41_PP_EMPTY_AHEAD=1`) against `off`: the round S1 24.235 against 25.392 ms (-1.157 ms, -4.55 %), L1 26.985
against 28.164 (-1.179 ms, -4.19 %), the controls within 0.6 %, the loop's counts 64 of 64 with 64 sleeps and 0
takes, answer to submission 1.40 ms (was 2.84): PREDICTED 1.14 to 1.27 ms, met. `an` (`VLLM_V41_PP_ANSWER_NO_SLEEP=1`)
and `ang` (`VLLM_V41_DRAFT_GEMV2=1`) no loss; `ang` against `off` S1 -1.279 ms, L1 -1.338 ms; `ang` at 128K: decode
after a one-step turn 170.0 tok/s (5.88 ms a token) and after a cold read 137.9 tok/s on forced 1,000-token
sequences, prefill 4,564 tok/s. In the adopted set's check, under four long chats, the host ENDED at 11:15:48Z
with no shutdown sequence and no kernel line: read as a power trip at 200 W; the owner's rule of 2026-11-17 applied,
caps to 150 W at 11:45Z; the window continued as p3r: the check PASS 11:55:49Z (high 30,714 MiB), the serving boot
11:58:19Z with FIVE lines (rel, mid, ahead, nosleep, gemv2) at 150 W. The owner's 425,781-token read: first token
152.03 s (2,801 tok/s), against 100.8 s at 200 W in p2: long prompts about a third slower at 150 W. Defect found: the
restart tool for a power trip could not have started the build (it required kit 6c); the caps step ran alone and the
window continued by hand.

## 2026-11-18, 07:54Z to 08:05Z: window p3l, line discovery on kit 9M; the index capture (Amendments 51 and 52)

Kit 9M = kit 9L with the wait's line a plain once-per-process line (kit 9L's `info_once` never printed on this
build) and the statistics line printing the ROUND (answer to answer). One boot with every decode switch on, four
short requests, nothing judged: every line in a form the builders' table had; the wait's and the drafter kernel's
lines print AT BOOT (the warm-up reaches them). First round figures printed by a server, short prompts with the
switches on: medians 23.976 to 24.237 ms; answer to submission 0.552 ms (2.83 without the switches after a
one-step read). The capture boot wrote one decode step's index operands on 12 of 12 workers (20 files, 133 MiB)
for the tensor-core scorer's design. 10 min 22 s without a server.

## 2026-11-18, 06:30Z to 06:47Z: window p3, decode first, stopped at its first boot by the witness (Amendments 46, 48 and 50)

Kit 9L (kit 9 plus `v1/engine/core.py` with `VLLM_V41_PP_EMPTY_AHEAD`, `VLLM_V41_PP_ANSWER_NO_SLEEP` and the
diagnostic `VLLM_V41_PP_LOOP_STATS`). `base` passed every measuring line (identical 7 of 7: kit 9L's bytes give the
accepted answers with every switch off) and was failed by the witness, which asked for the wait's line with the value 0
from the middle workers: the engine prints no such line (the sixth stop by a check that had not met a real log).
What the boot measured: THE LOOP'S OWN COUNTS confirm the time budget's reading on the cards: after a one-step read
no empty step in 0 of 64 answers, 128 sleeps, 64 blocking draft takes, 2.83 ms from the answer to the next
submission; after a many-step read 64 of 64, 64 sleeps, 0 takes, 1.41 ms. Four together on fresh text 316.6 and
323.6 tok/s in sum, one request alone 130.5 and 131.0; prefill 4,570 tok/s at 127,663 tokens (200 W). Decode
comparisons became repeat for repeat (Amendment 48) because each repeat is its own text and tokens a round differ
with the text (26 % between repeats of one cell). 16 min 26 s without a server.

## 2026-11-18, 04:56Z to 05:57Z: window p2, the middle-stage skip PASSES and is adopted with the release setting (Amendments 40, 43, 45 and 47)

Kit 8, START A (`rel` serving). `off`: reads 3,810 / 3,696 / 3,284 / 2,757 tok/s at 60,704 / 93,027 / 247,952 /
425,776 tokens. `mid` (`VLLM_V41_PP_MID_PREFILL_SKIP=1`, measured for the first time): identical 7 of 7; reads
+10.4 / +16.9 / +31.0 / +49.6 % (the line needed +5 % at 93K); the turn beside a 250,000-token read: first token
3.01 s against 3.61, the read's first token 61.86 against 86.96 s. PREDICTED +3 to +8 %: wrong, low by half and
more; the gain grows with length. `adopt` (both switches, the first time two switches met a card): reads +19.7 /
+22.1 / +34.1 / +53.2 % (4,562 / 4,515 / 4,405 / 4,223 tok/s), 7 of 7, the rehearsal high 30,394 MiB. Report lines,
one request alone, about 128,000 tokens of public text, 200 W: `adopt` prefill 4,426 tok/s (127,678 tokens in
28.85 s), decode after the cold read 124.3 tok/s, after a one-step turn 112.0 tok/s (answers of different lengths
and text: not a clean pair; window p3 measures decode on fixed sequences). Adopted by the check and the serving boot
05:57:17Z (two lines). 61 minutes without a server.

## 2026-11-18, 04:17Z to 04:30Z: window p2s, THE RELEASE SETTING SERVES; the first serving check on a card (Amendments 41 and 44)

`rel` (`VLLM_V41_EMPTY_CACHE_TOKENS=8192`) could not be switched on by a setting: the record did not know the value.
The serving end was built (`ADOPTED.env`, written from an allow-list by kit; the record's `adopted_sha`; an adopt
chain) and run alone: kit 8 in, the check on the test port (its first run on a card: the API server's environment
holds 8192, the release line from no worker, identical 7 of 7, the rehearsal high 30,394 MiB), the serving boot
04:29:52Z with ONE adopted line. 12 min 27 s without a server. The owner's cold read of 425,782 tokens on the
serving port right after: first token 128.04 s, 3,325 tok/s, high 30,472 MiB, no out-of-memory line (p1's `off`
boot: 173.5 s). PREDICTED 150 to 175 s: wrong, faster.

## 2026-11-18, 03:49Z: the order of the remaining work (Amendment 42)

By time saved on a real task per hour of work and per minute without a server: p2s, p2, then DECODE FIRST (kit 9's
host-loop change, the drafter's faster GEMV, the receive-buffer wait, the token cache), then kernels where window
t1's traces give at least about 0.3 ms a round or 2 % of a chunk (the dense path, the index at long context, the
prompt-read gaps C1 and C2; closed by the traces: the static hand-off, the shared-expert overlap, the one-launch
expert layer, decode attention at one row), then kit 8's scheduler switches, the store, a soak, the Engram precision
test last.

## 2026-11-18: first public release

The build's record as it stands after window p1: the engine kits, the measurements, the architecture, the history,
the kernel and optimization plan, the time budget, three reviews and the tools whose cleaning is done. What is left
out is in [NOT-YET-PUBLISHED.md](NOT-YET-PUBLISHED.md).

## 2026-11-17 23:30Z to 2026-11-18 00:07Z: window p1, kit 8 meets the cards

With its switches off kit 8 answers the accepted prompts token for token as kit 6c.
`VLLM_V41_EMPTY_CACHE_TOKENS=8192` reads fresh text 12.6 % faster at 61,000 tokens and 9.9 % at 93,000 with
identical answers and no higher card memory: adoptable. The middle-stage skip was not measured: the arm's witness
failed its boot on vLLM's warning about an unregistered variable. First valid figure for a turn of 4,200 new tokens
on a 128,600-token chat: 3.31 s of reading.

## 2026-11-17 21:20Z to 22:40Z: the time budget read; an independent review; strict judges

The trace analysis of window t1 (`research/TWELVE-CARD-TIME-BUDGET.md`). An independent review found that the token
cache returned wrong ids after an edit at its cut (held, repaired, re-proven on CPU), that the equality judges
accepted cut answers (made strict; the historical identity results stand), and that for long agent work one
request's decode is the largest wait (the order of work changed to decode first).

## 2026-11-17 20:24Z to 20:44Z: window t1, the time budget captured; the split switch inconclusive

Arm T captured all 120 traces. Arm G (`TM_GEMM_TUNE=max_splits=1`) passed its delivery line and ended INCONCLUSIVE
at its quality line: the switch stays out.

## 2026-11-17 18:58Z to 19:30Z: 200 W; the move to the public port; arm G without a verdict

All twelve V100s at 200 W (prompt read +13.7 %, decode within 2 %, one run). The build moved to the default's
port in 3 minutes with the new serving form. Arm G's first run stopped on a witness that read an environment a
titled worker blanks.

## 2026-11-17 18:14Z: the default server

The twelve-card build became the owner's default DeepSeek server; the eight-card build is the fallback. The serve
phase was split into checks on a test port with a record, and a serving boot that sends no request.

## 2026-11-17 15:47Z to 18:05Z: the store windows st3, st3b and st4; the split-count cause

st3 stopped at its first boot on the driver's own rank count (fixed). st3b stopped on a question of the build's own
reads: the same tokens in a tail chunk of another size gave other layer-0 pages. A read-only review traced it to the
dense library's split count, and a one-card probe confirmed it in 25 seconds. st4: every server test of the store
passed at four slots; a restored turn is not token-identical to a cold read (that rule line was wrong and stays
FAILED).

## 2026-11-17 12:42Z to 13:48Z: store window st1; four slots (window s4)

st1: the store's kit changes nothing with the store off; the store's own fingerprint check refused the first store
boot (fixed in the store's kit). s4: four slots passed every line and stand.

## 2026-11-17 10:53Z to 11:20Z: the card limit (window k1); the step 0 probe

The card limit changed no token and no memory mark; a counting defect of the driver cost a second boot. The probe
showed large four-card sums going through NCCL in half precision in an order that depends on the buffer's size.

## 2026-11-17 07:55Z to 10:16Z: windows o1 and o2, three slots

o1 closed three slots on a mis-scaled long-prompt limit; o2's corrected rule accepted three slots with 4,092-row
chunks (288.0 tok/s in sum for three together). Kit 7 closed.

## 2026-11-17 00:16Z to 07:12Z: windows w1 to w5

w1: the first three-stage boot answered wrongly; the cause, the scheduler's spacing, was found on a CPU and fixed in
kit 3 (window w3). w3: every routed expert in NVFP4 booted for the first time. s1: slot counts on the quantized mix.
w4: kit 6b did not load; kit 6c (w5) did.

## 2026-11-16: the plan

The layout (three stages cut at layers 14 and 28, the layout of 1Cat-vLLM #1031), the first kit and its acceptance
rule, fixed before any data.
