# Changelog

What changed in the build, newest first. Times are UTC. The figures are in
[docs/measurements.md](docs/measurements.md); the story is in [docs/history.md](docs/history.md).

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
