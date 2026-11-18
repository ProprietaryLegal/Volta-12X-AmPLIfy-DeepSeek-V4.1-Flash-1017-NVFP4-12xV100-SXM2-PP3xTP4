# Volta 12X AmPLIfy

DeepSeek-V4.1-Flash-1017 served on twelve Tesla V100-SXM2 32 GB cards from 2018, with every routed expert at
NVIDIA's NVFP4 precision on the cards. The model runs as three pipeline stages of four tensor-parallel cards. All
15,360 routed experts are held in card memory: none is compressed and none is read from host memory.

This is the first public release of the build's record: the engine change as install kits, the measurements with
their conditions, the architecture, the history of every test window, the kernel and optimization plan, the time
budget and the tools whose cleaning is done. What is not yet published is listed in
[NOT-YET-PUBLISHED.md](NOT-YET-PUBLISHED.md). It will be updated.

Built by **Nick Mermiges, Esq.**, founder of [PLI Labs](https://proprietarylegal.ai/labs.html) · [proprietarylegal.ai](https://proprietarylegal.ai)

[What this is](#what-this-is) · [What runs](#what-runs) · [Measured](#measured) · [What worked](#what-worked) ·
[What did not work](#what-did-not-work-or-is-not-yet-good) · [In progress](#in-progress) · [Next](#next) ·
[Hardware](#hardware) · [Running it](#running-it) · [Repository layout](#repository-layout) ·
[Related builds](#related-builds)

Every figure on this page is labelled. **MEASURED** means a log or probe on our hardware, with its test window named
(windows are short labels such as `o2`, `s4`, `t1`, `p1`; dates and UTC times are in
[docs/measurements.md](docs/measurements.md)). **ESTIMATED** means arithmetic or a model, with its basis.
**UPSTREAM** means someone else's number. A figure from one run says so.

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
  four conversations kept warm and a context of 524,288 tokens. One request decodes at about 128 to 130 tok/s; the
  pipeline carries one request per step, so several requests at once add total throughput only up to three stages
  (four together: 306.6 tok/s in sum at 150 W). It is not a batched-decode engine.
- **What it is tuned for.** Long-context agent work: one request at a time, prompts of 60,000 to 200,000 tokens that
  grow by small turns. Since 2026-11-17 it is the owner's default DeepSeek server.

## What runs

The standing configuration.

| Item | Value |
|---|---|
| Model | DeepSeek-V4.1-Flash-1017: 40 layers, 384 routed experts per layer, DSpark drafter |
| Routed experts | 15,360 in NVFP4 on the cards: 5,376 / 5,376 / 4,608 by stage; 0 in host memory |
| Non-expert tensors | the ISTA-DASLab GSQ-RCO GGUF |
| Layout | PP3 x TP4: three pipeline stages of four tensor-parallel cards, layers 0-13 / 14-27 / 28-39 |
| Slots | four (accepted in window s4); three remain available |
| Prompt chunks | 4,092 rows (`max-num-batched-tokens 4096`) |
| Context | 524,288 tokens |
| Card limit | 31,998 MiB on every stage (`VLLM_V41_CARD_LIMIT_MIB=31998`, a fuse at the owner's line) |
| Engine | 1Cat-vLLM's DeepSeek-V4.1 package as carried in PLI Labs' engine (commit `f20d155`) with kit 6c over it (engine `e7b2417`, 11 files): [`kit/PP3KIT6c-over-cut`](kit/PP3KIT6c-over-cut) |
| Power and clock | 200 W per card since 2026-11-17 18:58Z (150 W before), clock cap 1,477 MHz |
| Serving | every start check runs on a test port and is recorded; the serving boot on the public port sends no request and needs a record that matches the installed bytes ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), section 13) |

Not in what serves: the conversation store switched on (its whole server test list passed in window st4), the
dense library's split switch (inconclusive), and kit 8 (see "In progress").

## Measured

All MEASURED on 2026-11-17 and early 2026-11-18.

### Speed

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
- **A release setting that reads faster.** `VLLM_V41_EMPTY_CACHE_TOKENS=8192` read fresh text 12.6 % faster at
  61,000 tokens and 9.9 % at 93,000, with identical answers and no higher card memory (window p1, one boot). It is
  adoptable and not yet in what serves.
- **200 W.** The prompt read rose from 3,284 to 3,733 tok/s; decode moved by under 2 % (one run).

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
  3,200 edits, 0 sequences different from a full encode). It has not met the cards.
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

## In progress

- **Kit 8** ([`kit/PP3KIT8-over-cut`](kit/PP3KIT8-over-cut)): with its switches off it is kit 6c on the cards
  (window p1). Its switches are NOT on the serving build: the middle-stage prompt skip (`VLLM_V41_PP_MID_PREFILL_SKIP`,
  not yet measured), the turn pipeline (`VLLM_V41_PP_TURN_CHUNKS`) and the hold of decode steps after a prompt's end.
- **The release setting** (`VLLM_V41_EMPTY_CACHE_TOKENS=8192`) passed its arm and waits for the adopt boot.
- **The token cache** (repaired on CPU) and the drafter's faster kernel (`VLLM_V41_DRAFT_GEMV2`): not on the cards.
- **The conversation store** switched on in what serves: after a paired on/off comparison of the first token.

## Next

The order of work is section 6 of [documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md](documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md):
decode for one request first, then the first read and the first token of a turn, then no long context read twice
(the store), then one rounding, and batched decode for several requests last.

## Hardware

- A twelve-V100-SXM2 server: three NVLink boards of four 32 GB cards (32,510 MiB each). Tensor parallelism stays
  inside one board; the pipeline hand-off between boards goes over PCIe. Two boards sit behind one PEX88096 switch,
  the third behind a PLX 8749. The first hand-off stays inside the PEX switch; the second crosses host bridges.
- Volta has FP16 `mma.sync.m8n8k4` only: no FP4 or FP8 tensor cores. Every 4-bit format is decoded in software into
  FP16 operands, so low-bit experts pay by reading fewer bytes, not by faster math.
- 150 W and 1,477 MHz per card until 2026-11-17 18:58Z; 200 W since, with the same clock cap.

## Running it

[docs/OPERATIONS.md](docs/OPERATIONS.md) says how the build is installed, started, checked and stopped. The window
driver and launcher that ran our windows are not in this first release (they name our host's layout); the
operations page describes what they do so that the steps can be followed with your own scripts.

## Repository layout

| Path | What it is |
|---|---|
| `README.md`, `CHANGELOG.md` | this page; what changed, by date |
| `NOT-YET-PUBLISHED.md` | what this first release leaves out, and why |
| `docs/` | [architecture](docs/ARCHITECTURE.md), [operations](docs/OPERATIONS.md), [measurements](docs/measurements.md), [history](docs/history.md), [upstream contributions](docs/upstream-contributions.md); [index](docs/README.md) |
| `documents/plans/` | the [kernel and optimization plan](documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md) |
| `documents/reviews/` | the [upstream review](documents/reviews/UPSTREAM-REVIEW-20261117.md), the [chunk-boundary reading](documents/reviews/CHUNK-BOUNDARY-READING.md) |
| `research/` | this build's own findings ([TWELVE-CARD-RESEARCH.md](research/TWELVE-CARD-RESEARCH.md)) and its [time budget](research/TWELVE-CARD-TIME-BUDGET.md) |
| `engine/` | [README.md](engine/README.md): every kit with its engine commit, what it is and its state |
| `kit/` | the engine change as install kits (`apply.sh`, `rollback.sh`, `MANIFEST`, `files/`). `PP3KIT6c-over-cut` is the serving kit |
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
