# Architecture

How the twelve-card build is laid out and why. Sources: the build plan
sections 2 to 7 and 9 (through Amendment 31), the kit `kit/PP3KIT6c-over-cut`, the launcher `run/swap_arm12.sh`,
the window driver `window/run_window_pp3.sh` and `documents/reviews/CUT-DEPENDENCE-GEMM-SPLIT-20261117.md`. The plan,
the launcher and the window driver are not in this first release ([NOT-YET-PUBLISHED.md](../NOT-YET-PUBLISHED.md));
the names are kept so that the record reads as it was written.
Labels: **READ** = read in code or a document named on the line; **ARITHMETIC** = computed from measured or read
values; **MEASURED** = a window on the cards, named; **ESTIMATED** = a model, with its basis.

## 1. The picture

```
   prompt tokens
        |
  +-----v------+  hand-off  +------------+  hand-off  +------------+
  |  stage 0   |  (PCIe,    |  stage 1   |  (PCIe,    |  stage 2   |
  | layers 0-13|  in one    |layers 14-27|  across    |layers 28-39|
  |  board X   |  switch)   |  board Y   |  bridges)  |  board P   |
  | 4 cards TP |----------->| 4 cards TP |----------->| 4 cards TP |
  +------------+            +------------+            +-----+------+
   embedding                 no embedding              drafter, head,
   Engram of layers 1, 14    5,376 experts             embedding
   5,376 experts                                       4,608 experts
                                                             |
                    sampled token, accepted count, drafts <--+
                    (back to the earlier stages, one step per stage later)
```

- **Tensor parallel inside a board.** Each stage is four cards on one NVLink board. Every matrix of a layer is
  split over the four cards, and their partial results are summed with an all-reduce inside the board.
- **Pipeline parallel between boards.** A stage runs its layers and hands the residual to the next stage over
  PCIe. There is no NVLink between boards.
- **One request moves through the three stages in three engine steps.** A step carries one request, so three
  requests can be in flight at once, one per stage. That is why the build gains with three requests and not with
  one, and why a fourth request at once adds little.
- **A long prompt read is already pipelined.** The engine keeps one prompt chunk step per stage in flight: a chunk
  counts as computed when it is scheduled, and the batch queue holds `pp_size` steps (READ, and RUN in a CPU model of
  the real engine loop and scheduler, 2026-11-17). A lone read is at its
  bound from scheduling.
- **A turn is not.** The new rows of a turn on a grown chat are fewer than one chunk, so they run all forty layers
  as ONE chunk through the three stages in series, so a turn reads far fewer tokens per second than a long read
  (window p1: a turn of 4,200 new tokens on a 128,600-token chat read for 3.31 s, about 1,270 tok/s, against about
  3,800 tok/s for a long read; ARITHMETIC from the MEASURED figures). For long-context agent work this is a large
  wait, and the target of kit 8's turn pipeline (section 12).

## 2. Why three stages, and why these cuts

| Fact | Value | Label |
|---|---|---|
| Card memory | 32,510 MiB = 31.75 GiB per card | READ, PLAN section 2 |
| One routed expert in NVFP4, per card of its stage | 4,874,240 bytes | READ `nvfp4_8r_native.py:74-75` |
| One layer's 384 experts, per card | 1.7432 GiB | ARITHMETIC |
| Twenty layers, per card | 34.86 GiB: more than a card | ARITHMETIC |
| All forty layers | 278.9 GiB, against 381 GiB on twelve cards and 254 GiB on eight | ARITHMETIC |

So two stages of twenty layers cannot hold every expert in NVFP4, and three stages can. The cuts follow from
three constraints (ARITHMETIC, PLAN section 3):

1. A stage must start at an index source, a layer that computes its own top-k selection: layers 2, 8, 14, 20, 24,
   28, 32, 36. Otherwise a selection would have to cross the boundary.
2. The last stage also carries the drafter and the head, so its reserve is larger (7.19 against 4.48 GiB in the
   default's lineage). It holds at most 12 layers: 28 to 39.
3. The remaining 28 layers split 14 / 14 only at layer 14.

The result is layers 0-13 / 14-27 / 28-39 (`VLLM_PP_LAYER_PARTITION=14,14,12`). It is the layout of the owner's
1Cat-vLLM pull request #1031: "Only legal stage cuts are accepted: 20 for PP2; 14 and 28 for PP3." Moving a cut by
one layer moves 1.743 GiB of experts per card and breaks constraint 1, so nothing is gained by moving it (PLAN
Amendment 5).

Layouts that were considered and not chosen (PLAN section 3): cuts 20 / 28 (the first board still holds twenty
layers: 73 % of experts in NVFP4); two stages plus the third board as an expert annex (two PCIe trips per layer);
one tensor-parallel group of eight or twelve (all-NVFP4 does not fit eight cards; twelve is not a legal group
size); an even 13 / 26 split (not index sources).

## 3. What crosses a stage boundary

The engine of the eight-card builds had one rule: only the residual may cross a boundary, and the only cut it
accepted was layer 20 (READ `contract.py`, PLAN section 2). Three things break that rule at cuts 14 and 28, and
the kit handles each.

| State | Problem at three stages | What the kit does |
|---|---|---|
| The compressed attention rows that layer 20 writes | layers 20 to 39 all read them; with a cut at 28 the writer (stage 1) and some readers (stage 2) are on different stages | Layer 20 hands its rows to the hand-off (one row per token: [T, 512] and [T, 128] in FP16). The last stage writes them into its own copy, a mirror of KV source 20, before any of its layers runs. This is #1031's design |
| Layer 28's top-k selection | none: layer 28 is an index source and recomputes its own selection | nothing crosses |
| Engram at layer 14 | Engram's reader and hash state live on the first stage; layer 14 is the first layer of the middle stage | Since kits 5c and 6c the first stage applies layer 14's Engram as its last operation before the hand-off, with layer 14's Engram module loaded there at its checkpoint place. No Engram rows cross the boundary |
| Pair-compressor state (written at layers 2, 8, 14) | it is committed once per step; the engine committed only on the first stage | A stage that holds such state and does not sample runs the same commit protocol the first stage runs (`pp_commit.py`) |

Each stage has a typed plan that says what it owns and what crosses its two boundaries (READ
`kit/PP3KIT6c-over-cut/files/models/deepseek_v41/contract.py`, class `StagePlan`): its first and last layer, its
own KV sources, the mirrored source it receives, the source it exports, the Engram it applies for the next stage,
and whether it holds committed state. The qualified layouts are listed: one stage, two stages cut at 20, three
stages cut at 14 and 28. Everything else is refused with its reason. The two-stage plan is derived by the same
function and tested equal to the earlier behaviour.

Sizes of a hand-off:

- Decode: under 0.5 MB per rank (ESTIMATED, PLAN section 5).
- Prompt read: the residual of one chunk, 640 MiB per rank at 8,188 rows (ARITHMETIC, PLAN Amendment 11), half of
  that at 4,092 rows (ARITHMETIC).

## 4. The scheduler's spacing rule

The stages before the last learn a step's outcome (the sampled token, the accepted count, the next drafts) through
a ring, exactly one step per pipeline stage later. A request's next decode step must therefore not start before
those steps have passed.

- At two stages the engine's sync loop happens to give a lone request the right spacing.
- At three stages it does not: the first decode step after a prompt ran one call early, and the first stage
  embedded an earlier request's tokens. Window w1 showed it as 15 of 15 wrong answers with a divergence at token 1.
- Kit 3 fixed it: `Scheduler._update_after_schedule` sets `next_decode_eligible_step = current_step + pp_size`, as
  the async scheduler already did. The test runs the real engine loop with the real scheduler (17 of 28 cases
  fail before the fix, all pass after; the two-stage sequence is asserted unchanged).
- The same kit raised the slot scheduler's stall guard from two empty held steps to `max(2, pipeline_depth)`. At
  three stages a decoder alone leaves two empty steps between its own, which is normal work and not a stall.

This rule is the reason the build answers correctly. It is in every kit from kit 3 on. Window w3 confirmed it on
the cards.

## 5. Slots

A slot is the kept state of one conversation on the cards: its rings, its KV store and one set of CUDA graphs.

- **One request per step.** A step of the pipeline carries exactly one request (READ `v41_multi_seq.py`, quoted
  in the kernel and optimization plan, section 1). A slot does not batch requests into a step. A three-stage
  pipeline has three steps in flight, so three requests fill it.
- A fourth request at once adds little total speed. On the quantized twelve-card mix four together gave the same
  sum as three (MEASURED, window s1). On the all-NVFP4 build four together gave 306.6 tok/s on fresh text against
  288.0 for three (MEASURED, windows s4 and o2): 6 % more.
- A slot also keeps a chat warm. With two slots, a third chat takes a slot and the chat that lost it is read
  again (46.7 s to the first token in window o2). With three slots it took 2.1 to 2.6 s. That is what the fourth
  slot is for: a fourth chat kept warm.
- The engine accepts up to eight slots since kits 4 and 4b (three numeric guards and one label table were sized
  for four).
- Memory per slot: about 510 MiB on the fullest card on the quantized mix (MEASURED, window s1); 328 MiB on the
  middle stage at equal chunk size on the all-NVFP4 build (MEASURED, window o2, quoted in PLAN Amendment 18).
- The cost of the fourth slot: about 345 MiB on the fullest card (30,398 MiB with four grown chats in window s4
  against 30,052 to 30,054 MiB with three in windows o2 and k1; ARITHMETIC), and more waiting when every slot is
  busy: four documents read at once reach their first token after 129 to 166 s each, three after 105 to 121 s
  (MEASURED, windows s4 and k1).
- **Standing: four slots** (accepted in window s4; PLAN Amendments 18 and 19). Three slots, the configuration of
  windows o2 and k1, stay available.

## 6. Prompt chunks

A prompt is read in chunks. Everything a prompt read allocates is sized by the chunk's rows.

- The profile's `max-num-batched-tokens` is 4096, which gives chunks of 4,092 rows. The two-slot configuration
  used 8192 (chunks of 8,188 rows).
- Halving the chunk is what makes room for the third slot. At two slots and 8,188-row chunks the middle stage
  stood at 28,661 MiB after warm-up and peaked at 31,336 MiB (MEASURED, window w5). At three slots and 4,092-row
  chunks it stands at 28,667 MiB and peaks at 30,052 MiB (MEASURED, window o2).
- The cost is prompt-read speed: 3,286 tok/s against 3,410 on one read each (MEASURED, window o2).
- The allocator's cache is given back at the entry of a prompt read of 2,048 rows or more
  (`VLLM_V41_EMPTY_CACHE_TOKENS`, the engine's default). A lower setting was measured and rejected (window w5).
- A prompt cut at a different place gives the same model in another rounding. Nothing lossy crosses a chunk
  boundary. Two sources of the difference are known:
  1. **The four-card sum.** A sum under 8,388,608 bytes takes the engine's own all-reduce path, which adds the four
     ranks in FP32 and rounds once; a larger sum goes through NCCL in FP16, in an order that depends on the
     buffer's size and the element's place in it (MEASURED at the all-reduce entry, step 0 probe). Long prompts
     differ as much between 8,188-row and 8,124-row chunks as between 8,188-row and 4,092-row chunks (MEASURED,
     window o2). Short prompts are identical across chunk sizes.
  2. **The dense library's split count** (found 2026-11-17, PLAN Amendment 23). Above 32 rows the general dense
     multiply of the engine's TurboMind library chooses how many splits of the inner dimension to use from the
     chunk's row count, and rows computed under different split counts round differently. Traced on one card for
     layer 0's `wkv` projection: 10 splits at 2,244 to 2,292 rows, 1 at 2,308, 4 at 2,196 to 2,228 and at 3,148
     and 3,164, 2 at 3,696, 3,712, 6,376 and 6,392, 5 at 4,092 and 8,188 (MEASURED, `tools/gemm_split_probe.py`;
     every count equal to the CPU model `tools/tm_dispatch_emulation.py` written before the run). Rows under one
     split count are bit-identical whatever the row count; with the library's switch `TM_GEMM_TUNE=max_splits=1`
     all 120 compared pairs are bit-identical. So a prompt's LAST chunk is read in the arithmetic its row count
     selects. Full chunks of 4,092 and 8,188 rows fall on one count, so this is not the cause of the long-prompt
     difference between the two chunk sizes (source 1 is). The eight-card build has the same code.
  No token is shown wrong by either source. Two reads of one text are comparable bit for bit only when their
  chunks fall alike, which is why the store's tests on this build use lengths whose tails fall on one split count
  (M12 at 22,736 and M1 at 22,704 tokens, PLAN Amendment 23).
- **Pages are shared by token hash.** The block pool keeps duplicates of a token hash, and a snapshot hit takes the
  oldest surviving duplicate, not the pages its own read wrote (READ `block_pool.py:62-91`,
  `kv_cache_manager.py:257-282`; seen in window st3b's dump, PLAN Amendments 22 and 23). A request can therefore
  start from pages another request's read wrote in the other rounding. It is the base engine's rule on both
  builds; a snapshot hit tied to its own read's pages is item B5 of the kernel plan.
- **The prompt skip runs on the last stage only.** A chunk that ends 3,072 rows or more before the prompt's end is
  skipped by the last stage (`model.py:1039-1043`; window s4's log shows skip lines from the last stage's workers
  only). The first and middle stages compute their fourteen layers on such a chunk; on the middle stage layer 20's
  FFN and layers 21 to 27 are computed for nothing, since the stages after it need only the rows up to layer 20's
  attention, which feed the KV stores and the mirror of source 20 (READ and MEASURED, 2026-11-17;
  it corrects an earlier row of the kernel inventory). A switch for the middle stage is built on a CPU
  in kit 8 and has not met the cards (section 12).
- **A turn that leaves its chat's prompt at the last token is read cold** after an answer longer than the ring's
  reach. The slot cannot continue it, the snapshot one grain below is not served by the block pool, and a store
  entry serves only a prompt that begins with all of it. MEASURED: 43.84 s in window st4 and 36.0 to 37.9 s in
  window t1, against about a second for a turn that continues the whole prompt. A client that replays only the
  answer's text (as the two probes did) meets it; a client that replays the reasoning with each answer does not. PLAN Amendments 25 and 31; kernel plan tier D.

## 7. Memory per stage

Per card, 4,092-row chunks, kit 6c. The standing configuration has four slots (window s4). The three-slot rows
are the configuration of windows o2 and k1.

| | Stage 0 (layers 0-13) | Stage 1 (layers 14-27) | Stage 2 (layers 28-39) | Label |
|---|---|---|---|---|
| Routed experts, all NVFP4 | 5,376 | 5,376 | 4,608 | MEASURED, every boot |
| Their records | 24.40 GiB | 24.40 GiB | 20.92 GiB | ARITHMETIC, PLAN section 4 |
| Profile reserve (three and four slots) | 3,686 MiB | 3,686 MiB | 7,361 MiB | READ, PLAN Amendments 13 and 18 |
| **Four slots:** after warm-up | 28,795 MiB | 29,013 MiB | 27,495 MiB | MEASURED, s4 |
| Four slots: headroom under the allocator limit at the end of warm-up | 3,203 MiB | 2,985 MiB | 4,503 MiB | MEASURED, s4 |
| Four slots: highest sample, four grown chats | 29,838 MiB | 30,398 MiB | 28,892 MiB | MEASURED, s4 |
| Three slots: after warm-up | 28,535 MiB | 28,667 MiB | 27,253 MiB | MEASURED, o2 |
| Three slots: outside the allocator | 867 MiB | 881 MiB | 865 MiB | MEASURED, o2 |
| Three slots: highest sample, three grown chats | 29,576 MiB | 30,054 MiB | 28,654 MiB | MEASURED, k1 |
| Three slots: allocator limit under the card limit | 31,131 MiB | 31,117 MiB | 31,133 MiB | MEASURED, k1 |

The fourth slot added 260 / 346 / 242 MiB to the level after warm-up (ARITHMETIC, s4 against o2). The reserves
were not changed for it: a slot's own state lowers the planner's budget by itself, and the four-slot boot still
logged 0 cold experts (PLAN Amendment 18 and window s4).

- **The store costs no card memory.** With the store on as it would serve (no dump) and four grown chats, the
  stages peaked at 29,802 / 30,396 / 28,894 MiB against 29,836 / 30,398 / 28,896 MiB with it off (MEASURED, window
  st4, boots KN and KA).
- **The free-memory line.** The owner allowed 20 % less free card memory on 2026-11-17. From the next arm judged under
  it the line is 32,100 MiB (410 MiB free) and the serving ceiling 31,800 MiB. The standing build boots under 31,998
  MiB until such an arm passes (PLAN Amendment 26). Nothing on this build is bound by the line today: every expert
  is on the cards, so the room goes to slots or larger prompt chunks, decided on the target workload's shapes.
- **The middle stage is the tight one.** It holds as many experts as the first stage and gains KV source 14. The
  plan's arithmetic named it before the first boot; window w3 measured it 190 MiB under the line at two slots.
- **What kits 5c and 6c took off it:** the token embedding (113,766,400 bytes, 108.5 MiB per card; it is built
  only where tokens enter and where the drafter runs), the 96 MiB receive buffer for Engram rows, the copy of
  each received chunk, and the Engram projection (25.9 MiB as loaded, on the first stage now).
- **The reserve.** The planner puts in host memory whatever does not fit a card after the reserve. The reserve on
  the first and middle stage was lowered to 3,686 MiB so that the planner keeps every expert on the cards at
  three slots, and the same reserve holds at four. The boot must still log 0 cold experts: a reserve too small for
  a stage is refused there.
- **What moves at run time.** The long-lived memory is fixed at boot: the stores for the maximum context, every
  slot's graphs, the pools. The eager tensors of a prompt read and the allocator's cache move (READ, PLAN
  Amendment 6).

## 8. The card limit

`models/deepseek_v41/card_limit.py`, added in kit 6 (READ, PLAN Amendment 6).

- At the end of warm-up every worker logs its card's used memory, the allocator's reserved memory and what lives
  outside the allocator.
- With `VLLM_V41_CARD_LIMIT_MIB=N` each worker sets its allocator limit to N minus its own outside figure. The
  card's used memory may then reach N on every stage.
- At the limit the allocator gives back its cached blocks and retries. An allocation that still does not fit
  raises an out-of-memory error in that worker. The step fails, the engine stops with the error in its log,
  `/health` fails and every request in flight gets an error. That is loud and it is the whole server: a request
  cannot be failed alone in the middle of a pipeline step.
- A worker already over its limit at the end of warm-up does not start.
- The standing value is 31,998 MiB on every stage: the owner's line. It is a fuse, not a budget. One value for
  all stages, because a lower figure on a roomier stage would only make a failure more likely (PLAN Amendment 13).
- The rule for switching it on: on every stage the allocator limit must leave at least 512 MiB over the largest
  reserved figure the configuration reaches. At three slots the headroom is 2,420 / 1,946 / 3,352 MiB (ARITHMETIC
  from window o2's figures). Window k1 ran it on the cards: no token and no memory mark changed. At four slots the
  highest card sample is 30,398 MiB, 1,600 MiB under the limit (MEASURED, window s4).

## 9. Boards and device order

READ the launcher and the plan, section 5.

- Boards: X and Y, both behind one PEX88096 switch; P behind a PLX 8749 (four cards each).
- Standing order `xyp`: the first hand-off (stage 0 to stage 1) stays inside the PEX switch, as in the eight-card
  default. The second (stage 1 to stage 2) crosses to the other host bridge. The alternative `pxy` exists in the
  launcher; the plan records no window that used it.
- Hand-off pairs are (rank i of a stage, rank i of the next). Before any three-stage boot the driver runs an NCCL
  probe on the eight pairs; a pair that is not PASS ends the window.
- The first stage's four cards must be the first four visible devices: the Engram reader requires device = rank
  (READ `deepseek_v41_engram_io.py:229-231`).
- The launcher derives the device list from the PCI layout it reads at launch and refuses any other layout. It
  also refuses to boot while a bridge above a card has ACS redirect set, because that hangs card-to-card transfers.

## 10. What is the build's own, and what is shared

- **Own tree.** The build has its own engine copy and venv on the host, with the kit applied there only, and its
  own run scripts, profiles, state and logs. The eight-card build's engine is never patched and its trees are
  never written. The driver checks that before every boot of the eight-card build.
- **Shared, read-only by path:** the model files, the NVFP4 packs and the routing tally of the eight-card build.
  The RAM lock pins those files, so both builds load the same pages.
- **The default since 2026-11-17.** The owner made it the default server at about 18:14Z. The build serves on the
  default's port 18042 since 19:20:34Z (PLAN Amendments 27 and 29). The eight-card build is the fallback; it is
  down while this build serves (the cards are the same twelve; it uses eight of them) and comes back only on the
  owner's command (`run_window_pp3.sh TAG final`). Port 18043 is the test port.
- **The drafter** stays on the last stage, tensor parallel 4. Its layers 37 to 39 are inside layers 28 to 39.

## 11. The kit

Kit 6c: eleven files over the eight-card build's engine (commit `f20d155`); engine commit `e7b2417`
(`kit/PP3KIT6c-over-cut/MANIFEST`, with the base and new sha256 of every file).

| File (under `vllm/`) | What the kit changes |
|---|---|
| `models/deepseek_v41/contract.py` | the typed stage plan per stage; the list of qualified layouts; which tensors a stage loads (`stage_keeps_weight`) |
| `models/deepseek_v41/model.py` | the stage plan drives stage bounds, stores, the hand-off (layer 20's rows, the mirror), the Engram applied on the first stage for layer 14, the commit protocol on stages that do not sample; three stages accepted |
| `models/deepseek_v41/pp_commit.py` | wording only |
| `models/deepseek_v41/dense_tm.py` | dense preparation of the moved Engram projection |
| `models/deepseek_v41/slots.py` | per-slot state for up to eight slots |
| `models/deepseek_v41/card_limit.py` | new: the card memory lines and the limit |
| `model_executor/model_loader/gguf_loader.py` | the loader's stage filter asks the model which tensors a stage keeps |
| `v1/core/sched/scheduler.py` | the spacing rule (section 4) |
| `v1/core/sched/v41_multi_seq.py` | slots at three stages; the stall guard; slot labels for eight slots |
| `v1/worker/gpu/model_runner.py` | commit hooks on every stage that is not the last; the card-limit hook |
| `v1/worker/gpu/v41_multi_seq_state.py` | slot owners for up to eight slots |

PLAN section 9 holds a table of which real entry point runs each of these files and whether a CPU test goes
through it. Two earlier kits passed CPU tests that did not go through the server's path, so that table is part of
the design.

Unchanged on purpose (PLAN section 6): the KV store, the loader's payload reading, the scheduler's step logic, the
drafter, and the two-stage path.

Two more kits exist beside kit 6c and are not part of the standing configuration:

- **The conversation store's kit** (owned by a private working repository; 26 files, five shared with kit 6c, pinned by its
  manifest `75646659...`, engine `e6df5ed`) is in the trees since window st3 with the store OFF. With the store off
  it changes no token and no memory mark (windows st1, st3b, st4). It records the two-tier tier digest on this
  build's loader path (twelve digest lines per boot, two distinct values, as predicted). Switching the store on is
  environment only and part of the store's fingerprint, together with the kit, the profile and every `VLLM_*`
  setting.
- **Kit 8** (engine branch `pp3-12card-pipe`, CPU only): see section 12.

## 12. Known limits of the design

- One request alone gains nothing from the third board. Its round carries the same forty layers of work
  (MEASURED, windows w3 and w5).
- A step carries one request. Requests beyond the three stages wait. Batched decode, as upstream 1Cat-vLLM #1482
  has it for four stages, is a different scheduling design and a study in the kernel and optimization plan (item
  B1); it is not in this engine.
- The conversation store and the pinned-prefix cache are not in the kit. The kit keeps one packed store per stage and a
  second copy of layer 20's rows on the last stage, so a restore must fill both copies (PLAN section 13). The
  store's kit patches some of the same files and goes over kit 6c in a fixed order.
- On skipped prompt chunks the middle stage computes layers 21 to 27 for nothing (section 6). It is correct, and
  prompt reading gains less than the layer count suggests (PLAN section 13). Whether it costs time depends on which
  stage bounds a read: in window t1 the middle stage's 4,092-row chunk took 891 ms against the first stage's 793 ms
  (MEASURED, 200 W, one run).
- A turn's new rows run as one chunk through the three stages in series (section 1).
- **In progress, not on the cards: kit 8** (engine branch `pp3-12card-pipe`, built on a CPU). Three switches, off by default: the turn pipeline
  (`VLLM_V41_PP_TURN_CHUNKS`: a turn's last rows read in chunks of a multiple of 64 so the three stages overlap; it
  changes the cut, so its answers are another rounding, judged by closeness); the middle-stage skip
  (`VLLM_V41_PP_MID_PREFILL_SKIP`: the middle stage stops after layer 20 on chunks the last stage skips, exact by the
  skip's contract); and a hold of the decode steps that follow a prompt's last step, so a small turn beside a long
  read is not queued behind the read's chunks. Its arm's rule is written into the plan before its data.
- The dense library's split switch (`TM_GEMM_TUNE=max_splits=1`) stays out of the standing configuration: its arm
  ended INCONCLUSIVE at its quality line in window t1 (PLAN Amendment 31).
- The large four-card sums of a prompt read are rounded in FP16 through NCCL, and a prompt's last chunk is read in
  the split count its row count selects (section 6). A cached or restored turn is therefore not token-identical to
  a cold read of the whole chat (window st4's restore pair: equal in 26 of 137 positions), while a restore compared
  with turns started at the same position is identical in state and answer (the store's S2, window st4).

## 13. Serving as the default

Since the owner's word of 2026-11-17 about 18:14Z the build is the default, and its port 18042 must never hold a server
that is still being checked, nor can a client's request be allowed to fail a check (PLAN Amendment 27). The serve
phase was therefore split in two driver phases (READ `window/run_window_pp3.sh`, header and phases `servecheck`,
`servedefault`, `servestate`):

| Phase | Port | What it does |
|---|---|---|
| `servecheck` | 18043 | every check of the old serve phase as arm C: 0 cold on every rank, the card limit on 12 of 12 workers, the seven cold long prompts identical to the accepted capture, the rehearsal with one grown chat per slot under the line. On PASS it stops its arm and writes the record `run/state/SERVE-CHECKED` (`tools/serve_checked.py`). It never declares the build serving |
| `servedefault` | 18042 (`SERVE_PORT`) | the serving boot as arm S from the port copy of the profile (`tools/make_p18042_profile.sh`: one line differs, mode 444, never rewritten). It sends no request. It needs a record that matches the trees as they are; after the boot it checks /health, tracebacks, twelve sizing lines with the record's hot counts and 0 cold, the card limit on 12 of 12 and no draft-fallback warning. A failed check stops the arm and nothing serves |
| `servestate` | none | read-only: the record against the trees (exit 0 match, 36 servecheck owed) |
| `serve` | 18043 only | the old phase; it refuses `SERVE_PORT=18042` and names the two phases above |

The record holds the configuration, the card limit, the profile's sha256 without its port line, the kit, the
store's kit state, the sha256 of the effective kit rows, the hot counts, the capture's summary and the rehearsal's
highest sample. **The full check is owed whenever the record does not match the trees** (any change of kit, store
kit state, profile, configuration or card limit). While it matches, a return from a test arm is `servedefault`
alone: no second capture and rehearsal for unchanged bytes.

The chains that end serving all end with this pair rule (`window/serving_lib.sh`): `chain_start.sh` (from nothing
serving), `chain_move.sh` (from a serving arm on either port), `chain_split.sh` (arm G between two serving
periods), `chain_budget.sh` (arm T, and arm G if asked, between two serving periods) and `chain_serve.sh` (from the
eight-card fallback). None boots the fallback by itself: after a failure nothing serves and the last line names the
two commands the owner may run. MEASURED: the first move to 18042 took 3 minutes without a server (window mv1);
ESTIMATED about 14.5 minutes when the record is stale. Commands and timings: [OPERATIONS.md](OPERATIONS.md).

Two facts met on the cards on the way:

- **A titled worker's `/proc` environment is blank.** A worker sets its process title with setproctitle, which
  reuses the memory of the environment block, so `/proc/<pid>/environ` of every titled worker is empty while the
  process's own `getenv` still returns its variables (MEASURED on a CPU with the build's own library; window g1).
  A witness that a switch reached the workers reads the library's own log lines instead (`tools/split_witness.py`;
  PLAN Amendment 29).
- **Power.** The V100s run at 200 W since 2026-11-17 18:58Z with the clock cap at 1,477 MHz (PLAN Amendment 28).
  A prompt read rose 13.7 % (3,733 against 3,284 tok/s); a decode round, which keeps each card busy about a third of
  the time, stayed within 2 % (one run, Amendment 29). After an unplanned power trip at 200 W the owner sets the
  caps back to 150 W and starts the build from nothing (the owner's standing rule).
