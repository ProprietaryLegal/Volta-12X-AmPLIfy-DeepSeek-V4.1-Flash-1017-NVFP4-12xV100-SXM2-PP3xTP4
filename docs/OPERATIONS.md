# Operations

How the twelve-card build is installed, started, checked and stopped, for a reader with twelve V100-SXM2 32 GB cards
of their own. Our window driver and launcher are not in this release ([NOT-YET-PUBLISHED.md](../NOT-YET-PUBLISHED.md));
this page describes what they do, so that the same steps can be followed with your own scripts. Paths are
placeholders: `<build>` is the build's own directory, `<models>` the model files.

## What you need

- Twelve V100-SXM2 32 GB cards in three NVLink boards of four. Tensor parallelism must stay inside a board.
- The engine of PLI Labs' eight-card build (1Cat-vLLM #1277 with the patch series of Volta sPLIt and the native
  expert path of Volta Max uPLIft) in its own source tree and venv under `<build>/engine/`, never shared with
  another server: the serving kit (kit 11 since 2026-11-24) is applied to both. See [engine/README.md](../engine/README.md)
and [RUNNING.md](../RUNNING.md), which gives the twenty adopted lines as served.
- The model files: DeepSeek-V4.1-Flash-1017 (config, tokenizer, the DSpark drafter), NVIDIA's
  DeepSeek-V4.1-Flash-1017-NVFP4 experts as the native packs of Volta Max uPLIft, and the ISTA-DASLab GSQ-RCO GGUF
  for the non-expert tensors. Nothing is downloaded by this repository.
- PCIe ACS redirect cleared on the switches above the cards: with it set, card-to-card transfers hang.

## Install the kit

    bash kit/PP3KIT9S-over-cut/apply.sh <build>/engine/src/vllm <build>/engine/venv/lib/python3.12/site-packages/vllm <backup>

Only while nothing serves from those trees. It refuses unless every file it replaces has its base sha256. Kits 9X to
11 are published without their `apply.sh`, the conversation store's modules and their compiled libraries
([NOT-YET-PUBLISHED.md](../NOT-YET-PUBLISHED.md)); their `MANIFEST`s list every row.

## The profile

The serving profile of Volta Max uPLIft's two-tier native path, changed to three stages:

- pipeline parallel size 3, tensor parallel size 4, `VLLM_PP_LAYER_PARTITION=14,14,12`;
- `max-num-batched-tokens 4096` (prompt chunks of 4,092 rows) and `max-num-seqs 4` (four slots);
- context 524,288; residency `two_tier` with every routed expert hot;
- per-stage reserves 3,686 / 3,686 / 7,361 MiB (stages 0, 1, 2): small enough that the tier planner keeps every
  expert on the cards, which the boot must confirm;
- in the environment: `VLLM_V41_CARD_LIMIT_MIB=31998` (the card limit; leave it out to boot without it) and the
  adopted lines ([RUNNING.md](../RUNNING.md), section 3): the eleven speed lines, the lean hand-off, the packed
  experts, the loop counters, the conversation store with its four budgets and its folder;
- `enable-prompt-tokens-details: true`, so every reply's usage carries the cached-token count.

## Device order

The first stage's four cards must be the first four visible devices (the Engram reader requires device = rank).
Order the twelve cards stage by stage: the four cards of one board for stage 0, then stage 1, then stage 2. Put the
first hand-off inside one PCIe switch where you can. Before the first boot, run an NCCL send/receive between each
card and the card of the same rank on the next stage (eight pairs); do not boot three stages over a pair that fails.

## Start and check

Our rule: a server on the public port is never one that is still being checked. So the start is in two parts.

1. **Checks, on a test port.** Boot the profile on a port no client uses, and require all of:
   - every rank logs 0 cold experts and the hot counts 5,376 / 5,376 / 4,608;
   - all twelve workers log their allocator limit under the card limit, and no worker refuses at the end of warm-up;
   - a fixed set of long prompts (public text, temperature 0) answered token for token as an accepted capture you
     took once on these bytes, compared by a strict judge (equal length, equal token ids);
   - one grown chat per slot (four chats of 110,000 to 140,000 tokens with turns) at once, with the highest card
     sample at or under your line and no out-of-memory line in the log (`tools/oom_lines.py` counts real ones, never
     the card limit's own announcement).
   On a pass, stop the server and write a record of what was checked: profile, kit, configuration, card limit, hot
   counts, the capture's summary, the highest sample (`tools/serve_checked.py write`).
2. **The serving boot, on the public port.** Boot the same bytes from a copy of the profile that differs only in its
   port line. Send it no request. Check the boot lines only (health, no traceback, the sizing lines with the record's
   hot counts, the card limit on 12 of 12). Allow it only while the record matches the installed bytes
   (`tools/serve_checked.py match`); after any change of kit, profile, configuration or card limit the full check
   is owed again.

Load takes about 134 s from launch to ready on our host.

## Requalify a new kit

A kit change on a set that serves is one window: install the kit in both trees with nothing serving, boot it on the
test port with the adopted lines unchanged, and require identity 7 of 7 to the accepted capture, the same store record
set opened, none of the kit's new switches in the API server's environment, the kit's dispatch witness on 12 of 12
workers, the store, API, pin and state checks PASS, the adopted lines byte-identical, Xid 0 and the four-chat high under
your line. Record the reads and decode cells; do not judge them. On a pass write the record and make the serving boot;
on a failure put the earlier kit back and boot it on its matching record. Kits 9Z, 10 and 11 moved this way (windows
z9a, aw6 and aw7); kits 10 and 11 took 32 min 52 s and 33 min 45 s with nothing serving.

## Stop

Wait for a moment with no request, then send the server's process group SIGTERM and wait until all twelve cards are
at 0 MiB. A move from one serving boot to another, with a matching record, took about 3 minutes without a server.

## Test arms

A test arm boots on the test port, never on the public port, with its acceptance lines written down before its
data. Our arms of 2026-11-17 and 2026-11-18 (the slots, the card limit, the store, the split switch, the time budget,
kit 8) and their results are in [measurements.md](measurements.md) and [history.md](history.md). Since window aw1 one
chain runs a list of arms, one change per boot, with the reference first and its repeat last; a change is adopted when
each speed cell it targets beats the reference by the band (the larger of 1.8 % and the window's own spread between the
reference and its repeat; 3 % before window aw7) and the seven long prompts stay within the keep line (median first
divergence at least 104 tokens, median mean |dlogprob| at most 0.018).

## Faults

An Xid line, a hang or a dead engine at any step: stop everything and start nothing until the cause is understood.
A power trip at 200 W: our standing rule sets the cards back to 150 W before the build is started again.
