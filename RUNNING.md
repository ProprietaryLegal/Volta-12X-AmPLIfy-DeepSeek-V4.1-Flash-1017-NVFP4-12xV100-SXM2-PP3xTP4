# Running it

What follows is how we run the served build, the fifth release's serving set: kit `PP3KIT11-over-cut` with the twenty
adopted lines, four slots, 175 W per card and a 1,452 MHz clock ceiling. It comes from
[docs/OPERATIONS.md](docs/OPERATIONS.md), [engine/README.md](engine/README.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
(sections 9 and 13), the `Makefile` and the tools in `tools/`. Our launcher, window driver, chains and bench tools are not
in this release, because they name our host's layout ([NOT-YET-PUBLISHED.md](NOT-YET-PUBLISHED.md)); where a step was
theirs, this page says what they do, so the step can be followed with your own scripts. Paths are placeholders:
`<build>` is the build's own directory, `<models>` the model files.

## 1. The engine and the kit

The engine is the one PLI Labs' eight-card builds serve, at commit `f20d155`: 1Cat-vLLM
[#1277](https://github.com/1CatAI/1Cat-vLLM/pull/1277) with Volta sPLIt's patch series and Volta Max uPLIft's native
expert path. Give the twelve-card build its own source tree and venv (Python 3.12) under `<build>/engine/`, never shared
with another server. Then apply a kit to both trees, while nothing serves from them. Kit 11, which serves, cannot be
applied whole from this repository: the conversation store's modules, the compiled libraries and the `apply.sh` of
kits 9X to 11 are withheld ([NOT-YET-PUBLISHED.md](NOT-YET-PUBLISHED.md)), and its published files call the store's
modules on the store's code paths. Kit 9S, the fourth release's serving kit, applies whole:

```sh
bash kit/PP3KIT9S-over-cut/apply.sh <build>/engine/src/vllm \
    <build>/engine/venv/lib/python3.12/site-packages/vllm <backup-dir>
```

`apply.sh` refuses unless every file it replaces has its base sha256 in both trees and every new file is absent, and
the backup directory does not exist yet. It saves the originals into the backup directory, copies the kit's 18 files into
both trees through a temporary name and checks every result; a failure while copying restores both trees from the
backup. No pip, no build, no restart. `rollback.sh` restores the backup byte for byte. Each kit's `MANIFEST` holds one
tab-separated row per file (the path under `vllm/`, the base sha256 or `NEW`, the new sha256) and header lines naming
the engine commit and the base. In this published copy the comment lines that named private hosts, repositories,
folders or reviewers were reworded and the new sha256 recomputed (a header line says so); the code is unchanged.

Each kit's `MANIFEST` still lists every row, the withheld ones with their sha256, so a tree assembled with the same
bytes can be checked against it. The kit's CPU tests run inside an engine tree with the kit applied:

```sh
pytest --noconftest tests/models
```

A kit's test must go through the server's real entry point: two kits of 2026-11-17 passed CPU tests that did not, and
failed on the cards (windows s1 and w4). Kits 9Q, 9T, 9R, 9U and 9V are kit 9S (or 9N) plus the files of the three
failed windows, each with a default-off switch; apply one only to test it. Kits 9W to 11 are listed with their states
in [engine/README.md](engine/README.md).

## 2. The cards and the model files

- Twelve V100-SXM2 32 GB cards in three NVLink boards of four. Tensor parallelism must stay inside a board.
- PCIe ACS redirect cleared on the switches above the cards: with it set, card-to-card transfers hang.
- Power and clock as served: 175 W per card and a 1,452 MHz clock ceiling. After an unplanned power trip our standing
  rule sets the cards back to 150 W before the build is started again.
- Device order: the first stage's four cards must be the first four visible devices (the Engram reader requires
  device = rank). Order the twelve cards stage by stage: the four cards of one board for stage 0, then stage 1, then
  stage 2. Put the first hand-off inside one PCIe switch where you can.
- The hand-off probe: before the first boot, run an NCCL send and receive between each card and the card of the same
  rank on the next stage, eight pairs. Do not boot three stages over a pair that fails. Our probe is not published.
- The model files under `<models>`: DeepSeek-V4.1-Flash-1017's config, tokenizer and DSpark drafter; NVIDIA's
  DeepSeek-V4.1-Flash-1017-NVFP4 experts as the native packs of Volta Max uPLIft; the ISTA-DASLab GSQ-RCO GGUF for the
  non-expert tensors. Nothing is downloaded by this repository.

## 3. The profile and the switches

The serving profile is the two-tier native profile of Volta Max uPLIft changed to three stages:

- pipeline parallel size 3, tensor parallel size 4, `VLLM_PP_LAYER_PARTITION=14,14,12`;
- `max-num-batched-tokens 4096` (prompt chunks of 4,092 rows) and `max-num-seqs 4` (four slots);
- context 524,288;
- residency `two_tier` with every routed expert hot;
- per-stage reserves 3,686 / 3,686 / 7,361 MiB (stages 0, 1, 2): small enough that the tier planner keeps every expert
  on the cards, which the boot must confirm.

The environment as served, the card limit and the twenty adopted lines:

```sh
VLLM_PP_LAYER_PARTITION=14,14,12
VLLM_V41_CARD_LIMIT_MIB=31998              # the card limit; leave it out to boot without it
VLLM_V41_EMPTY_CACHE_TOKENS=8192           # release cached card memory only at prompt steps above 8,192 rows
VLLM_V41_PP_MID_PREFILL_SKIP=1             # the middle stage skips its late layers on chunks the rule allows
VLLM_V41_PP_EMPTY_AHEAD=1                  # spacing steps submitted while a step is in flight
VLLM_V41_PP_ANSWER_NO_SLEEP=1              # no sleep between an answer and the next submission
VLLM_V41_DRAFT_GEMV2=1                     # the drafter's faster GEMV
VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD=128     # a short turn is not queued behind another chat's long read
VLLM_V41_PREFIX_TOKEN_CACHE=1              # tokenize only what a chat added
VLLM_V41_PP_TURN_CHUNKS=3                  # a turn's last rows as a head and three 1,024-row chunks
VLLM_V41_INDEX_TC=1                        # the sparse-attention index scorer on the tensor cores
VLLM_V41_WO_A_BATCHED_4K=1                 # the attention output projection of a 4,092-row chunk in one launch
TM_GEMM_TUNE=max_splits=1                  # the multiply library's split count fixed at one
VLLM_V41_PP_LEAN_HANDOFF=1                 # the lean pipeline hand-off (kit 9T's)
VLLM_V41_EXPERT_PACKED576=1                # expert records without zero padding (needs re-cut packs)
VLLM_V41_PP_LOOP_STATS=64                  # the engine core's loop counters every 64 answers (diagnostic)
VLLM_V41_STORE=1                           # the conversation store and its pinned prefixes
VLLM_V41_STORE_PINNED_RAM_BYTES=25769803776          # 24 GiB of host memory for pinned prefixes
VLLM_V41_STORE_CONVERSATION_RAM_BYTES=68719476736    # 64 GiB of host memory for conversations
VLLM_V41_STORE_PINNED_DISK_BYTES=214748364800        # 200 GiB of disk for pinned prefixes
VLLM_V41_STORE_CONVERSATION_DISK_BYTES=429496729600  # 400 GiB of disk for conversations
VLLM_V41_STORE_DIR=<store-dir>             # the store's folder; its records belong to one set of numerical switches
```

The profile also carries `enable-prompt-tokens-details: true`, so every reply's usage carries the cached-token count.
Leave unset to serve as we do: `VLLM_V41_TP_EXACT_SUM` (kit 9V's failed window), `VLLM_V41_PARTIAL_PREFIX` (kit 9Y, not
run) and `VLLM_V41_ATTN_TIMING` (a diagnostic). The receive-buffer wait, the mHC prompt kernel (kit 10) and the early
control header (kit 11) are on by their kits' defaults and need no line; `VLLM_V41_MHC_PREFILL=0` and
`VLLM_V41_PP_CONTROL_AHEAD=0` give the kit before, and with `VLLM_V41_PP_LEAN_HANDOFF=0` the control header must be off
too, or the worker refuses to boot. `VLLM_V41_PP_TURN_CHUNKS`, the batched projection and the split count each move
rounding, so a boot with a different set of lines must be held to the capture taken on that set (section 4). At a full
conversation disk budget the store's save walks every entry; keep the conversation budget at what your disk and your
traffic allow.

In our tree the lines live in `run/state/ADOPTED.env`, written from an allow-list by kit, and the serving record carries
their digest.

## 4. Start and check

Our rule: a server on the public port is never one that is still being checked. So the start is in two parts.

1. **Checks, on a test port.** Boot the profile on a port no client uses, and require all of:
   - every rank logs 0 cold experts and the hot counts 5,376 / 5,376 / 4,608;
   - all twelve workers log their allocator limit under the card limit, and no worker refuses at the end of warm-up;
   - a fixed set of seven long prompts (public text, temperature 0) answered token for token as an accepted capture you
     took once on these bytes, compared by the strict judge (`tools/closeness.py`: equal length, equal token ids, the
     same key set; anything else INCONCLUSIVE, exit 3);
   - one grown chat per slot (four chats of 110,000 to 140,000 tokens with turns) at once, with the highest card sample
     at or under your line and no out-of-memory line in the log.

   Count the out-of-memory lines with the tool that never counts the card limit's own announcement:

   ```sh
   python3 tools/oom_lines.py <server-log>     # prints the number; an unreadable log raises, never a 0
   ```

   On a pass, stop the server and write the record of what was checked:

   ```sh
   python3 tools/serve_checked.py write --config <config> --card-limit 31998 --window <tag> \
       --hot <hot-counts> --capture <capture-summary> --high <highest-sample-MiB> [--tree <build>] [--record <file>]
   python3 tools/serve_checked.py fields --config <config> --card-limit 31998 [--tree <build>]   # what it records
   ```

2. **The serving boot, on the public port.** Boot the same bytes from a copy of the profile that differs only in its
   port line. Send it no request. Allow it only while the record matches the installed bytes:

   ```sh
   python3 tools/serve_checked.py match --config <config> --card-limit 31998 [--tree <build>] [--record <file>]
   ```

   After the boot check the boot lines only: the server's `/health`, no traceback, the twelve sizing lines with the
   record's hot counts and 0 cold, the card limit on 12 of 12 workers, no draft-fallback warning. A failed check stops
   the boot and nothing serves. After any change of kit, profile, configuration or card limit the full check of part 1
   is owed again; while the record matches, a return from a test arm is the serving boot alone.

**A new kit on a set that serves** is requalified in one window, nothing serving from the stop to the serving boot:
install the kit in both trees, boot it on the test port with the adopted lines unchanged, and require identity 7 of 7
to the accepted capture, the same store record set opened, none of the kit's new switches in the API server's
environment, the kit's dispatch witness on 12 of 12 workers, the store, API, pin and state checks PASS, the adopted lines
byte-identical, Xid 0 and the four-chat high under your line; record the reads and decode cells without judging them;
then write the record and make the serving boot. On a failure put the earlier kit back and boot it on its matching
record. Ours took 32 min 52 s (kit 10) and 33 min 45 s (kit 11). `tools/mhc_window_receipt.py` and
`tools/control_ahead_window_receipt.py` count kits 10 and 11's witnesses by booted worker.

The smoke test is part 1, on the test port: the lone short requests, the seven long prompts and the four-chat rehearsal.
The serving boot gets no request from the start procedure. Load takes 164 to 197 s from launch to ready on our host on
the serving set (134 s in windows s4 and p5).

## 5. Stop

Wait for a moment with no request, then send the server's process group SIGTERM and wait until all twelve cards are at
0 MiB. A move from one serving boot to another, with a matching record, took about 3 minutes without a server.

## 6. A test window

Every change was tried in a window: a rule with its lines and its prediction written before any data, a reference boot
(`Wr`) and a candidate boot (`Wc`) that differ by one change, both on the test port, the serving port down for the
window, then the way back to the serving set by the serving boot on a matching record. Our driver ran each window as a
chain of phases and stopped a chain at the first failed line; it is not published. The judges it called are:

| Step | Tool | What it decides |
|---|---|---|
| The switch reached the workers | `tools/pipe_witness.py LOG --mid {0,1} --hold N [--turn N] [--after-read]`; `tools/split_witness.py {log LOG \| procs PGID \| both LOG PGID} WANT` | the log's own lines, never `/proc/<pid>/environ`, which a titled worker blanks |
| The answers | `tools/closeness.py DIR REF ARM`, or after `DIR`: `--verdict REF ARM` (CLEAN, JUNK or UNCLEAR), `--keeps REF ARM MIN_FIRST MAX_DLP`, `--scaled REF ARM BASE MIN_PROMPT`, `--nearer REF ARM OTHER` | identity and closeness of two captures; exit 0 pass, 1 fail, 3 INCONCLUSIVE |
| Speed against a baseline | `tools/speed_floor.py SUMMARY LABEL READ [--n 4]` | each figure at least 95 % of window s4's; a gain reported, never required |
| Requests alone and together | `tools/slots_cells.py {run,judge,lineh,summary}` | speed cells, the starvation judge, alone against together |
| Memory | `tools/oom_lines.py LOG` | real out-of-memory lines |
| The serving record | `tools/serve_checked.py {fields,write,match,record-from}` | what was checked and whether it matches the bytes |
| The store's boot line | `tools/store_line.py LOG LABEL ROOT [SAME_SET]` | the store's line and the twelve tier digests |
| A kit's dispatch witness (kits 10 and 11) | `tools/mhc_window_receipt.py LOG RECEIPT`, `tools/control_ahead_window_receipt.py LOG RECEIPT` | the first dispatch of the mHC kernel and the control header's first transfer on 12 of 12 booted workers; exit 34 INCONCLUSIVE |
| The time budget | `python3 -I tools/timebudget_parse.py {log,traces,all} ...` with `VLLM_V41_ATTN_TIMING=1` and the torch profiler | the time per stage and phase; it reads, never writes |
| Probes without a model | `tools/allreduce_probe.py` (one board of four cards), `tools/gemm_split_probe.py` (one card), `tools/tm_dispatch_emulation.py [M ...]` (CPU) | the four-card sum, the multiply library's split count, and the CPU model of its dispatch |

A window's line that cannot decide says INCONCLUSIVE and is never a pass; the judges exit non-zero on a failure and on an INCONCLUSIVE. A
kit's new log lines are collected on one boot that judges nothing before any window judges by them. An Xid line, a hang
or a dead engine at any step: stop everything and start nothing until the cause is understood.

Our receipts went under `<build>/logs/window/<TAG>/` for each window and `run/state/` for the serving record and the
adopted lines. They are not in this repository: they hold generated text. The figures they gave are in
[docs/measurements.md](docs/measurements.md).

## 7. CPU checks of this repository

```sh
make test     # python3 -m pytest -q tests --ignore=tests/models -p no:cacheprovider
make check    # bash tools/check_data_boundary.sh --tree HEAD
```

On the tree of this page `make test` gives 41 passed and `make check` prints `data-boundary: OK`. The tests read real log
lines from `tests/data/`. Every published kit file matches its `MANIFEST` (491 of 491 files in 26 kits, 52 rows withheld; check with
`sha256sum` against the third column).
