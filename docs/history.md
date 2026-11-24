# History

How the twelve-card build got to its standing configuration of four slots and became the owner's default DeepSeek
server: the windows of 2026-11-16 and 2026-11-17 in order, one paragraph each, with what failed and why. Times are
UTC. The rule and the full result of every window are in the build's plan, section 9 (not in this first
release; "Amendment N" below names its entries). The figures are collected in
[measurements.md](measurements.md).

A window is a planned period in which what serves is stopped, test arms are booted on the cards and a server is
restored. Every window was run by the build's owner, who alone starts and stops anything on the server.
Every window's acceptance rule was written before its data.

## Before the first window (2026-11-16, about 23:17Z to 2026-11-17, 00:15Z)

The owner ordered the work at about 23:17Z: every routed expert at full NVFP4 precision across all twelve cards,
researched and planned at once. The planning read the engine, found that it accepts two
stages cut at layer 20 and nothing else, and found the three-stage precedent in PLI Labs' own 1Cat-vLLM pull request
#1031 (cuts at layers 14 and 28, about 3 tok/s on an eager engine). The first kit (5 files, 463 lines added) was
written and tested on a CPU: 59 tests on a CPU machine, 47 under the host's real import. The plan and its acceptance rule
were fixed at 23:49Z, before any data. At 23:59Z the owner decided that this would be its own build, kept
separate in its own directory and repository. The build got its own tree on the host (its own engine copy, the
kit applied there only) and its own repository. At 00:04Z the owner chose two slots at full precision. A
usage-ranked mix for twelve cards, prepared for an hour on a misreading of the owner's words, was withdrawn on the
owner's correction. At 00:13Z the owner asked for test prompts to be replayed on the build once it ran, so the
driver got a phase that leaves the build serving.

## w1: the first three-stage boot answers wrongly (00:16Z to 01:01Z)

The preconditions, the reference captures, the stop of the default, the eight hand-off probes and the all-native
reference arm were clean. Arm X, three stages holding a quantized expert mix, booted as predicted and failed
rule line 1: 15 of 15 answers differed from a two-stage reference boot of the same experts. The pattern was exact:
token 0 right, a divergence at token 1 on 12 of 12 prompts, three to five junk tokens, then the reference's answer
for 100 tokens or more. A chat alone did not repeat itself. Nothing served and the server was restored. The cause was not found by reading.

## The cause (01:34Z) and kit 3

The cause was found by running the real engine loop on a CPU. Under sync scheduling nothing spaced a request's
steps by the pipeline depth. The stages before the last receive a step's sampled token and next drafts one step
per stage later. At two stages the engine's call sequence happens to match. At three stages the first decode step
after a prompt ran one call before its packet arrived, and the first stage embedded an earlier request's tokens.
Kit 3 sets the next eligible decode step to the current step plus the pipeline depth, as the async scheduler
already did. Its test fails in 17 of 28 cases before the fix and passes after. A second three-stage defect was
found by test on the way: the slot scheduler's stall guard counted two empty held steps as a stall, which three
stages produce in normal work. The acceptance rule for arm X was amended before any kit 3 data (Amendment 3): the
default itself has two answer states across boots, so digest equality cannot be the test. Arm X must read CLEAN by
the closeness tool and pass the alone-against-together check.

## w2: the diagnostic window (from 01:19:34Z)

This window ran with kit 2, which still had the faulty cadence, to isolate the mechanisms. The prediction was
written at 01:34Z before its three-stage results: two stages on the build's engine clean, every three-stage arm
junk. The two-stage arm read CLEAN. The arm with only the Engram hand-over and the middle commit read JUNK, 12 of
12. The arm with only the mirror read JUNK, 12 of 12. So the fault was in the cadence of any three-stage pipeline,
not in one mechanism. The one-slot arm passed its boot checks and then, at 01:49:39Z, the four cards of the first
stage logged Xid 43 within 48 ms. The chain stopped by the crash rule and everything was held for the owner's word. At
about 01:55Z the owner said to keep building and fix the errors. Which
device check fired was not established. The one-slot arm was clean in the next window with the fix in.

## w3: the fix on the cards, and the first all-NVFP4 boot (about 02:1xZ to 03:01Z; served to 04:27Z)

With kit 3, arm X passed rule line 1: closeness CLEAN with 0 of 12 early divergences, and the
alone-against-together check passed. The one-slot arm read CLEAN with no Xid. Then arm F2, every routed expert in
NVFP4 at two slots, booted for the first time: 15,360 hot and 0 cold. One request alone decoded at 106.5 tok/s on
fresh text, and three sent at once at 108.5 tok/s each (two slots). The highest card
sample was 31,808 MiB on the middle stage, 190 MiB under the owner's line, as the plan's arithmetic had named the
middle board the tight one. The owner's quality score line came out INCONCLUSIVE. The build then served from
03:01:22Z for an external client's replay of test prompts. Under that load the
fullest card crept from 31,778 to 31,836 MiB in half an hour and stepped to 31,988 MiB at 04:11:51Z, 10 MiB under
the line. The step is unexplained. No stage showed a floor that keeps rising. The owner stopped the build at
04:27:12Z with no out-of-memory line and restored the default.

## s1 and s2: more slots on the quantized mix (about 04:3xZ to 05:49Z)

After w3's first figures the owner allowed tests of four, six and eight requests at once. Kit 4 raised three
guards from four slots to eight. In window s1, three slots and four slots passed on the three-stage engine with
the quantized expert mix. Four at once gave the same total as three. The six-slot arm booted, served, and its engine
core died on the first requests: a log line named a slot's fields by indexing the string 'abcd', a fourth place
sized for four slots outside the three guards. Kit 4's CPU test had driven the slot policy without the
scheduler's real logging. Kit 4b fixed it with a test through the real engine loop. Window s2 started the
six-slot arm again at 05:35:09Z. At 05:45:57Z the owner ended the testing of the quantized mix: the full-NVFP4 build
was to be finished and optimized. The owner stopped s2 at 05:48Z. No six- or eight-slot verdict exists.

## w4, first run: the memory kit does not boot (05:54Z to 06:02Z)

Kits 5 and 6 were prepared while w3's build served. They take the token embedding off the middle stage (108.5 MiB
per card) and let the first stage apply layer 14's Engram as its last operation, so that no Engram rows cross to
the middle stage. Kit 6 adds a card memory limit that is off unless a variable sets it. In window w4, on kit 6b,
arm X did not boot: the first stage's four ranks refused with the kit's own check, "the Engram of layer 14 was
not loaded". The GGUF loader drops the tensors of a layer that is missing on a stage before any payload is read.
The kit kept the moved Engram under another name and renamed its tensors later, in a function that never saw
them. The kit's test had loaded weights into module trees, not through the loader the server uses. It was the
second kit of the night whose CPU test did not go through the server's path. Nothing served and the default was
restored.

## w5: kit 6c passes, two slots serve (about 06:2xZ to 07:12:05Z)

Kits 5c and 6c keep layer 14's Engram at its checkpoint place on the first stage and let the loader's stage
filter ask the model which tensors a stage holds. Their test runs the pack's real tensor inventory through the
real loader path, with counts held to the server's own log lines. On the cards, arm X read CLEAN and not worse
than kit 3 (the same 115 tokens and 0.0063), so the moved Engram computes the same thing. Arm F2 showed 15,360
hot and 0 cold, loaded in 134 s and peaked at 31,336 MiB (window w3: 31,808). A rehearsal of two grown chats peaked at 31,334 MiB. Releasing the allocator's cache at 512 rows gave the same peak and made larger
turns' first token twice as late, so that idea was closed. The build served two slots from kit 6c. After w5 the
quality line was replaced (Amendment 10): the owner's score could not reach its threshold however close an arm
was. The receipts themselves were clear: the all-NVFP4 build tracks the all-native reference for a median 208
tokens.

## o1: a third slot is closed by a rule that was mis-scaled (07:55Z to 08:26Z)

The idea: a third slot fills a three-stage pipeline, and the room for it on the middle stage is in the working
memory of a prompt read, which smaller prompt chunks would cut. Arm F3c, three slots with 4,096-row chunks,
booted with 0 cold experts and read CLEAN on short prompts. On long prompts it failed rule line c: the median
mean log-probability difference against the two-slot arm was 0.0333 against 0.0126 allowed. The arm ended there
and the chain served two slots. The limit 0.0126 had come from short prompts. The owner then measured on the
serving build that the same configuration repeats long prompts bit for bit and that short prompts are identical
across chunk sizes. So the two builds differ only where their forwards hold different numbers of rows: a
differently cut read is the same model in another rounding.

## o2: three slots with 4,096-row chunks are accepted (09:09:05Z to 10:08:51Z)

The rule was rewritten before the window (Amendment 12). A comparison arm with every chunk 64 rows shorter
measures how far a harmless change of the cut moves long-prompt answers, and the three-slot arm may move them at
most twice as far. The prediction held: the 64-row shift moved them by 0.0307, the halved chunk by 0.0331. Arm
F3c passed every line: three requests together 288.0 tok/s in sum on fresh text, a
prompt read at 96 % of the two-slot arm's speed, and a highest card sample of 30,052 MiB, 1,946 MiB under the
line. Two other three-slot arms kept 8,188-row chunks. One of them awaited every prompt hand-off before the next
forward (kit 7). It bought no memory: 31,664 MiB against 31,628 MiB without the switch. Both predictions for
those arms were wrong. Kit 7 was closed and kit 6c stays the build's kit. The third slot also keeps a long chat
warm: a turn after another chat was read took 2.1 to 2.6 s to its first token against 46.7 s at two slots.
Amendment 13 then fixed the standing configuration and how it is served.

## The chunk-boundary reading (10:51Z) and the step 0 probe (11:19Z)

A read-only study of the engine showed that nothing lossy crosses a chunk boundary, and that the kernels chosen
by row count do not separate the measured cases. Its hypothesis: the sum over the four tensor-parallel ranks. A
probe on one board, with no model, confirmed it at the engine's all-reduce entry. A sum small enough for the
engine's own path is added in FP32 and rounded once. A larger sum goes through NCCL in FP16, in an order that
depends on the buffer's size and the element's place in it. The NCCL sum differs from the exact one in 32.89 %
of elements. It is the engine's general arithmetic, not something of three stages. A change that removes it is a candidate, not work
in progress.

## k1: the card limit on the cards, and a false alarm (10:53:38Z to 11:13:38Z)

The standing configuration was served on kit 6c under the card limit of 31,998 MiB. Every line held: load 134 s,
0 cold, the limit on all twelve workers with the predicted allocator limits, the accepted long-prompt capture
repeated 7 of 7, and three grown chats at once peaking at 30,054 MiB. Then the chain printed "out-of-memory lines
12", refused to serve and booted once more without the limit. There was no out-of-memory event. The driver's
pattern matched the card limit's own announcement line, which names the exception it would raise, once per
worker. The owner found it by reading the log. The cost was a second boot, about 9 minutes more without the
default. The second boot repeated the first, which showed that the limit changes no token and no memory mark.
Amendment 17 followed: the limit is on for the standing build, nothing is booted twice, and every pattern a
check applies to a log is run against a real log of that kind of boot before a window. It was the third time a
check of this build met its own text or path on the cards for the first time.

## st1: the store's first window is stopped by the store's own check (12:42:04Z to about 12:59Z)

The owner's rule is that the conversation store and the pinned-prefix cache must work on every build. Two store
windows were prepared with the store's maintainers, under a strict stop rule because they fall in a working day. In window st1,
the first boot had the store's kit in the tree and the store off: 0 cold, the limit on twelve workers, the
accepted capture 7 of 7, three grown chats at 30,052 MiB. So the store's kit moves no token when it is off. The
second boot switched the store on. All twelve workers logged it, and then the engine core refused to start: the
workers' tier digests had not reached the scheduler, and the store's fingerprint must cover them. The store
records a worker's tier digest only in the engine's three-tier loader path. This build's
profile uses the two-tier path, which recorded none. The refusal is the store working as designed. The stop rule
restored the default. No record set was made. The plan's table of what had never met a server had not
named this loader path: nobody had booted the store against this build's loader configuration, on any bed. The
fix is in the store's kit, by the owner's rule of no bypass: in the two-tier branch it records, per rank, the
digest of the arrays that branch builds, with a CPU test through worker report, collection and scheduler with
twelve workers that fails on the earlier bytes. The store's kit of engine `e6df5ed` (a private working repository)
carries it and is pinned by the build.

## Between st1 and s4 (12:51Z to 13:17Z)

At 12:51:07Z the owner asked for four slots. Amendment 18 fixed the rule for a four-slot arm on
kit 6c alone, under the card limit, at 12:57:51Z, before its data. At 13:08:44Z the owner confirmed the build's public
name. At 13:12:44Z the owner ordered the working repository made formal, with the Volta architecture research
and the custom-kernel research copied in, and at 13:17:32Z a private GitHub repository for it.

## s4: four slots pass every line (13:32:14Z; the arm 13:33:59Z to 13:47:00Z)

The arm was the standing three-slot profile with one line changed, the slot count, on kit 6c alone and under the
card limit. The reserves stayed as they were, on the reasoning that a slot's own state lowers the planner's
budget by itself. The boot showed 15,360 hot and 0 cold experts, so that reasoning held. The accepted long-prompt
capture came out identical 7 of 7: a slot changes no token. Each chat alone answered as four together. Four
requests together gave 306.6 tok/s in sum on fresh text, against 288.0 for three together in window o2. The plan
had predicted the same sum as three, and it rose 6 %. One request ran at 128.1 tok/s and the prompt read at 3,284
tok/s, both where three slots had them. The rehearsal with four grown chats at once peaked at 30,398 MiB, 1,600
MiB under the owner's line, against a prediction of about 30,400. Two things got slower, as expected when four
chats share three stages: four documents read at once reached their first token after 129 to 166 s each (three
at once: 105 to 121 s), and a larger turn with four chats open waited 9.33 s (three chats: 8.16 s). Turns under
2,048 new tokens were faster than predicted: 4.12 s against about 6 s. The owner read the arm's own last line
before the verdict was reported. The chain ended in `final` and the default serves again.

## After s4: Amendment 19 (13:48Z)

Four slots stand. `window/chain_serve.sh` serves them by default and the serve phase's rehearsal runs four grown
chats. Three slots stay available as `SERVE_CONFIG=F3c`. The store's windows will run at four slots, cut in two
windows because one would keep the default down for over 100 minutes (ESTIMATED), on a new empty root, with the
fixed store kit. Their rule gained two conditions from st1: all twelve workers log their tier digest, and a
restart finds the same record set. The order of the work after that is in the kernel and optimization plan
([documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md](../documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md),
section 6).

## The store's window stops on the owner's own check (15:47Z to 15:56Z)

From 15:47Z one owner ran the server, this build and the conversation-store work together. Its first act was to
start the store window it had inherited, st3 (both halves
in one window since Amendment 20, when the store's tools proved faster than estimated), with the store's kit (engine
`e6df5ed`) over kit 6c in both trees. Boot KA came up as asked, and for the first time on this build twelve workers
logged the store kit's two-tier tier digest with the predicted values. Then the driver said "24 expert boot lines,
want 12" and the stop rule restored the default. Its rank count took every line beginning "nvfp4_8r: TP rank", and
the store's kit logs a second such line per worker even with the store off. No line of the build had failed. The
cause was the new owner's, and it was the fourth of its kind (Amendment 17): the kit's new line was in the kit and
in the store tool's own pattern, and the driver's readers had not been run on it. The fix fed the rank and hot
counts from the sizing lines only, with a test on the two real logs, and the rerun st3b started at 15:56:24Z
(Amendment 21).

## st3b: the store works, and the build's own reads disagree (15:56Z to 16:20Z)

Boot KA passed: the store's kit with the store off changed no token (7 of 7) and no memory mark. Boot KD1 was the
first boot of this build with the store on. The record set was made and handed to twelve workers, the state check
passed on all twelve three times, and a 130,000-token chat was read and saved; the highest card sample with the
store and its dump on was 30,398 MiB. Then the pinned-prefix cache test M12 said INCONCLUSIVE and the window stopped by
its rule. The chats' restored state equalled the saved state byte for byte both times. What differed was the
test's second reference: it began from layer-0 window pages that another read of the same tokens had written, and
those pages differed in six of nine blocks between a read whose tail chunk had 2,292 rows and one whose tail had
2,308; tails that fall on one split count gave identical pages. Two things were read in the dump and
confirmed later: pages are shared between requests by token hash, so a request can start from another
read's pages; and on this build the same tokens in a tail chunk of another row count come out in a few other bytes.
The mechanism was open. The default served again from 16:20:29Z (Amendment 22).

## The cause, found by reading and proven in 25 seconds on one card (16:24Z to 16:52Z)

A read-only reviewer was given the dump and the code. It found that layer 0's window row is a per-row function of the
token and its position, and that one operation on its path depends on the chunk's row count: the `wkv` projection,
which above 32 rows runs the dense multiply library's general multiply, whose cost model picks the number of
splits of the inner dimension from the row count. 2,292 and 2,308 rows straddle 2,304 = 18 x 128 and fall on 10
splits and 1. The reviewer wrote the cost model again in Python and its predictions went on record. The owner then
ran a probe on one card of the spare board, with no model and no server, from 16:49:38Z to 16:50:03Z. The first
start at 16:48:55Z had failed at its first native call, because the source tree lacks two compiled libraries; the
probe was pointed at the installed package, the one that serves. The second start traced the split count at
sixteen row counts, and all sixteen equalled the model. Rows under different counts differed; rows under one count
were bit-identical; with the library's switch `TM_GEMM_TUNE=max_splits=1` all 120 pairs were bit-identical. It was
a second, separate source of "the same model in another rounding", beside the four-card sums of the morning. The
store's test lengths were chosen so that each test's two reads fall on one split count, and nothing else changed
(Amendment 23; the split-count report (not in this first release)).

## st4: every store test passes, and the owner's own line fails (17:15Z to 18:05Z)

Between the windows the owner gave the rest of the day's cards to this build. In st4 the store passed its whole test list on
twelve cards at four slots: the state checks on twelve workers, M12 and M1 at the new lengths (as predicted), a
restart that found the same record set and brought a 130,000-token chat back from disk with its state equal and
its answer identical, the pinned-prefix cache's check after the restart, and 0 waits over 2,976 saves. As it would serve,
the store moved no token and cost no card memory. One line failed, and it was the owner's own. The restore pair
restored a 131,466-token turn after a restart with its first token after 1.23 s, where the first boot had needed
43.84 s, but its answer was not token-identical to the first boot's. The first boot's turn had left the chat's
prompt at its last token, as a client does that replays only the answer's text, and had been read cold from 0. A
cached turn and a cold read of one text differ in rounding on this engine. The line asked for more than the engine
gives; it was the wrong line to write, and it stays FAILED as written (Amendment 25). The way the first turn was
read cold became an item of its own.

## The build becomes the default (18:05Z to 18:48Z)

The window sv1 that followed put the build up to serve on `:18043` at 18:16:40Z with every start check passed.
Between 18:06Z and 18:14Z the owner decided three things: the build is to be optimized for long-context agent work
(one request at a time, prompts of 60,000 to 200,000 tokens that grow by small turns); the free card memory may be
20 % smaller; and the build is the new default server. The first re-weighted the kernel plan's order to the speed
of one request (Amendment 26). The second was read as a free-memory line of 32,100 MiB from the next arm. The third made the
build the default and the eight-card build the fallback (Amendment 27). The default's port is `:18042`, and a
public port must never hold a server that is still being checked, so the serve phase was split in two: every
check on the test port, recorded; then a serving boot on `:18042` that sends no request and needs a matching
record. The phases and the chains to start, move and return were built and tested on a CPU; none of it had met a
card. Clients followed the build to `:18043` at 18:17:25Z. The server's own counters showed where a long agent task
waits: a turn's new rows run all forty layers as one chunk through the three stages in series.

## 200 W, the move to `:18042`, and arm G's evidence that could not exist (18:57Z to 19:31Z)

The owner raised the power limit to 200 W: all twelve V100s were set at 18:58Z with a backup of the limits
script, and the build kept serving. A standing rule followed for one case: after a power trip at 200 W the caps go
back to 150 W and the build is started again.
Every speed figure before 18:58Z had been taken at 150 W, so arm G's speed line was given a same-power baseline
before its data (Amendment 28). At 19:17Z the build moved to `:18042` in 3 minutes, the new phases' first run on
the cards, as written. One gap showed: the split chain refused the window the move had made, because it carried no
record of the fallback's digests; it was copied by hand and the chains now carry it. Arm G then booted with the
split switch and stopped at its first line. Its witness asked that each of the twelve workers have the variable in
its `/proc` environment, and all twelve showed it unset. The witness's author measured why: a worker
sets its process title with setproctitle, which takes the memory of the environment block, so the `/proc`
environment of every titled worker is empty by construction while the process still reads its variables. The
library's own line "max_splits 1" stood 16 times in the arm's log and never in the standing build's. The switch
had been delivered; the rule had asked for evidence that cannot exist. The line was corrected before the rerun
(Amendment 29). On the serving build at 200 W the prompt read measured 3,733 tok/s against 3,284 at 150 W, and
decode moved by under 2 %.

## t1: the time budget, and the switch stays out (20:24Z to 20:43Z)

The second stop of the evening ran two arms between serving periods, with twenty minutes' notice to the server's users.
Arm T booted the standing configuration with the engine's own timers and the torch profiler and captured all 120
traces (Amendment 30's rule T1). From its log alone: a 4,092-row prompt chunk takes 793 ms on the first stage and
891 ms on the middle one, whose index grows with context; the expert path is 37 to 40 % of a chunk on those two
stages; about 200 ms per chunk lie outside any stage's forward; a decode round of one request is 25.1 ms; a small
turn on a 128,000-token chat reaches its first token in 1.24 s. Four of the probe's turns were read cold, both
4,200-token turns among them, for the reason the restore pair had met: the probe replayed answers without their
reasoning, so each turn left the stored prompt at its last token. The probe is being corrected, and the engine gap
behind it moved up the plan. Arm G passed its corrected first line and ended INCONCLUSIVE at its quality line:
against the all-native reference its distance was smaller than the standing build's (0.0079 against 0.0090), but
its first divergence came at 96 tokens where 104 are asked. A likely reading, not established: the reference was
captured under the default split counts, which the standing build shares. The switch stays out until a reference
in its own arithmetic exists. The build served on `:18042` again from 20:43:34Z by the serving boot alone on the
matching record (Amendment 31).

## Where it stands at the end of 2026-11-17

The build is the owner's default on `:18042` at 200 W, with the store's kit in the trees and the store off. In progress
and not on the cards: kit 8 (the turn pipeline, the middle-stage prompt skip and a hold for small turns beside long
reads, on the engine branch `pp3-12card-pipe`), the two arms the time budget's trace analysis asks for
(`research/TWELVE-CARD-TIME-BUDGET.md`, written the same evening: the cache release inside a first read and the
server's front end on a continued chat), the steps before the store is switched on in what serves, and
the question of a higher-precision Engram table. The order is the kernel and optimization plan's section 6.

## The evening review and window p1 (2026-11-17 21:20Z to 2026-11-18 00:07Z)

The time budget's trace analysis was written at 21:20Z (Amendment 32): a decode round of one request is 26.25 ms in
the traces, of which the experts are 29 %, the dense multiplies 21 %, the drafter 10 % and 14 % lies outside any
kernel; the round grows to 31.0 ms at 249,000 tokens, and the growth is the index. Two findings became arms: the
cards idle in every prompt chunk step while the host frees and re-allocates card memory, and the server's front
end costs a third of a small turn's wait on a long chat. The owner then ordered the fastest and biggest gains first,
switched on only when proven, and no trade of quality for speed (Amendment 34).

An independent review read the whole campaign at 21:32Z to 22:21Z and found four things the owner had measured
wrong, each re-run or re-read before acting on it (Amendment 35). The token cache built for the front end returned
wrong token ids on a hit when an edit removed a special token at its cut: it was held, repaired, and re-proven by
two independent fuzzes (3,200 edits on the owner's side, 0 sequences different from a full encode; Amendment 36).
The equality judges accepted a cut answer as identical: they were made strict (equal length, equal ids, a missing
key INCONCLUSIVE), and the three historical identity results were re-checked on whole sequences and stand
(Amendment 37). The cache-release hypothesis was overstated: about half of the host's freeing and re-allocating
overlaps card work, so the prediction was halved to about 9 %. And for long agent work one request's decode, not
the first token of a turn, is the largest wait: the order of work was changed to decode first. The owner also put
the precision questions (the Engram table at official precision) at the very end of the plan.

Window p1 (23:30Z to 00:06:33Z, 36.5 minutes without a server, Amendments 38 and 39) put kit 8 on the cards for
the first time, one boot per variant. With its switches off, kit 8 answered the accepted long prompts token for
token as kit 6c. The release setting `VLLM_V41_EMPTY_CACHE_TOKENS=8192` passed every line: reads 12.6 % faster at
61,000 tokens and 9.9 % at 93,000 (its floor was 8 %), identical answers, and no higher card memory; the
prediction of a higher memory mark was wrong. It is adoptable by the next window's adopt boot. The middle-stage skip
was not measured: its boot was active, but the arm's witness failed it on vLLM's own warning about an unregistered
environment variable, a line no CPU test had produced. It was the fifth stop of an arm by a check that had not met
a real log of its kind. The first valid figure for a turn of 4,200 new tokens on a 128,600-token chat came from the
same window: 3.31 s of reading, not the 2.37 s the turn model had assumed. The build served again from 00:06:33Z,
unchanged.

## The fifth release: everything on, the slowdown, and one change per boot (2026-11-19 evening to 2026-11-24)

The fourth release's set did not stay the serving set for long. At about 16:50Z on 2026-11-19 the owner decided to
switch on the conversation store, its pinned prefixes and every optimization already built. The lean hand-off and the
packed experts had failed their speed lines alone, and the store its timing line; they went on together as kit 9X with
the cached-token count of kit 9W, which an engineer had first thought was lost in the scheduler and was in fact dropped
in two serving files. Window x1 stopped three minutes after its boot on a witness that read a list of names instead of
the installed kit. Window x2 qualified the set on one boot (identity 7 of 7, every witness, the four-chat high 758 MiB
under b4a's as predicted), and kit 9X served on twenty lines from 18:46:11Z. A finding of the window: its cells step had
judged the A arm against itself, a pass by construction; it now records A and judges only a second arm against it.
Kit 9Y, the partial-prefix hit, was built that evening and has not run.

The community benchmarks ran on that set from 2026-11-20 to 2026-11-22. The speed measure after them found the server
about seven times slower. Clocks, power, memory and the drafter's acceptance were normal; the engine core's own turn had
grown to about 55 ms. The store's conversation tier had filled to its 400 GiB disk budget, and the scheduler asked the
store for its statistics on every step, which kit 9X answered by walking every entry. Window fx1 moved the tier aside
and the speed came back. Kit 9Z keeps the counts instead. Its first install, early on 2026-11-23, was refused by the
serving check, because no route qualified a new kit on the all-on set; kit 9X went back after 7 minutes. The route was
written as one chain, reviewed, and run as window z9a, which served kit 9Z from 16:20:36Z.

From the evening of 2026-11-23 a campaign worked through a ranked list of levers for decode and prefill, each
confirmed or closed by its own window or by a probe on a separate V100 bench card. One chain ran a list of arms, one
change per boot, with the reference first and its repeat last. Its first run (aw1) stopped at its own evidence check on
a measuring tool's progress line that its simulations, built on stand-ins that print nothing, had never met; the fix
put the evidence on a stream of its own and replayed every measuring part against the real tools. Window aw1f
measured: the speculative tree's counter cost 5 to 16 % decode where nothing was predicted, and its table closed the
tree; the tensor-core prompt scorer was exact and slower, and closed. Windows aw2 to aw4 showed the mHC transition
kernel fast, +4 to +5 % on every read, and twice outside the keep line. Its arithmetic differed from the reference by
one half-precision unit in a few hundred of twenty million outputs, too little to explain the loss; aw4 then showed
the stack token-identical across two boots of the same kit, so the loss was real. Kit 10's kernel adds an exact
coefficient stage: in aw5 it read +3.1 to +3.3 % faster, token-identical, and was adopted; aw6 requalified it and it
served from 12:28:30Z on 2026-11-24. The drafter's prompt tail moved nothing and closed.

The rest of the list closed on the bench card: the materialize-once mHC coefficient overlap (after vLLM #57603) missed
its per-call target, the group-wave expert prefill port of 1Cat-vLLM #1595 was exact and 5 to 6 ms a layer slower than
the resident pack, and a 4-bit pack for the drafter's query projections was slower than the FP8 GEMV it would replace,
because Volta decodes 4-bit values in software. The early control header on the pipeline hand-off was estimated at 0.8
to 1.2 % of a decode round, below any band one window can see; the owner had it installed, the adoption band became
the larger of 1.8 % and the window's own spread, and window aw7 requalified kit 11, which serves since 15:57:15Z.
