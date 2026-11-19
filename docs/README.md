# Documents

| File | What it is |
|---|---|
| [../README.md](../README.md) | the front page: what the build is, what runs, the measured tables, what worked and what did not, what is in progress |
| [ARCHITECTURE.md](ARCHITECTURE.md) | three stages and their cuts, what crosses a boundary, the scheduler's spacing rule, slots, prompt chunks and the two sources of another rounding, memory per stage, the card limit, the kits, serving without checks on the public port |
| [OPERATIONS.md](OPERATIONS.md) | how the build is installed, started, checked and stopped |
| [measurements.md](measurements.md) | every measured figure with its window, power limit and conditions; predictions against results |
| [measurements-2026-11-19-morning.md](measurements-2026-11-19-morning.md) | the bench and prefix-sharing measure of the morning of 2026-11-19 on the ten-line set (kit 9S before the split count), moved from the measurements page unchanged |
| [measurements-2026-11-18.md](measurements-2026-11-18.md) | the bench and prefix-sharing measure of 2026-11-18 on the set before that (kit 9N), moved from the measurements page unchanged |
| [history.md](history.md) | the windows from w1 to p1 in order, with what failed and why |
| [upstream-contributions.md](upstream-contributions.md) | the upstream pull requests of PLI Labs that this build rests on |
| [../CHANGELOG.md](../CHANGELOG.md) | what changed, by date |
| [../documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md](../documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md) | the plan for custom kernels and optimizations: where the build stands, the method, the time budget, the items in tiers A to D, what was dropped, the order of work |
| [../documents/reviews/UPSTREAM-REVIEW-20261117.md](../documents/reviews/UPSTREAM-REVIEW-20261117.md) | what the upstream projects did since the engine's base: 1Cat-vLLM #1482 (four stages on 16 cards), #1537 and others |
| [../documents/reviews/CHUNK-BOUNDARY-READING.md](../documents/reviews/CHUNK-BOUNDARY-READING.md) | why a differently cut prompt read rounds differently: the reading that led to the four-card sum probe |
| the split-count report (not in this first release; its findings are in `ARCHITECTURE.md` section 6 and `measurements.md`) | why the same layer-0 rows differ between tail chunks of 2,292 and 2,308 rows: the multiply library's split count |
| [../research/TWELVE-CARD-RESEARCH.md](../research/TWELVE-CARD-RESEARCH.md) | this build's own findings, and where each is written down |
| [../research/TWELVE-CARD-TIME-BUDGET.md](../research/TWELVE-CARD-TIME-BUDGET.md) | where the time of a read, a turn and a decode round goes on three stages (window t1) |
| [../engine/README.md](../engine/README.md) | every kit with its engine commit, what it is and its state |
| [../tools/README.md](../tools/README.md) | the published judges and probes |
| [../NOT-YET-PUBLISHED.md](../NOT-YET-PUBLISHED.md) | what this first release leaves out |

"Amendment N" and "PLAN section N" in these documents name entries of the build's plan, which is not in this first
release; every window's result is in `measurements.md` and `history.md`.
