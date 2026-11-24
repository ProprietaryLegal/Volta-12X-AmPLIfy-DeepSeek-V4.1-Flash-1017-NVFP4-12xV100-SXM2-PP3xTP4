# Tools

The judges and probes of the twelve-card build that are published in this release. They use the Python standard
library unless a line says otherwise, send public text only and record numbers, ids and digests, never prompt or
reply text. Paths that were the host's are placeholders read from the environment (`B12_ROOT` for the build's
tree, `MODELS_ROOT` for the model files).

| Tool | Use |
|---|---|
| `closeness.py` | Token-level closeness of two captures, with the strict equality judge (equal length, equal ids, the same key set; anything else INCONCLUSIVE): verdict, keeps, nearer, scaled |
| `oom_lines.py LOG` | The number of real out-of-memory errors in a server log (never the card limit's own announcement line) |
| `slots_cells.py` | N requests together: speed cells, the starvation judge, alone against together |
| `speed_floor.py` | A speed line: each figure at least a share of a baseline's |
| `opt2_decide.py` | Window o2's decision rule between two- and three-slot arms |
| `serve_checked.py` | The record of a passed serving check, and whether it matches the installed bytes |
| `store_line.py` | The conversation store's boot line judged |
| `split_witness.py` | Evidence that a library switch reached the workers (their `/proc` environment is blank once titled) |
| `pipe_witness.py` | The witness of kit 8's switches in a boot log |
| `allreduce_probe.py` | The four-card sum probe: NCCL against the engine's own all-reduce path (needs one board of four cards and the engine) |
| `gemm_split_probe.py`, `tm_dispatch_emulation.py` | The one-card probe of the multiply library's split count (needs one card and the engine), and the CPU model of the library's dispatch that predicted it |
| `timebudget_parse.py`, `classmap-nvfp4-12card.json` | The parser of a server log with `VLLM_V41_ATTN_TIMING=1` and of torch-profiler traces: the time budget by stage and phase |
| `mhc_window_receipt.py LOG RECEIPT` | Kits 10 and 11: the mHC prefill kernel's first-dispatch line counted on every booted worker, each bound to its boot's process id; writes the two counts, never overwrites a receipt |
| `control_ahead_window_receipt.py LOG RECEIPT` | Kit 11: the control header's first-transfer line counted the same way; exit 33 when a worker never printed it |
| `check_data_boundary.sh` | The data-boundary scan (weights, archives, corpora, private folders, large files, credential patterns) |

Tests: `python3 -m pytest tests --ignore=tests/models` (CPU, any machine).
