# PP3KIT9X-over-cut

Kit 9W plus everything built and not yet serving on 2026-11-19, under the owner's decision to switch them on: the
conversation store, the Messages tokenize route, kit 9T's lean pipeline hand-off, kit 9U's packed-576 experts and kit
9V's exact four-card sum (still default off). Qualified with everything but the exact sum in window x2 and served from
2026-11-19 18:46:11Z on twenty adopted lines; superseded by kit 9Z on 2026-11-23.

**What it adds over 9W.**
- The conversation store (`VLLM_V41_STORE=1` with four byte budgets and a folder): finished prompts' KV state saved to
  host memory and disk, restored on a later request that contains the whole saved prompt, and pinned entries that a
  client warms on purpose. Its record set is fingerprinted by the numerical switches, so a boot with different
  arithmetic opens a new, empty set.
- `POST /v1/messages/tokenize`: the prompt token ids `/v1/messages` would run for the same body.
- `VLLM_V41_PP_LEAN_HANDOFF=1` (kit 9T) and `VLLM_V41_EXPERT_PACKED576=1` (kit 9U); both had failed their speed lines
  alone (windows c3b and c7a) and went on by the owner's decision, their identity and memory lines still deciding.
- `VLLM_V41_TP_EXACT_SUM` (kit 9V), unset as served.

**Files.** `MANIFEST` has 52 rows over the cut and is complete. Withheld from this copy:
- the conversation store's eight modules: `entrypoints/serve/v41_store/__init__.py`,
  `entrypoints/serve/v41_store/api_router.py`, `models/deepseek_v41/kv_store.py`, `models/deepseek_v41/store_dump.py`,
  `models/deepseek_v41/store_io.py`, `models/deepseek_v41/store_worker.py`, `v1/core/sched/v41_store.py`,
  `v1/core/sched/v41_store_files.py`. The store is a separate private working repository's code, and its own tests and
  probes read the host's saved conversations in place. The engine files that call it are published; with the eight
  modules absent the kit does not apply, which is why their rows stay: each sha256 shows what the private kit holds.
- `v41qpn576.so` and `v41qpn576.sha256` (two rows): the packed experts' compiled library, which the installer copies
  from the build; its source is published under `kit/PP3KIT9U-over-cut/lib/v41qpn576/`.
- `apply.sh`: the wrapper that installs the packed-576 link; its code defaults to the host's pack folder.
  `apply_files.sh`, `rollback_files.sh`, `rollback.sh` and `packs.py` (the pack manifest check) are here.

**Published copy.** Comment lines naming private hosts, repositories or tools were reworded in some kit files and their
sha256 recomputed in `MANIFEST` (code unchanged); the MANIFEST's own header line says how many.
