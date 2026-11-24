# PP3KIT10-over-cut

Kit 9Z plus a register-resident mHC transition kernel for prompt chunks, on by default (`VLLM_V41_MHC_PREFILL=0` is kit
9Z). Adopted in window aw5 and requalified and served from window aw6 (2026-11-24 12:28:30Z); superseded by kit 11 in
window aw7.

**What it changes.** The model's mHC transition (the expansion, the mixing coefficients and the collapse around every
attention and expert block) ran as several small kernels per layer. For prompt steps of 30 rows and above, the kernel of
`lib/v41mhc/v41mhc.cu` keeps a row's state in registers through the whole transition, with an exact coefficient stage.
Replayed on a separate V100 bench card over 776 transition calls captured from a real read, every call was bit-equal to
kit 9Z's, and a call of 4,092 rows took 2.048 ms against kit 9Z's 3.417 ms (MEASURED, medians). Decode steps are untouched. The Python side is
`model_executor/kernels/mhc/deepseek_v41_fused.py` and `deepseek_v41_regs.py`. The earlier kernel without the exact
coefficient stage was faster on the cards and moved the output past the closeness line twice (windows aw3 and aw4); this
one does not move it.

**Speed** (MEASURED, window aw5, same-window reference arm on kit 9Z, 175 W): cold reads +3.24 / +3.14 / +3.30 % at
94K / 250K / 420K tokens (4,825.6 / 4,634.7 / 4,396.2 against 4,697.0 / 4,507.1 / 4,278.5 tok/s), decode +0.0 to +0.5 %,
the seven long answers token-identical, the fullest card unchanged at 29,796 MiB.

**Files.** `MANIFEST` has 55 rows and is complete. Withheld: the compiled library `v41mhc.so` (one row): build it from
`lib/v41mhc/` with `build.py`; the row pins the bytes the build served, and `lib/SHA256SUMS` lists them too. Also the
store's eight modules, the packed experts' compiled library and `apply.sh`, as in kit 9X (`README.md` of
`PP3KIT9X-over-cut`). The kernel's CPU reference and tests are in `tests/v41_mhc/`.

**Published copy.** As kit 9X: reworded comment lines, their sha256 recomputed in `MANIFEST`.
