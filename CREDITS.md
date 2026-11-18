# Credits

Volta 12X AmPLIfy is PLI Labs' work: the kit that teaches the fast engine a third pipeline stage, the scheduler's
spacing rule for a pipeline of three, the middle-stage memory changes, the card limit, the window driver with its
rules, the serving form (the checks on a test port and a serving boot that sends no request),
the probes (the four-card sum, the dense library's split count), the time budget's tools, and the measurements. It stands on the projects below. Each is credited where its code or ideas are used. If
we have missed or misdescribed anyone's contribution, open an issue and it will be corrected.
Pull request states and authors were read on GitHub on 2026-11-17 where a line says so; the other lines are carried
from the credits of the eight-card builds (checked there on 2026-11-03 and 2026-11-09).

## The layout: 1Cat-vLLM #1031

- **[1CatAI/1Cat-vLLM #1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031)**, "DeepSeek-V4.1-Flash on V100: PP3xTP4
  (12 GPUs, 256K) and PP2xTP4 (8 GPUs)", opened 2026-10-07 by PLI Labs' founder (open on 2026-11-17). It ran DeepSeek-V4.1-Flash from the official checkpoint on twelve V100 cards as three stages
  of four, with `VLLM_PP_LAYER_PARTITION=14,14,12`. This build takes from it:
  - the two cuts, at layers 14 and 28, and the rule behind them ("Only legal stage cuts are accepted: 20 for PP2; 14
    and 28 for PP3");
  - the mirror of KV source 20 on the last stage ("Under PP3, stage 3 holds a mirror of kv source 20, which is
    shipped in the PP payload");
  - the evidence that the three-stage layout is exact: PP3 against PP2 bitwise identical over 2,744 tensors
    (UPSTREAM, the pull request's own figure).
  The pull request ran at about 3 tok/s on an eager engine. The kit in this repository is new code on the fast
  engine of the eight-card builds; no file of #1031 is copied here. The record of that earlier twelve-card work is
  in the research hub [volta-v100-research](https://github.com/ProprietaryLegal/volta-v100-research).
- **[#1030](https://github.com/1CatAI/1Cat-vLLM/pull/1030)** (same author; open on 2026-11-17): the MXFP4 MoE
  geometry for V4.1 (384 experts, top-6; drafter 128, top-3) and the skinny split-K, which #1031 is stacked on.

## Model and weights

- **DeepSeek** ([deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
  MIT): the model, its official MXFP4 routed experts, the DSpark speculative drafter, the config and tokenizer, and
  the Engram design.
- **NVIDIA** ([nvidia/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4),
  MIT): the NVFP4 pack whose routed experts this build serves, all 15,360 of them.
- **ISTA-DASLab** ([DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF),
  MIT): the GSQ-RCO quantization. The eight-card builds take the non-expert tensors and the IQ4_NL Engram from it,
  and this build loads the same files.
- **ggml** ([ggml-org/ggml](https://github.com/ggml-org/ggml), MIT): the GGUF container and the IQ quantization
  formats.

## Engine and kernels

- **1CatAI** ([1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), Apache-2.0): the SM70 vLLM fork.
  - **yangzhuxinyzx**: [#1277](https://github.com/1CatAI/1Cat-vLLM/pull/1277) (merged 2026-10-27), the
    DeepSeek-V4.1-Flash package for 4x V100. Volta sPLIt ported it to two boards; this build's engine descends from
    that port. The kit changes files of that package (`models/deepseek_v41/`) and of the engine around it.
  - **Peuqui**: [#742](https://github.com/1CatAI/1Cat-vLLM/pull/742) (merged 2026-10-03), the `sm70_skinny` QPN MoE
    backend for NVFP4 and MXFP4, which runs every routed expert of this build;
    [#1362](https://github.com/1CatAI/1Cat-vLLM/pull/1362) (merged 2026-11-16), "Skip the embedding, final norm, LM
    head and encoder towers a pipeline stage does not hold", the precedent for taking the embedding off the middle
    stage (our change is new code); [#1282](https://github.com/1CatAI/1Cat-vLLM/pull/1282) and
    [#1280](https://github.com/1CatAI/1Cat-vLLM/pull/1280), prompt-read work carried in the engine's patch series.
  - **kezboard233**: [#1157](https://github.com/1CatAI/1Cat-vLLM/pull/1157) (merged 2026-11-10), the native
    block-scale QPN8 GEMM, read for the kernel research.
  - Every author Volta sPLIt credits for its patch series: see its
    [CREDITS.md](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/CREDITS.md).
- **Dennis Vertzyas** ([dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny), MIT kernels and docs,
  Apache-2.0 fork patches): the QPN W4A16 kernels and their prepack, which reach the engine through #742, and the
  Volta bandwidth findings (the read ceiling of about 879 GB/s). Copies of some of its files are in the Volta research hub,
  [volta-v100-research](https://github.com/ProprietaryLegal/volta-v100-research), with its LICENSE and NOTICE.
- **OpenMMLab**: the TurboMind dense multiply library of lmdeploy, carried in the engine's tree under
  `csrc/sm70_turbomind/lmdeploy/` (its source files read "Copyright (c) OpenMMLab"; READ in the build's tree on
  2026-11-17). Its dispatch picks the split count of a multiply from the row count; this build found that on the
  cards and uses the library's own switch `TM_GEMM_TUNE` in its arm G (windows g1 and t1).
  `tools/tm_dispatch_emulation.py` re-implements that dispatch's cost model in Python to predict the split counts.
  The licence of the carried TurboMind sources is that of lmdeploy; the folder in the engine tree carries no
  licence file, so check lmdeploy's own repository before reusing them.
- **vLLM** ([vllm-project/vllm](https://github.com/vllm-project/vllm), Apache-2.0): the serving engine underneath,
  and its torch-profiler hooks used by the time budget (arm T).
- **NVIDIA** NCCL and the CUDA toolkit: the four-card sums and the hand-off probes.

## Our earlier builds

- **Volta Max uPLIft** (PLI Labs, Apache-2.0): the native NVFP4 expert path, the tier planner, the packs and the
  routing tally this build reads, the launcher and serve scripts this build's own are derived from, and the closeness
  method.
- **Volta Full NVFP4** (PLI Labs, Apache-2.0): the host-memory expert residency with which this build's all-native
  reference is computed.
- **Volta sPLIt** (PLI Labs, Apache-2.0): the PP2 x TP4 port and the engine underneath. Its credits (DeepSeek,
  ISTA-DASLab, 1CatAI and the authors of its patch series, vLLM, Phil Tompkins, TensorFold, sudoingX, Dennis
  Vertzyas) apply to everything this build runs on.
- **The conversation store and pinned-prefix cache** (PLI Labs, a private working repository): the store's kit that
  sits over kit 6c in this build's trees (store off in what serves), its test tools and its rule. Its whole server
  test list passed on this build in window st4.

## Research

PLI Labs' Volta architecture research and custom-kernel research, which this build stands on, are published in the
research hub [volta-v100-research](https://github.com/ProprietaryLegal/volta-v100-research), with the author and
licence of every file, including the third-party kernels and the pull request descriptions.
