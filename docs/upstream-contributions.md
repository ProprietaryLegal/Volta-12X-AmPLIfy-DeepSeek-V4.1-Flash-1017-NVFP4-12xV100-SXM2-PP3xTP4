# Our upstream contributions

Pull requests PLI Labs wrote for the upstream engine, 1CatAI/1Cat-vLLM, and what each means for this build. States
and dates were read on GitHub on 2026-11-17 (`gh pr list -R 1CatAI/1Cat-vLLM --author <account> --state all`).
"PLI Labs (founder)" marks pull requests opened from the personal GitHub account of PLI Labs' founder;
"ProprietaryLegal" is PLI Labs' organization account. Work by other people that this build uses is in
[CREDITS.md](../CREDITS.md), not here.

## The one this build rests on

**[#1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031): "[Model][SM70] DeepSeek-V4.1-Flash on V100: PP3xTP4 (12
GPUs, 256K) and PP2xTP4 (8 GPUs)".** PLI Labs (founder), opened 2026-10-07, open on 2026-11-17.

It serves DeepSeek-V4.1-Flash from the official checkpoint on V100 cards: twelve cards as three pipeline stages of
four, or eight cards as two stages with part of the experts in host memory. What this build takes from it (READ:
the pull request's diff, quoted in the build plan, section 2):

- **The cuts.** "Only legal stage cuts are accepted: 20 for PP2; 14 and 28 for PP3." This build runs layers 0-13 /
  14-27 / 28-39 with `VLLM_PP_LAYER_PARTITION=14,14,12`, the pull request's own setting.
- **The mirror.** "Under PP3, stage 3 holds a mirror of kv source 20, which is shipped in the PP payload." Layer 20
  emits one row per token; the last stage writes the same rows into its own copy. Layer 28 recomputes its own
  top-k, so no selection crosses.
- **The proof that the layout is exact.** PP3 against PP2: bitwise identical over 2,744 tensors (UPSTREAM).
- **Its memory record.** HBM peaks of 31.5 / 32.0 / 28.5 GiB per card at 256K context (UPSTREAM; its dense weights
  were FP8, larger than this build's).

What this build adds to it. The pull request ran at about 3 tok/s on an eager engine (UPSTREAM). This build runs
the same layout on the fast engine of the eight-card builds (CUDA graphs, the DSpark drafter, packed KV, slots):
129.5 tok/s for one request on fresh text and 288.0 tok/s in sum for three together (MEASURED, window o2, 150 W),
306.6 tok/s in sum for four together (window s4, 150 W), and a prompt read of 3,733 tok/s at 200 W (PLAN Amendment
29). Since 2026-11-17 it is the owner's default DeepSeek server (PLAN Amendment 27). The kit is new code: the typed stage plan, the scheduler's spacing rule for three stages, the Engram applied on the first
stage, the card limit ([ARCHITECTURE.md](ARCHITECTURE.md)). No file of #1031 is copied into this repository.

The record of the earlier twelve-card work behind #1031 (port design, end-to-end results, hardware notes) is in the
research hub [volta-v100-research](https://github.com/ProprietaryLegal/volta-v100-research). Its branch is at
[ProprietaryLegal/1Cat-vLLM](https://github.com/ProprietaryLegal/1Cat-vLLM/tree/feat/deepseek-v41-sm70); patches,
launch scripts and results are at
[ProprietaryLegal/deepseek-v4.1-flash-sm70](https://huggingface.co/ProprietaryLegal/deepseek-v4.1-flash-sm70) (READ
in Volta sPLIt's `docs/upstream-contributions.md`).

## The native-weights route

| PR | Author | State on 2026-11-17 | Title | For this build |
|---|---|---|---|---|
| [#1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031) | PLI Labs (founder) | open (opened 2026-10-07) | [Model][SM70] DeepSeek-V4.1-Flash on V100: PP3xTP4 (12 GPUs, 256K) and PP2xTP4 (8 GPUs) | the layout and the cuts (above) |
| [#1030](https://github.com/1CatAI/1Cat-vLLM/pull/1030) | PLI Labs (founder) | open (opened 2026-10-07) | [Kernel][SM70] Generalise the MXFP4 MoE geometry and add skinny split-K 4/6/9/12/18 | V4.1's expert geometry (384 experts, top-6; drafter 128, top-3) on the SM70 MXFP4 path; #1031 is stacked on it |
| [#1021](https://github.com/1CatAI/1Cat-vLLM/pull/1021) | PLI Labs (founder) | **merged** 2026-11-16 (opened 2026-10-07) | [Kernel][SM70] Keep 32x32 UE8M0 block-FP8 dense weights in FP8 on Volta | the dense weights of the official checkpoint kept in FP8; not on this build's path (its non-expert tensors come from the GGUF) |

Kernels and measurements of #1021:
[ProprietaryLegal/sm70-fp8-block32-kernels](https://huggingface.co/ProprietaryLegal/sm70-fp8-block32-kernels) (READ
in Volta sPLIt's `docs/upstream-contributions.md`).

## The eight-card engine work, offered upstream on 2026-11-05

This build's engine descends from the eight-card builds, so it carries this work. Whether each switch is set in the
standing profile was not checked for this document.

| PR | State on 2026-11-17 | Title |
|---|---|---|
| [#1370](https://github.com/1CatAI/1Cat-vLLM/pull/1370) | open | [Model][SM70] DeepSeek-V4.1: PP2 x TP4 for the #1277 package (8 V100s) |
| [#1371](https://github.com/1CatAI/1Cat-vLLM/pull/1371) | open | [Bugfix][V1][PP] Preserve synchronous speculative step ownership |
| [#1372](https://github.com/1CatAI/1Cat-vLLM/pull/1372) | merged 2026-11-16 | [Perf][SM70][DeepSeek-V4] Tile live packed decode index scoring |
| [#1373](https://github.com/1CatAI/1Cat-vLLM/pull/1373) | merged 2026-11-10 | [Perf][V1][PP] Retain persistent request rows across empty steps |
| [#1374](https://github.com/1CatAI/1Cat-vLLM/pull/1374) | open | [Perf][SM70][DeepSeek-V4] Shard the DSpark main projection output rows |
| [#1375](https://github.com/1CatAI/1Cat-vLLM/pull/1375) | merged 2026-11-16 | [Spec Decode][DeepSeek-V4.1] Coupled sampled DSpark drafts with keyed verification |
| [#1376](https://github.com/1CatAI/1Cat-vLLM/pull/1376) | open | [Perf][DeepSeek-V4.1] Skip layers 21-39 in early prefill chunks (VLLM_V41_PREFILL_SKIP_ROWS) |
| [#1377](https://github.com/1CatAI/1Cat-vLLM/pull/1377) | open | [Spec Decode][DeepSeek-V4.1] Coupled drafts and keyed verification on vocabulary shards |
| [#1379](https://github.com/1CatAI/1Cat-vLLM/pull/1379) | open | [Model][MRV2][DeepSeek-V4.1] DeepSeek-V4.1 and its DSpark drafter on Model Runner V2 |
| [#1381](https://github.com/1CatAI/1Cat-vLLM/pull/1381) | merged 2026-11-10 | [Bugfix][DeepSeek-V4.1] Restore packed KV bytes and reserve speculative rewind span |
| [#1382](https://github.com/1CatAI/1Cat-vLLM/pull/1382) | merged 2026-11-10 | [Perf][SM70] Admit small aligned FP16 sum2 messages to the TP4 push all-reduce (VLLM_V41_SUM2_PUSH_ANY, default off) |
| [#1383](https://github.com/1CatAI/1Cat-vLLM/pull/1383) | merged 2026-11-16 | [Perf][SM70][DeepSeek-V4.1] Build the routed-expert plan inside the q8 quantize launch (VLLM_V41_PLAN_FUSE, default off) |
| [#1384](https://github.com/1CatAI/1Cat-vLLM/pull/1384) | merged 2026-11-16 | [Perf][SM70][DeepSeek-V4.1] Fast routed-expert decode kernels (VLLM_V41_EXPERT_FAST, default off) |
| [#1385](https://github.com/1CatAI/1Cat-vLLM/pull/1385) | merged 2026-11-16 | [Perf][SM70][DeepSeek-V4.1] Instruction-diet routed-expert kernels (VLLM_V41_EXPERT_DIET, default off) |
| [#1386](https://github.com/1CatAI/1Cat-vLLM/pull/1386) | merged 2026-11-16 | [Perf][SM70][DeepSeek-V4.1] Reduce decode index scores over the live prefix through an IPC bank (VLLM_V41_INDEX_REDUCE, default off) |
| [#1387](https://github.com/1CatAI/1Cat-vLLM/pull/1387) | merged 2026-11-16 | [Perf][SM70][DeepSeek-V4.1] Compact-scheduled grouped prefill GEMM for IQ routed experts (VLLM_V41_PREFILL_IQ_GEMM, default off) |
| [#1388](https://github.com/1CatAI/1Cat-vLLM/pull/1388) | open | [Model][SM70] DeepSeek-V4.1 indexer: fused SM70 prefill index scores (VLLM_V41_INDEX_FUSE, default off) |
| [#1389](https://github.com/1CatAI/1Cat-vLLM/pull/1389) | open | [Perf][SM70] DeepSeek-V4.1 decode: fuse attention prep and post kernels (VLLM_V41_FUSE_ATTN_PREP/POST, default off) |
| [#1390](https://github.com/1CatAI/1Cat-vLLM/pull/1390) | merged 2026-11-16 | [Perf][SM70] DeepSeek-V4.1 decode: dense GGUF mat-vec on a v41k dmv13_x kernel (VLLM_V41_DMV_X, default off) |
| [#1380](https://github.com/1CatAI/1Cat-vLLM/pull/1380) | merged 2026-11-09 | [Docs] README: DeepSeek-V4.1-Flash-1017 on 8x V100 PP2 x TP4 (community result) |
| [#1488](https://github.com/1CatAI/1Cat-vLLM/pull/1488) | merged 2026-11-16 (opened 2026-11-11) | [Docs] README: DeepSeek-V4.1-Flash-1017 NVFP4 hot experts on 8x V100 PP2 x TP4 (community result) |

All of these are from the ProprietaryLegal account. Titles as GitHub shows them on 2026-11-17.

## Earlier

| PR | Author | State | Title |
|---|---|---|---|
| [#651](https://github.com/1CatAI/1Cat-vLLM/pull/651) | ProprietaryLegal | merged 2026-09-26 (opened 2026-09-17) | [Model][SM70] Accept DFlash embed replica across split load_weights calls |
| [#100](https://github.com/1CatAI/1Cat-vLLM/pull/100) | PLI Labs (founder) | closed, not merged (opened 2026-07-10) | Fix GPTQ/compressed-tensors WNA16 on sm_70 (V100): auto-detect Volta Marlin, actionable errors (#87, #94) |
| [#96](https://github.com/1CatAI/1Cat-vLLM/pull/96) | PLI Labs (founder) | closed, not merged (opened 2026-07-03) | sm70/V100: fail loudly when torch<2.10 skips _C_stable_libtorch; tolerate UUID-form CUDA_VISIBLE_DEVICES; add sm_70 build guide |

## In this repository, not offered upstream

No pull request has been opened for this build's own work. What exists
here and could be offered:

- the third pipeline stage for the #1277 package on the fast engine (the typed stage plan, the mirror, the Engram
  applied on the first stage, the commit protocol on stages that do not sample);
- the sync scheduler's spacing rule for a pipeline deeper than two, and the slot scheduler's stall guard;
- slots above four;
- the card memory limit;
- the finding that the dense multiply library picks its split count from a chunk's row count, so that a prompt's
  last chunk rounds by its size (PLAN Amendment 23, measured on one card), and the reading that a snapshot hit takes
  the oldest page of a token hash rather than its own read's (the split-count report (not in this first release)
  section 6). Both are in the engine of every V100 build of this lineage; neither has a fix here yet (kernel plan
  items B4 and B5).

The kernel and optimization plan names what should go back
([documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md](../documents/plans/KERNEL-AND-OPTIMIZATION-PLAN.md), section 5):
the step spacing rule for a synchronous pipeline of three stages, which belongs with PLI Labs' pull request 1031, and
the fix for the half-precision large sums once it exists here. The tier digest of the two-tier loader path is the
store's own change (a private working repository), not this build's.

Whether and when to offer these upstream is the owner's decision.

## v100-skinny

[dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny) is Dennis Vertzyas' work. It is not ours. The
ProprietaryLegal account holds a mirror of it for reference. The research hub carries copies of some of its files, with its
LICENSE and NOTICE.
