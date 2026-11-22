# Community benchmarks on the twelve-card build (2026-11-20 to 2026-11-22)

The sets the [1Cat-vLLM README](https://github.com/1CatAI/1Cat-vLLM) cites, run against this build's serving port as it
serves: HumanEval+ and MBPP+ (EvalPlus), MATH-500, GSM8K, MMLU, MMLU-Pro, LegalBench, LongBench-E, and WikiText-2
perplexity. Every score is **MEASURED** by the set's own scorer against the set's own answers; nothing in the build
graded itself. The comparison column is other people's figures for the original DeepSeek-V4.1-Flash-1017 weights
(**UPSTREAM**, self-reported on their model cards, not checked by us) and vals.ai's LegalBench; each with its setting,
because the settings differ.

## What was measured

- The model: DeepSeek-V4.1-Flash-1017, every routed expert in NVIDIA's NVFP4 pack, the non-expert tensors from the
  ISTA-DASLab GSQ-RCO GGUF, on twelve V100-SXM2 32 GB cards as PP3 x TP4 (the build this repository describes).
- The serving set: kit `PP3KIT9X-over-cut` (the fourth release's `PP3KIT9S-over-cut` plus the packed experts, the lean
  hand-off and the prompt store, measured in window x2 on the evening of 2026-11-19; kit 9X and window x2's record are not
  yet in this repository and follow in the next release), four slots, 524,288 context, the DSpark drafter on.
- Thinking on (the server's default), temperature 0, the model's own end of turn as the stop.
- No server setting changed for the runs; they were ordinary traffic on the serving port, at most three requests at
  once so one slot stayed free for other users.
- A small relay between the scorers and the server did three things and nothing else: removed the scorers' stop
  strings (written for base models; they would cut a thinking model inside its reasoning), raised `max_tokens` to
  32,768 where a scorer asked for less, and set thinking on. It never changed a prompt or the sampling and never
  retried. 5,640 requests, 5,640 answered (HTTP 200); 5,602 ended at the model's own stop, 38 at the 32,768-token cap
  (an answer cut at the cap scores as wrong).
- Power: the cards ran at 175 W until a host restart on 2026-11-21 (a disk fault, not power), at 150 W from the restart
  to 2026-11-22 00:42Z, then 175 and 185 W. The cap changes speed, not answers; the runs resumed from the scorers' own
  caches after the restart, so no item was scored twice.
- Tools: EvalPlus 0.3.1, lm-evaluation-harness 0.4.13 (`local-chat-completions`, `--apply_chat_template
  --fewshot_as_multiturn`), math-verify 0.9.0, datasets 5.2.0.

## Scores

| Set | Items | Ours (MEASURED) | Original 1017 weights (UPSTREAM, self-reported) |
|---|---:|---|---|
| HumanEval / HumanEval+ | 164 (all) | **98.2 / 92.7** pass@1 | HumanEval 95.73 (157/164), one greedy completion (Infatoshi) |
| MBPP / MBPP+ | 378 (all) | **97.4 / 82.5** pass@1 (368 / 312 of 378) | none found |
| MATH-500 | 500 (all) | **97.0** (485/500) | none found |
| GSM8K | 1,319 (all) | **94.7** strict, 94.5 flexible | 96.97 (1279/1319) (Infatoshi); 93.5, step by step, greedy (MoonMath) |
| MMLU | 1,140 (20 of each of 57 subjects) | **92.5** ±0.8 (1055/1140) | 88.22, all 14,042 (MoonMath); 84.89, all 14,042, thinking off, 8-token answer cap (Infatoshi); 85.1, Sep release, 10 per subject (msuiche) |
| MMLU-Pro | 1,400 (100 of each of 14 categories) | **86.4** ±0.9 (1210/1400) | 85.7, 300-question sample, reasoning on; 69.90, 4,000-question sample, no reasoning (MoonMath) |
| LegalBench | 250 (lm-eval's 5 tasks, 50 each) | **72.0** ±2.6 (180/250) | 83.28 ±0.46, all 162 tasks, Sep release through DeepSeek's API, reasoning max, temperature 1 (vals.ai) |
| LongBench-E | 450 (9 English tasks, 50 each) | **68.8** ±1.5 | none found |
| WikiText-2 perplexity | 16,376 tokens (8 segments of 2,048) | **2.394** | none found |

Sources of the comparison column: Infatoshi, `huggingface.co/Infatoshi/DeepSeek-V4.1-Flash-1017-NVFP4-uncensored`
("unmodified base" rows of its comparison, four RTX PRO 6000); MoonMath, `huggingface.co/MoonMath-ai/DeepSeek-V4.1-Flash-1017-uncensored`
("original" rows); msuiche, `huggingface.co/msuiche/DeepSeek-V4.1-Flash-NVFP4`; vals.ai's DeepSeek-V4.1-Flash
LegalBench page. Read 2026-11-21. DeepSeek publishes no chat-model score for these sets; its card's base-model rows are a
different model on few-shot prompts and are not set beside ours.

What the column can and cannot say. Our MMLU and MMLU-Pro are samples (20 per subject; 100 per category) and run with
thinking on; Infatoshi's MMLU caps the answer at eight tokens with thinking off. vals.ai's LegalBench is all 162 tasks
through the API; ours is the five tasks lm-eval ships. The comparison is a sanity check that the compressed build did
not lose the model, not a ranking.

### Per-set detail

**HumanEval+ / MBPP+.** `evalplus.codegen --backend openai --greedy`, then `evalplus.evaluate`. EvalPlus's client gives up on a request after 100 s and sends it again; a thinking model's longest answers take minutes, so a small wrapper raised that limit to 7,200 s and turned off the client's silent retries. Prompts, sampling and scoring are EvalPlus's own. MBPP: two items
(Mbpp/100, Mbpp/260) came back empty after reaching the token cap and count as failures.

**MATH-500.** HuggingFaceH4/MATH-500, DeepSeek-R1's published instruction ("Please reason step by step, and put your
final answer within \boxed{}.") appended to each problem; scored with math-verify (`verify(gold, answer)`) on the final
answer, not the reasoning. 496 of 500 ended at the model's own stop.

**GSM8K.** lm-eval `gsm8k_cot_llama`, 8-shot, multiturn.

**MMLU.** lm-eval `mmlu_cot_llama`, zero-shot, `--limit 20`. Groups: humanities 92.7, social sciences 93.3, STEM 93.2,
other 90.8.

**MMLU-Pro.** lm-eval `mmlu_pro`, 5-shot multiturn, `--limit 100`. By category: biology 95, math 94, physics 94,
economics 92, chemistry 91, business 88, computer science 88, engineering 88, philosophy 88, health 81, law 79,
psychology 79, other 77, history 76.

**LegalBench.** lm-eval `legalbench` group (5 tasks), `--limit 50`, scored by lm-eval's `get-answer` filter: proa 98,
function_of_decision_section 80, abercrombie 78, corporate_lobbying 62, international_citizenship_questions 42. Of the
70 answers scored wrong, 18 gave the right label followed by an explanation the strict filter does not accept
(corporate_lobbying 11, international_citizenship_questions 7); counting those, 79.2. The table gives the strict 72.0.

**LongBench-E.** lm-eval `longbench_e`, `--limit 50` per task. The first 50 items of each task are its shorter end:
prompts median 4,277 and at most 12,877 tokens. Code completion 67.6 (lcc 71.7, repobench-p 63.5), few-shot 71.4
(samsum 37.9, trec 84.0, triviaqa 92.1), multi-document QA 78.9 (2wikimqa 77.7, hotpotqa 80.1), single-document QA 56.1
(multifieldqa_en 59.8, qasper 52.5). The build's long-context reading at 32K to 256K tokens is in the README's speed
tables.

**WikiText-2.** The shape 1Cat uses as a gate: wikitext-2-raw-v1 test text joined with blank lines, tokenized by the
model's tokenizer, the first eight consecutive 2,048-token windows, each sent as token ids to `/v1/completions` with
`prompt_logprobs`; perplexity = exp(−mean log-probability) over the 16,376 scored tokens. Segments 1.94, 1.52, 1.70,
3.24, 3.57, 2.20, 2.49, 3.40.

## Speed

Scores do not depend on speed; these figures say how fast the same build answers.

**The build's measured speeds (2026-11-19, 175 W per card, MEASURED; the README's speed tables and
[measurements.md](measurements.md) for the fourth-release bench).** Fourth-release bench (kit 9S, greedy, thinking off, 1,024 tokens out): code
prompts 1K to 256K tokens, one stream 153 to 184 tok/s decode, cold prefill 1,567 tok/s at 1K and 4,478 to 4,697 tok/s
from 32K to 256K; legal prompts one stream 92 to 102 tok/s. Window x2 (kit 9X, the serving set these benchmarks ran
on), `slots_cells.py`, two passes: one request 134.4 / 135.8 tok/s on fresh text and 183.0 / 185.8 on repeated text;
four requests together, summed, 308.6 / 318.5 fresh and 385.9 / 462.5 repeated; cold reads of 60,705 to 425,780
tokens at 4,181 to 4,468 tok/s.

**The same measure on 2026-11-22 (MEASURED, 175 W, the same tool and texts as window x2, 07:01Z to 07:09Z).** One request
18.1 / 19.1 tok/s fresh and 24.5 / 25.6 repeated; four together, summed, 51.0 / 53.1 fresh and 74.9 / 80.3 repeated;
no request starved. That is about one seventh of window x2. The server's own log shows the decode round of a lone
request at 26 to 30 ms on 2026-11-20 and 90 to 165 ms since the benchmarks ran; the clocks, the power cap, memory and
the drafter's acceptance (3.1 to 5.7 tokens per round) are normal. The cause is under investigation (the lead suspect
is the prompt store's conversation tier, which the benchmarks filled to its disk budget); the speed figures will be
updated when it is found. The benchmark answers above were all produced in this slower state; the scores are what the
model answered, not how fast.

**Under the benchmark load (MEASURED, the relay's log, 5,640 requests over 2026-11-20 to 2026-11-22, three in flight).**
Median output rate per request, prompt reading included, by set: MATH-500 29.0, GSM8K 24.1, HumanEval+ 25.5, MBPP+
23.0, MMLU 15.9, MMLU-Pro 14.2, LegalBench 10.1, LongBench-E 10.4 tok/s. These are short answers arriving every few
seconds three at a time, in the slower state above, and are not a speed figure for the build.
