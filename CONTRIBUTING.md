# Contributing

Bug reports, measurements on other V100 hosts, corrections to our numbers or credits, and pull requests are all welcome.

## Reporting a bug

Open an issue on [Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4](https://github.com/ProprietaryLegal/Volta-12X-AmPLIfy-DeepSeek-V4.1-Flash-1017-NVFP4-12xV100-SXM2-PP3xTP4/issues). Include the kit
(`patches/CURRENT` or the `MANIFEST` you applied), the per-stage reserves in your profile, the loader lines that report the
hot and cold expert counts and the card-memory lines after warm-up, the engine log lines around the failure, and your GPU, clock and power settings.
Please remove addresses, host names and anything from your prompts that you would not post publicly.

Problems in the underlying engine belong upstream: [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), or our
earlier build, [Volta sPLIt](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4).

## Reporting a measurement

Report ms per round and tok/s the way [docs/measurements.md](docs/measurements.md) describes (T=0, prompt sizes, two runs
in one boot), the GPU clock cap and power limit, and what else ran on the machine. Token counts and timings only, never
prompt or generated text, and never a routing tally or per-expert list from your own traffic.

A number goes into the README or `docs/` only if it is MEASURED, with its date, configuration and conditions, or labelled
ESTIMATED with the arithmetic shown.

## Pull requests

- One change per pull request, with the reason in the description.
- A change that affects serving needs a before-and-after on V100 hardware (the accepted long prompts token for token, or
  closeness to the reference where the change alters rounding), or a clear statement that it has none yet; we can run it on ours.
- Keep the engine's licence and headers on any file you modify; add new sources to [CREDITS.md](CREDITS.md) and [NOTICE](NOTICE).
- Do not add prompts, outputs, logs, tallies or caches from real use.

By contributing you agree that your contribution is licensed under Apache-2.0, like the rest of the original code.
