# PP3KIT9W-over-cut

Kit 9S plus the cached-token count in the API server's usage (2026-11-19). Not served alone: it is the base of kit 9X.

**What it changes.** A prefix-cache hit already reached the engine's output; the count was dropped in two serving files.
The OpenAI chat route's streamed first chunk built its usage without the detail and sent `cached_tokens` only when it
was not zero, and the Anthropic route copied the whole prompt into `input_tokens`. Kit 9W changes those two files only;
the scheduler is kit 9S's byte for byte. With the profile line `enable-prompt-tokens-details: true` every reply carries
the field: on `/v1/chat/completions`, `prompt_tokens` is the whole prompt and `prompt_tokens_details.cached_tokens` the
cached part; on `/v1/messages`, `input_tokens` is the uncached part and `cache_read_input_tokens` the cached part, the
two summing to the rendered prompt, with an explicit 0 on a cold read.

**Files.** `MANIFEST` has 20 rows over the cut (kit 9S's 18 files plus the two serving files). Every row's file is here.
`tests/v41_usage/test_cached_tokens.py` runs both routes on CPU; on kit 9S its Anthropic and OpenAI cases fail, on 9W
they pass. `apply.sh` and `rollback.sh` are kit 9V's.

**Published copy.** Comment lines naming private repositories were reworded in some kit files and their sha256
recomputed in `MANIFEST` (code unchanged); the MANIFEST's own header line says how many.
