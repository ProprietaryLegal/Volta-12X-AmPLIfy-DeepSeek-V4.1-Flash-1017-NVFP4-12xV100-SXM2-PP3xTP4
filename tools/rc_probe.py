#!/usr/bin/env python3
"""rc_probe.py URL WIN_CELLS.py OUT.jsonl: is the copy task RC answered, or does the model spend its tokens thinking?

In windows w3 and w5 the all-NVFP4 arm scored copy 0.0000 on RC (return a 215-line Python file with one name
changed) in every cell, where the default scored 0.9808 and the default's mix on three stages 1.0000; every arm hit
the cell's 1,500-token limit. The cells keep tokens, finish reason and score, not how the tokens divide between the
reasoning and the answer, so they cannot tell "no answer yet" from "a wrong answer". This sends the owner's RC
prompt (from WIN_CELLS.py, its pinned public text) at max_tokens 1,500 (the cells' limit) and 4,000 and prints, per
request: tokens out, finish reason, characters of reasoning, characters of answer, and the owner's copy score of the
answer (win_cells.copy_score; an answer cut by the limit is scored against the expected text cut to its length).
Counts only, no text is written. Exits: 0 done, 3 server busy, 4 a request failed, 9 bad arguments."""
import importlib.util
import json
import sys
import time
import urllib.request


def ask(url, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    usage = finish = None
    reasoning, content = [], []
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("error"):
                raise RuntimeError(str(d["error"])[:200])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
                dl = c.get("delta", {})
                think = dl.get("reasoning_content") or dl.get("reasoning")
                if think:
                    reasoning.append(think)
                if dl.get("content"):
                    content.append(dl["content"])
    if usage is None or finish is None:
        raise RuntimeError("no usage block or no finish reason")
    return usage, finish, "".join(reasoning), "".join(content), time.time() - t0


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 9
    url, cells_path, out = argv
    spec = importlib.util.spec_from_file_location("win_cells", cells_path)
    wc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wc)
    if wc.metric_sum(url, ["num_requests_running", "num_requests_waiting"]) != 0:
        print("rc_probe: the server has a request running or waiting; stopping", file=sys.stderr)
        return 3
    prompt, expected = wc.prompts()["RC"], wc.expected_texts()["RC"]
    model = json.load(urllib.request.urlopen(url + "/v1/models", timeout=10))["data"][0]["id"]
    rows = []
    for max_tokens in (1500, 4000):
        try:
            usage, finish, reasoning, content, seconds = ask(url, model, prompt, max_tokens)
        except Exception as e:
            print(f"rc_probe: request failed at max_tokens {max_tokens}: {e!r}", file=sys.stderr)
            return 4
        score = wc.copy_score(expected, content, finish == "length")
        row = {"max_tokens": max_tokens, "prompt": usage["prompt_tokens"], "out": usage["completion_tokens"],
               "finish": finish, "reasoning_chars": len(reasoning), "answer_chars": len(content),
               "answer_words": len(content.split()), "expected_words": len(expected.split()),
               "copy_score": round(score, 6), "seconds": round(seconds, 2)}
        rows.append(row)
        print(f"RC max_tokens {max_tokens}: out {row['out']} tok, finish {finish}, reasoning {row['reasoning_chars']} "
              f"characters, answer {row['answer_chars']} characters ({row['answer_words']} words of "
              f"{row['expected_words']} expected), copy score {score:.4f}", flush=True)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
