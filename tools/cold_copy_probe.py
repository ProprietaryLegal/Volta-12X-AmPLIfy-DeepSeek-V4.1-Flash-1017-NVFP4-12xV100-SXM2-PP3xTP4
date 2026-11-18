#!/usr/bin/env python3
"""cold_copy_probe.py URL WIN_CELLS.py OUT.jsonl: every copy task of the owner's cells that has an expected text (RA,
RB, RC), each sent ONCE, COLD (fresh cache salt, nothing read before it), with max_tokens 4,000, one at a time.
REPORT ONLY (window o2): finished or cut by the limit, tokens out, characters of reasoning, characters and words of
answer, the owner's copy score. One request each at temperature 0: an observation, not a rate. Counts only; no text
is written. Exits: 0 done, 3 server busy, 4 a request failed, 9 bad arguments."""
import importlib.util
import json
import sys
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rc_cells_probe", HERE / "rc_cells_probe.py")
rcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rcp)
MAX_TOKENS = 4000


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 9
    url, cells_path, out = argv
    spec = importlib.util.spec_from_file_location("win_cells", cells_path)
    wc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wc)
    if wc.metric_sum(url, ["num_requests_running", "num_requests_waiting"]) != 0:
        print("cold_copy_probe: the server has a request running or waiting; stopping", file=sys.stderr)
        return 3
    prompts, expected = wc.prompts(), wc.expected_texts()
    model = json.load(urllib.request.urlopen(url + "/v1/models", timeout=10))["data"][0]["id"]
    rows = []
    for name in sorted(expected):
        try:
            usage, finish, reasoning, content, seconds = rcp.ask(url, model, prompts[name], MAX_TOKENS, salt=uuid.uuid4().hex[:12])
        except Exception as e:
            print(f"cold_copy_probe: request failed at {name}: {e!r}", file=sys.stderr)
            return 4
        score = wc.copy_score(expected[name], content, finish == "length")
        row = {"name": name, "max_tokens": MAX_TOKENS, "prompt": usage["prompt_tokens"], "out": usage["completion_tokens"],
               "finish": finish, "reasoning_chars": len(reasoning), "answer_chars": len(content),
               "answer_words": len(content.split()), "expected_words": len(expected[name].split()),
               "copy_score": round(score, 6), "seconds": round(seconds, 2)}
        rows.append(row)
        print(f"COLD {name} max_tokens {MAX_TOKENS}: prompt {row['prompt']} tok, out {row['out']} tok, finish {finish}, "
              f"reasoning {row['reasoning_chars']} characters, answer {row['answer_chars']} characters "
              f"({row['answer_words']} words of {row['expected_words']} expected), copy score {score:.4f}", flush=True)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
