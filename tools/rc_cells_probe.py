#!/usr/bin/env python3
"""rc_cells_probe.py URL WIN_CELLS.py OUT.jsonl: the copy task RC sent the way the owner's cells send it, with the
reasoning and the answer counted.

tools/rc_probe.py (window o1, on the serving default) got an EMPTY answer at 1,500 tokens (5,301 characters of
reasoning) where the cells score the same default 0.9808 at the same 1,500 tokens. The two requests have the same
body; what differs is what comes before: a cell first reads its prompt once with the same request and max_tokens 1,
and only then sends the request. So the probe did not ask the cells' question. This one does, and shows the
difference on the server it is pointed at:
  cells-1500   read with max_tokens 1, then the request with max_tokens 1,500   (exactly a cell)
  cells-4000   read with max_tokens 1, then the request with max_tokens 4,000   (a cell with room to finish)
  cold-1500    the request with max_tokens 1,500 and a fresh cache salt         (no read before it, nothing kept)
  cold-4000    the request with max_tokens 4,000 and a fresh cache salt         (the cold case with room to finish)
Per request: tokens out, finish reason, characters of reasoning, characters and words of answer, the owner's copy
score. Counts only; no text is written. Exits: 0 done, 3 server busy, 4 a request failed, 9 bad arguments."""
import importlib.util
import json
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rc_probe", HERE / "rc_probe.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


def ask(url, model, prompt, max_tokens, salt=None):
    import time
    import urllib.request

    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if salt is not None:
        body["cache_salt"] = salt
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
    import urllib.request

    if len(argv) != 3:
        print(__doc__)
        return 9
    url, cells_path, out = argv
    spec = importlib.util.spec_from_file_location("win_cells", cells_path)
    wc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wc)
    if wc.metric_sum(url, ["num_requests_running", "num_requests_waiting"]) != 0:
        print("rc_cells_probe: the server has a request running or waiting; stopping", file=sys.stderr)
        return 3
    prompt, expected = wc.prompts()["RC"], wc.expected_texts()["RC"]
    model = json.load(urllib.request.urlopen(url + "/v1/models", timeout=10))["data"][0]["id"]
    rows = []
    for name, max_tokens, read_first in (
        ("cells-1500", 1500, True), ("cells-4000", 4000, True), ("cold-1500", 1500, False), ("cold-4000", 4000, False)
    ):
        try:
            if read_first:
                ask(url, model, prompt, 1)  # the cell's read: the same request, one token
            usage, finish, reasoning, content, seconds = ask(
                url, model, prompt, max_tokens, salt=None if read_first else uuid.uuid4().hex[:12])
        except Exception as e:
            print(f"rc_cells_probe: request failed at {name}: {e!r}", file=sys.stderr)
            return 4
        score = wc.copy_score(expected, content, finish == "length")
        row = {"name": name, "max_tokens": max_tokens, "prompt": usage["prompt_tokens"],
               "out": usage["completion_tokens"], "finish": finish, "reasoning_chars": len(reasoning),
               "answer_chars": len(content), "answer_words": len(content.split()),
               "expected_words": len(expected.split()), "copy_score": round(score, 6), "seconds": round(seconds, 2)}
        rows.append(row)
        print(f"RC {name}: out {row['out']} tok, finish {finish}, reasoning {row['reasoning_chars']} characters, "
              f"answer {row['answer_chars']} characters ({row['answer_words']} words of {row['expected_words']} "
              f"expected), copy score {score:.4f}", flush=True)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
