#!/usr/bin/env python3
"""One-card, no-model probe: does layer 0's wkv projection give different bytes for the
same rows when the prompt chunk has another row count (TurboMind split-K chosen from M)?

Run on the server, ONE visible V100, nothing else needed (no server, no NCCL, no TP group):

    CUDA_VISIBLE_DEVICES=<one index> \
      $B12_ROOT/engine/venv/bin/python -I gemm_split_probe.py

The parent process starts each setting in a FRESH SUBPROCESS, because the native policy
(TM_GEMM_TRACE*, TM_GEMM_TUNE) is read from the environment once per process, at the
first native call (csrc/sm70_policy.h: legacy_policy(), a function-local static):
  default : the served dispatch (no TM_GEMM_TUNE)
  fix     : TM_GEMM_TUNE=max_splits=1 (TM gemm.cu:547-551 -> tuner/params.cc:22)
Both children get TM_GEMM_TRACE=1, TM_GEMM_TRACE_FILTER=x512x5120_ and
TM_GEMM_TRACE_LIMIT=8. The engine then prints one "[TM_GEMM_TRACE] desc=... splits=N ...
cta=AxBxC" line on stderr per traced dispatch (TM gemm.cu:490-515); the parent captures
the child's stderr and parses those lines. The limit of 8 covers exactly the first call
for each of the 8 row counts; the timing calls after them are not traced.

What a child runs, each step through the engine's own code (engine tree ENGINE_SRC):
  tokens    first 8,188 ids of the public file TEXT (default: the engine's own
            vllm/models/deepseek_v41/model.py), tokenizer.json of the served model, no
            special tokens. Only a count and a sha256 of the ids are printed.
  tensors   token_embd.weight (Q5_K), blk.0.attn_norm.weight (F32), blk.0.attn_kv.weight
            (Q8_0) and blk.0.attn_kv_a_norm.weight (F32) from the served GGUF shards, read
            with the engine's loader reader (vllm/transformers_utils/gguf_tensor_reader.py
            GGUFReader, as model_executor/model_loader/weight_utils.py:1532-1548 does).
            Names map to embed_tokens / attn_norm / attn.wkv / attn.kv_norm by
            model_executor/model_loader/gguf_adapters/deepseek_v41.py:21,30,35,36.
  embedding the raw Q5_K rows of the used ids, through torch.ops.vllm._apply_gguf_embedding
            (model_executor/layers/quantization/gguf.py:513-534, called by
            GGUFEmbeddingMethod.embedding gguf.py:1318-1330, native_enabled True). Per-row
            exact: a row's dequantization reads only that row, so a sub-table of the used
            rows gives the same bytes as the full table; the served TP4 all-reduce of the
            embedding adds three zeros to the owner's row (exact).
  collapse  hidden = embedding.float() expanded to four streams, pre = (1,0,0,0)
            (models/deepseek_v41/model.py:1164-1166), then collapse_hc with
            attn_norm.weight (model.py:1505 -> model_executor/kernels/mhc/deepseek_v41.py:420).
  wkv       prepare_gguf_projections([(raw Q8_0 bytes, 8)], float16, True, prefill_min_m=1)
            exactly as models/deepseek_v41/dense_tm.py:102-104 (_convert) calls it, after
            registering the (K, N) = (5120, 512) segment shape as dense_tm.py:392 does
            (gguf_dense_hmma.EXTRA_SEGMENT_SHAPES), under a vllm config holding a default
            KernelConfig (the engine's own test pattern,
            tests/kernels/quantization/test_gguf_segment_prefill.py:43-46). The call is
            apply_prepared_gguf_projections (as gguf.py:1077-1080 calls it): family 3,
            rows > 64 -> restore_and_apply -> _C.gguf_affine_gemm_sm70_out.
  window    V41Norm(512) kv_norm -> apply_rope(window table rope_table()) ->
            attention_qdq, the prompt-read lines of model.py:636-639, at positions
            23,468 + row (the twelve-card tail chunk starts at 3,008 + 5 x 4,092).

Output: counts, shapes, split counts, hashes and timings only. No token text, no ids.
Exit 0 whatever the verdict; non-zero only on an error (then the verdicts that depend on
the failed step are printed INCONCLUSIVE first).
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

ENGINE_SRC = os.environ.get("B12_ROOT", "/path/to/build") + "/engine/src"
GGUF_PATTERN = (
    os.environ.get("MODELS_ROOT", "/path/to/models") + "/gguf/"
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-{:05d}-of-00004.gguf"
)
GGUF_SHARDS = 4
TOKENIZER = os.environ.get("MODELS_ROOT", "/path/to/models") + "/base/tokenizer.json"
TEXT = ENGINE_SRC + "/vllm/models/deepseek_v41/model.py"

ROWS = 8188
# The first eight are the reviewer's set (subject pair, two control pairs, the full chunks). The lead added the tail
# rows of the store tests' present and candidate lengths: M12 at 22,736 (2,276 / 2,292), M1 at 22,704 (2,244 /
# 2,260) and at 22,688 as run (2,228 / 2,244), M3 at 22,656 (2,196 / 2,212), S2 at 130,000 (3,148 / 3,164).
MS = (2292, 2308, 6376, 6392, 3696, 3712, 4092, 8188, 2276, 2260, 2244, 2228, 2212, 2196, 3148, 3164)
K, N = 5120, 512
TAIL_START = 3008 + 5 * 4092  # 23,468: position of chunk row 0
BLOCK_ROWS = 16
BLOCK_LO, BLOCK_HI = 2148, 2292  # chunk rows of positions 25,616..25,759
BLOCK_FIRST_ID = (TAIL_START + BLOCK_LO) // BLOCK_ROWS  # 1601
CONTROL_PAIRS = ((6376, 6392), (3696, 3712))
SUBJECT_PAIR = (2292, 2308)
WARMUPS, TIMED = 2, 5
CARD_LIMIT_BYTES = 4 * 1024**3
SETTINGS = {
    "default": {},
    "fix": {"TM_GEMM_TUNE": "max_splits=1"},
}
TRACE_ENV = {
    "TM_GEMM_TRACE": "1",
    "TM_GEMM_TRACE_FILTER": "x{}x{}_".format(N, K),
    "TM_GEMM_TRACE_LIMIT": str(len(MS)),
}
TRACE_LINE = re.compile(
    r"\[TM_GEMM_TRACE\] desc=(?P<desc>\S*_(?P<m>\d+)x" + str(N) + "x" + str(K)
    + r"_\d+) policy=(?P<policy>\S+) measured=(?P<measured>\d)"
    + r" kernel=(?P<kernel>\S+) splits=(?P<splits>\d+) swizzle=\d+"
    + r" cta=(?P<cta>\d+x\d+x\d+)"
)
RESULT_PREFIX = "PROBE_RESULT "


def pairs():
    return [(a, b) for i, a in enumerate(MS) for b in MS[i + 1:]]


# --------------------------------------------------------------------------- child
def child(setting, args):
    sys.path.insert(0, args.engine_src)
    import numpy as np
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"exactly one visible card required, found {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise RuntimeError("SM70 (V100) required")
    torch.cuda.set_device(device)

    from types import SimpleNamespace

    from tokenizers import Tokenizer

    from vllm.config import set_current_vllm_config
    from vllm.config.kernel import KernelConfig
    from vllm.model_executor.kernels.mhc.deepseek_v41 import collapse_hc
    from vllm.model_executor.layers.quantization import gguf_dense_hmma
    from vllm.model_executor.layers.quantization.gguf import apply_gguf_embedding
    from vllm.model_executor.layers.quantization.gguf_turbomind import (
        _admitted_bands,
        apply_prepared_gguf_projections,
        prepare_gguf_projections,
    )
    from vllm.models.deepseek_v41.model import V41Norm
    from vllm.models.deepseek_v41.quant_rope import (
        apply_rope,
        attention_qdq,
        rope_table,
    )
    from vllm.transformers_utils.gguf_tensor_reader import GGUFReader

    result = {"setting": setting, "pid": os.getpid()}

    # Tokens of a public text.
    with open(args.text, encoding="utf-8") as handle:
        text = handle.read()
    ids = Tokenizer.from_file(args.tokenizer).encode(text, add_special_tokens=False).ids
    if len(ids) < ROWS:
        raise RuntimeError(f"{args.text} gives {len(ids)} tokens, {ROWS} needed")
    ids = np.asarray(ids[:ROWS], dtype=np.int64)
    result["tokens"] = {
        "source": args.text,
        "count": int(ids.size),
        "unique": int(np.unique(ids).size),
        "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
    }

    # Tensors, by name, from the served shards (last shard first; each name must occur once).
    wanted = {
        "token_embd.weight": 13,
        "blk.0.attn_norm.weight": 0,
        "blk.0.attn_kv.weight": 8,
        "blk.0.attn_kv_a_norm.weight": 0,
    }
    found = {}
    for index in range(GGUF_SHARDS, 0, -1):
        path = args.gguf.format(index)
        reader = GGUFReader(path)
        for tensor in reader.tensors:
            if tensor.name in wanted:
                if tensor.name in found:
                    raise RuntimeError(f"{tensor.name} occurs in two shards")
                if int(tensor.tensor_type) != wanted[tensor.name]:
                    raise RuntimeError(
                        f"{tensor.name} has type {int(tensor.tensor_type)}, "
                        f"expected {wanted[tensor.name]}"
                    )
                found[tensor.name] = (path, tensor)
        if len(found) == len(wanted):
            break
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise RuntimeError(f"tensors not found in the shards: {missing}")
    result["tensors"] = {
        name: {"shard": os.path.basename(path), "type": wanted[name],
               "data_shape": list(t.data.shape)}
        for name, (path, t) in found.items()
    }

    # Embedding rows through the engine's GGUF embedding op (per-row exact).
    embd = found["token_embd.weight"][1].data
    unique, remap = np.unique(ids, return_inverse=True)
    table = torch.from_numpy(np.ascontiguousarray(embd[unique])).to(device)
    remap_t = torch.from_numpy(remap.astype(np.int64)).to(device)
    embedding = apply_gguf_embedding(remap_t, table, 13, K, torch.float16, True)
    if embedding.shape != (ROWS, K) or embedding.dtype != torch.float16:
        raise RuntimeError(f"embedding {tuple(embedding.shape)} {embedding.dtype}")
    del table

    # collapse + attn_norm, as the first stage enters layer 0.
    attn_norm = torch.from_numpy(
        np.ascontiguousarray(found["blk.0.attn_norm.weight"][1].data, dtype=np.float32)
    ).to(device)
    hidden = embedding.float()[:, None].expand(-1, 4, -1).contiguous()
    pre = torch.zeros(ROWS, 4, device=device, dtype=torch.float32)
    pre[:, 0] = 1
    x = collapse_hc(hidden, pre, attn_norm)
    del hidden, pre, embedding
    if x.shape != (ROWS, K) or x.dtype != torch.float16:
        raise RuntimeError(f"layer-0 input {tuple(x.shape)} {x.dtype}")
    result["x_sha256"] = hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()

    # wkv through the engine's own preparation (dense_tm._convert's call).
    raw_kv = torch.from_numpy(
        np.ascontiguousarray(found["blk.0.attn_kv.weight"][1].data)
    ).to(device)
    if raw_kv.dtype != torch.uint8 or raw_kv.shape[0] != N:
        raise RuntimeError(f"attn_kv raw {tuple(raw_kv.shape)} {raw_kv.dtype}")
    gguf_dense_hmma.EXTRA_SEGMENT_SHAPES.add((K, N))
    config = SimpleNamespace(
        kernel_config=KernelConfig(),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    with set_current_vllm_config(config):
        projections = prepare_gguf_projections([(raw_kv, 8)], torch.float16, True, 1)
    if len(projections) != 1:
        raise RuntimeError(f"{len(projections)} projections prepared, 1 expected")
    projection = projections[0]
    if projection.rejection_reason is not None:
        raise RuntimeError(f"wkv preparation rejected: {projection.rejection_reason}")
    if getattr(projection, "segment_format", None) != 4:
        raise RuntimeError(
            "wkv is not the served Q8 segment projection (family 3, format 4): "
            f"segment_format={getattr(projection, 'segment_format', None)}"
        )
    blas = tuple(
        c for c in getattr(projection.kernel, "operator_capabilities", ())
        if "blas" in c.operator
    )
    if _admitted_bands(blas):
        raise RuntimeError(
            "the BLAS band is admitted here but was refused on the server "
            "(boot log: local_shape_has_no_prefill_calibration); the probe would "
            "not take the served path"
        )
    result["projection"] = {
        "family": 3,
        "segment_format": 4,
        "blas_admitted": False,
        "partition_weight_shape": list(projection.kernel.config.partition_weight_shape),
    }

    kv_norm = V41Norm(N).to(device)
    kv_norm.weight.data.copy_(
        torch.from_numpy(
            np.ascontiguousarray(
                found["blk.0.attn_kv_a_norm.weight"][1].data, dtype=np.float32
            )
        )
    )
    table = rope_table(device=device)

    def wkv(m):
        return apply_prepared_gguf_projections(x[:m], projections)

    def window(kv_low, m):
        positions = torch.arange(
            TAIL_START, TAIL_START + m, device=device, dtype=torch.int64
        )
        return attention_qdq(apply_rope(kv_norm(kv_low), positions, table))

    # One call per M first (these are the 8 traced dispatches), then the window chain.
    kv_low, win = {}, {}
    for m in MS:
        out = wkv(m)
        torch.cuda.synchronize()
        if out.shape != (m, N) or out.dtype != torch.float16:
            raise RuntimeError(f"wkv({m}) gave {tuple(out.shape)} {out.dtype}")
        kv_low[m] = out
    for m in MS:
        win[m] = window(kv_low[m], m)
    torch.cuda.synchronize()

    def compare(a, b):
        n = min(a.shape[0], b.shape[0])
        diff = a[:n].view(torch.int16) != b[:n].view(torch.int16)
        return {
            "rows": n,
            "elements": n * a.shape[1],
            "differing_elements": int(diff.sum()),
            "differing_rows": int(diff.any(dim=1).sum()),
        }

    result["pairs"] = {
        "kv_low": {f"{a}/{b}": compare(kv_low[a], kv_low[b]) for a, b in pairs()},
        "window": {f"{a}/{b}": compare(win[a], win[b]) for a, b in pairs()},
    }

    def block_hashes(t):
        return [
            hashlib.sha256(
                t[r:r + BLOCK_ROWS].contiguous().cpu().numpy().tobytes()
            ).hexdigest()[:12]
            for r in range(BLOCK_LO, BLOCK_HI, BLOCK_ROWS)
        ]

    a, b = SUBJECT_PAIR
    result["blocks"] = {
        "first_block_id": BLOCK_FIRST_ID,
        "chunk_rows": [BLOCK_LO, BLOCK_HI - 1],
        "kv_low": {str(a): block_hashes(kv_low[a]), str(b): block_hashes(kv_low[b])},
        "window": {str(a): block_hashes(win[a]), str(b): block_hashes(win[b])},
    }

    # Timing of the served wkv call (weight restore + GEMM), per M.
    timing = {}
    for m in MS:
        for _ in range(WARMUPS):
            wkv(m)
        samples = []
        for _ in range(TIMED):
            torch.cuda.synchronize()
            start = time.perf_counter()
            wkv(m)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1e3)
        timing[str(m)] = {"median_ms": statistics.median(samples), "samples_ms": samples}
    result["timing"] = timing
    result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    if result["peak_allocated_bytes"] > CARD_LIMIT_BYTES:
        raise RuntimeError(
            f"peak allocation {result['peak_allocated_bytes']} B exceeds 4 GiB"
        )
    print(RESULT_PREFIX + json.dumps(result), flush=True)


# --------------------------------------------------------------------------- parent
def run_setting(setting, args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TM_GEMM_")}
    env.update(TRACE_ENV)
    env.update(SETTINGS[setting])
    command = [
        sys.executable, "-I", os.path.abspath(__file__), "--child", setting,
        "--engine-src", args.engine_src, "--gguf", args.gguf,
        "--tokenizer", args.tokenizer, "--text", args.text,
    ]
    start = time.perf_counter()
    done = subprocess.run(command, env=env, capture_output=True, text=True)
    seconds = time.perf_counter() - start
    results = [
        line[len(RESULT_PREFIX):]
        for line in done.stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    traces = {}
    for line in done.stderr.splitlines():
        match = TRACE_LINE.search(line)
        if match:
            m = int(match["m"])
            if m in traces:
                raise RuntimeError(f"{setting}: two traced dispatches for M={m}")
            traces[m] = {
                "splits": int(match["splits"]),
                "cta": match["cta"],
                "kernel": match["kernel"],
                "policy": match["policy"],
                "measured": int(match["measured"]),
            }
    if done.returncode != 0 or len(results) != 1:
        tail = "\n".join(done.stderr.splitlines()[-25:])
        return None, traces, seconds, (
            f"child exit {done.returncode}, {len(results)} result lines; stderr tail:\n{tail}"
        )
    return json.loads(results[0]), traces, seconds, None


def report_setting(setting, result, traces, seconds):
    print(f"\n=== setting {setting} (fresh subprocess pid {result['pid']}, "
          f"{seconds:.1f} s, peak allocated {result['peak_allocated_bytes'] / 2**20:.0f} MiB)")
    if setting == "default":
        t = result["tokens"]
        print(f"tokens: {t['count']} ids ({t['unique']} distinct) of {t['source']}, "
              f"ids sha256 {t['ids_sha256'][:16]}; layer-0 input sha256 {result['x_sha256'][:16]}")
        for name, info in result["tensors"].items():
            print(f"tensor {name}: {info['shard']} type {info['type']} raw shape {info['data_shape']}")
        p = result["projection"]
        print(f"wkv: family {p['family']}, segment format {p['segment_format']}, "
              f"BLAS admitted {p['blas_admitted']}, weight shape {p['partition_weight_shape']}")
    print(f"{'M':>6} {'splits':>6} {'cta':>12} {'policy':>8} {'ms (median of 5)':>17}")
    for m in MS:
        tr = traces.get(m)
        splits = str(tr["splits"]) if tr else "NOT TRACED"
        cta = tr["cta"] if tr else "-"
        policy = tr["policy"] if tr else "-"
        print(f"{m:>6} {splits:>6} {cta:>12} {policy:>8} "
              f"{result['timing'][str(m)]['median_ms']:>17.3f}")
    print("pairs (common leading rows, bit compare): kv_low = wkv output, window = after "
          "kv_norm/RoPE/FP8 QDQ")
    for key in result["pairs"]["kv_low"]:
        k, w = result["pairs"]["kv_low"][key], result["pairs"]["window"][key]
        print(f"  {key:>10}: rows {k['rows']:>5}  kv_low diff elements {k['differing_elements']:>7} "
              f"rows {k['differing_rows']:>5} | window diff elements {w['differing_elements']:>6} "
              f"rows {w['differing_rows']:>5}")
    blocks = result["blocks"]
    a, b = (str(v) for v in SUBJECT_PAIR)
    for stage in ("kv_low", "window"):
        ha, hb = blocks[stage][a], blocks[stage][b]
        differ = [blocks["first_block_id"] + i for i, (x, y) in enumerate(zip(ha, hb)) if x != y]
        print(f"blocks {blocks['first_block_id']}-{blocks['first_block_id'] + len(ha) - 1} "
              f"(chunk rows {blocks['chunk_rows'][0]}-{blocks['chunk_rows'][1]}), {stage}, "
              f"{a} vs {b}: differ {differ if differ else 'none'}")
        print(f"    {a}: {' '.join(ha)}")
        print(f"    {b}: {' '.join(hb)}")


def equal(result, a, b, stage="kv_low"):
    return result["pairs"][stage][f"{a}/{b}"]["differing_elements"] == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--child", choices=sorted(SETTINGS), help=argparse.SUPPRESS)
    parser.add_argument("--engine-src", default=ENGINE_SRC)
    parser.add_argument("--gguf", default=GGUF_PATTERN,
                        help="shard path with one {:05d} field for the shard number")
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--text", default=TEXT, help="public text to tokenize")
    args = parser.parse_args()
    if args.child:
        child(args.child, args)
        return 0

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise SystemExit(
            f"set CUDA_VISIBLE_DEVICES to exactly one card index (now {visible!r})"
        )
    print("gemm_split_probe: wkv (blk.0.attn_kv, Q8_0, K 5120, N 512), M = "
          + ", ".join(str(m) for m in MS))
    print("each setting runs in a fresh subprocess (native policy is read once per process)")

    outcomes, failure = {}, False
    for setting in ("default", "fix"):
        result, traces, seconds, error = run_setting(setting, args)
        if error:
            failure = True
            print(f"\n=== setting {setting}: FAILED after {seconds:.1f} s\n{error}")
            outcomes[setting] = None
            continue
        report_setting(setting, result, traces, seconds)
        outcomes[setting] = (result, traces)

    print()
    default = outcomes["default"]
    if default is None:
        print("SPLIT DEPENDENCE: INCONCLUSIVE (the default setting did not run)")
    else:
        result, traces = default
        subject = not equal(result, *SUBJECT_PAIR)
        controls = all(equal(result, a, b) for a, b in CONTROL_PAIRS)
        if not subject:
            print("SPLIT DEPENDENCE: REFUTED (2292 vs 2308 bit-identical under the default dispatch)")
        elif controls:
            print("SPLIT DEPENDENCE: CONFIRMED (2292 vs 2308 differ; 6376 vs 6392 and "
                  "3696 vs 3712 bit-identical)")
        else:
            bad = [f"{a}/{b}" for a, b in CONTROL_PAIRS if not equal(result, a, b)]
            print(f"SPLIT DEPENDENCE: INCONCLUSIVE (2292 vs 2308 differ, but control "
                  f"pair(s) {bad} differ too)")
        if all(m in traces for m in MS):
            s = {m: traces[m]["splits"] for m in MS}
            print("split counts traced: " + ", ".join(f"{m}:{s[m]}" for m in MS)
                  + f"; 2292 vs 2308 splits {'differ' if s[2292] != s[2308] else 'EQUAL'}")
        else:
            print("split counts: NOT TRACED for "
                  + ", ".join(str(m) for m in MS if m not in traces)
                  + " (attribution to the split count not shown)")

    fix = outcomes["fix"]
    if fix is None:
        print("FIX max_splits=1: INCONCLUSIVE (the fix setting did not run)")
    else:
        result, traces = fix
        untraced = [m for m in MS if m not in traces]
        not_one = [m for m in MS if m in traces and traces[m]["splits"] != 1]
        if untraced or not_one:
            print(f"FIX max_splits=1: INCONCLUSIVE (setting not shown in effect: untraced "
                  f"{untraced}, splits != 1 at {not_one})")
        else:
            differing = [key for key, v in result["pairs"]["kv_low"].items()
                         if v["differing_elements"]]
            win_differing = [key for key, v in result["pairs"]["window"].items()
                             if v["differing_elements"]]
            if differing or win_differing:
                print(f"FIX max_splits=1: NOT ALL EQUAL (kv_low pairs {differing}, "
                      f"window pairs {win_differing})")
            else:
                print("FIX max_splits=1: ALL EQUAL (all 28 pairs bit-identical, kv_low and "
                      "window; splits 1 at every M)")
    if default is not None and fix is not None:
        print("median ms per wkv call (restore + GEMM, as served), default -> fix: "
              + ", ".join(
                  f"{m}: {default[0]['timing'][str(m)]['median_ms']:.3f} -> "
                  f"{fix[0]['timing'][str(m)]['median_ms']:.3f}" for m in MS))
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
