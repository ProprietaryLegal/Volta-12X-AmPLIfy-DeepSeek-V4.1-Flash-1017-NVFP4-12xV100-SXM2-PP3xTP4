# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_INDEX_TC: the decode index scores from the tensor-core scorer (v41idx3), through the engine's wrapper.

On CPU tensors, with both libraries replaced by their test libraries: v41idx_cpu_ops.so runs v41idx.cu's own host
bodies (the scorer that serves); v41idx3_cpu_ops.so runs v41idx3.cu's own bodies with each mma replaced by the
card's measured layout and adder (the decode-kernels work's kernels/v41idx3/: its own host tests, and
the one-card run of 2026-11-18, are where the kernel is proved; here it is the WRAPPER).

- on: `index_tiled.scores` writes, bit for bit, what the serving op writes for the same operands (columns past the
  end included), for operands the guard passes and operands it does not;
- its lines: the library at import, "active" once, the counters with the values arithmetic gives, only when a
  counter moved, not sooner than the interval; nothing is registered for the interpreter's shutdown;
- the prompt branch of select_packed_positions reads the counters first, and does not when the switch is off;
- off: the serving op is called, nothing of v41idx3 is loaded or printed;
- what refuses.

Environment (the test runner sets them):
  V41IDX_CPU_OPS_LIB    v41idx_cpu_ops.so
  V41IDX3_CPU_OPS_LIB   v41idx3_cpu_ops.so
"""

from __future__ import annotations

import importlib
import logging
import os

import pytest
import torch

pytestmark = pytest.mark.cpu_test

REF_LIB = os.environ["V41IDX_CPU_OPS_LIB"]
TC_LIB = os.environ["V41IDX3_CPU_OPS_LIB"]
LOG = "vllm.models.deepseek_v41.index_tiled"
KNOB = "VLLM_V41_INDEX_TC"
POISON = 0x7E55
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def halves(n: int, gen: torch.Generator) -> torch.Tensor:
    bits = torch.randint(0, 1 << 16, (n,), generator=gen, dtype=torch.int32)
    bits = (bits & 0x83FF) | (torch.randint(11, 18, (n,), generator=gen, dtype=torch.int32) << 10)
    return bits.to(torch.int16).view(torch.float16)


def fp4_rows(rows: int, low: int, high: int, gen: torch.Generator) -> torch.Tensor:
    """What quant_rope._qdq (fp4, group 32) leaves: E2M1 values times 2^e per 32 dimensions, e in low..high."""
    codes = torch.randint(0, 16, (rows, 128), generator=gen)
    mag = E2M1[codes & 7]
    value = torch.where((codes & 8) != 0, -mag, mag)
    exponent = torch.randint(low, high + 1, (rows, 4), generator=gen)
    return (value * torch.pow(2.0, exponent.float()).repeat_interleave(32, dim=1)).to(torch.float16).contiguous()


def operands(kind: str, tokens: int, capacity: int, seed: int):
    """close: the engine's class with close scales (the guard passes every cell: a row's A is at most 5, a key's B
    at most 2); plain: fp16 rows of any mantissa (the guard passes few or none)."""
    gen = torch.Generator().manual_seed(seed)
    keys = torch.randint(0, 256, (capacity, 68), generator=gen, dtype=torch.int32)
    if kind == "close":
        q = fp4_rows(tokens * 8, -3, -1, gen)
        keys[:, 64:] = torch.randint(126, 129, (capacity, 4), generator=gen, dtype=torch.int32)
    else:
        q = halves(tokens * 8 * 128, gen).view(tokens * 8, 128).contiguous()
        keys[:, 64:] = torch.randint(121, 132, (capacity, 4), generator=gen, dtype=torch.int32)
    return q, halves(tokens * 8, gen), keys.to(torch.uint8)


def wrapper(monkeypatch, tc: str | None, stats: str | None = "0", tc_lib: str | None = TC_LIB):
    monkeypatch.setenv("VLLM_V41_INDEX_TILED", "1")
    monkeypatch.setenv("VLLM_V41_INDEX_TILED_LIB", REF_LIB)
    for name, value in ((KNOB, tc), ("VLLM_V41_INDEX_TC_STATS", stats), ("VLLM_V41_INDEX_TC_LIB", tc_lib)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    from vllm.models.deepseek_v41 import index_tiled

    return importlib.reload(index_tiled)


def poisoned(tokens: int, n_out: int) -> torch.Tensor:
    return torch.full((tokens, n_out), POISON, dtype=torch.int16).view(torch.float16)


def through(module, q, w, keys, end: int, ratio: int = 1) -> torch.Tensor:
    tokens = q.shape[0] // 8
    out = poisoned(tokens, keys.shape[0] + 3)
    module.scores(out, q, w, keys, torch.full((tokens,), end * ratio - 1, dtype=torch.int64), 8, ratio)
    return out


def serving(q, w, keys, end: int, ratio: int = 1) -> torch.Tensor:
    tokens = q.shape[0] // 8
    out = poisoned(tokens, keys.shape[0] + 3)
    torch.ops.v41idx.index_scores_tiled(out, q, w, keys, torch.full((tokens,), end * ratio - 1, dtype=torch.int64),
                                        8, ratio)
    return out


def lines(caplog, part: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if part in r.getMessage()]


@pytest.mark.parametrize("kind", ["close", "plain"])
@pytest.mark.parametrize("tokens", [1, 6, 8])
def test_on_the_wrapper_writes_the_serving_scorer_s_bits(monkeypatch, kind, tokens):
    module = wrapper(monkeypatch, "1")
    q, w, keys = operands(kind, tokens, 4096, 10 * tokens + (kind == "plain"))
    for ratio in (1, 4):
        for end in (0, 1, 63, 64, 65, 1000 // ratio, 4096 // ratio):
            got = through(module, q, w, keys, end, ratio)
            want = serving(q, w, keys, end, ratio)
            assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (kind, tokens, ratio, end)
            assert bool((want.view(torch.int16)[:, end:] == POISON).all())


def test_the_lines_and_the_counters(monkeypatch, caplog):
    """Two calls of 6 tokens over 3,000 live keys of the close class: 2 calls, 36,000 cells, nothing over the
    limit. Then one call with 5 keys given scale bytes 114, 127, 127, 140 (in range, far over the limit whatever
    the row): 5 x 6 = 30 cells over, each with its eight heads recomputed: 240."""
    with caplog.at_level(logging.INFO, logger=LOG):
        module = wrapper(monkeypatch, "1")
        q, w, keys = operands("close", 6, 4096, 3)
        through(module, q, w, keys, 3000)
        through(module, q, w, keys, 3000)
        module.tc_report()
        module.tc_report()  # nothing moved: no second line
        far = keys.clone()
        far[100:105, 64:] = torch.tensor([114, 127, 127, 140], dtype=torch.uint8)
        got = through(module, q, w, far, 3000)
        module.tc_report()
    assert torch.equal(got.view(torch.int16), serving(q, w, far, 3000).view(torch.int16))
    assert lines(caplog, "v41idx3 library loaded")[-1] == f"V4.1 indexer: v41idx3 library loaded from {TC_LIB}"
    assert lines(caplog, "decode scorer active") == [
        "V4.1 indexer: v41idx3 tensor-core decode scorer active (heads 8); its counters are read at eager indexer "
        "calls, at most every 0 s."
    ]
    assert lines(caplog, "v41idx3 counters") == [
        # the second call's own reading (the interval is 0): what the first call counted
        "V4.1 indexer: v41idx3 counters: 1 calls, 18000 cells, 0 cells over the first look's limit, 0 heads "
        "recomputed by the serving chain.",
        "V4.1 indexer: v41idx3 counters: 2 calls, 36000 cells, 0 cells over the first look's limit, 0 heads "
        "recomputed by the serving chain.",
        "V4.1 indexer: v41idx3 counters: 3 calls, 54000 cells, 30 cells over the first look's limit, 240 heads "
        "recomputed by the serving chain.",
    ]


def test_nothing_is_left_for_the_interpreter_s_shutdown(monkeypatch):
    """The stop of a worker must not read card memory: the module registers nothing with `atexit` and has no exit
    line (the counts of the last decode steps show at the next request's first prompt step)."""
    import atexit

    registered = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: registered.append(fn))
    module = wrapper(monkeypatch, "1")
    q, w, keys = operands("close", 6, 1024, 8)
    through(module, q, w, keys, 1000)
    module.tc_report()
    assert registered == [] and not hasattr(module, "atexit") and not hasattr(module, "_tc_exit")


def test_the_counters_are_not_read_sooner_than_the_interval(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger=LOG):
        module = wrapper(monkeypatch, "1", stats="30")
        clock = [1000.0]
        monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
        q, w, keys = operands("close", 6, 1024, 4)
        through(module, q, w, keys, 1000)
        clock[0] += 29.0
        through(module, q, w, keys, 1000)
        module.tc_report()
        assert lines(caplog, "v41idx3 counters") == []
        clock[0] += 2.0
        module.tc_report()
    assert lines(caplog, "v41idx3 counters") == [
        "V4.1 indexer: v41idx3 counters: 2 calls, 12000 cells, 0 cells over the first look's limit, 0 heads "
        "recomputed by the serving chain."
    ]
    assert "at most every 30 s" in lines(caplog, "decode scorer active")[0]


class Reached(Exception):
    pass


def test_the_prompt_branch_reads_the_counters_first(monkeypatch):
    """attention_ops.select_packed_positions with more than 8 rows: on, its first act is index_tiled.tc_report();
    off, it is not called (the branch then goes on to its Triton kernels, which no CPU has: any error but ours)."""
    module = wrapper(monkeypatch, "1")
    from vllm.models.deepseek_v41 import attention_ops

    attention_ops = importlib.reload(attention_ops)
    assert attention_ops.index_tiled is module
    calls = []

    def recorder(*args):
        calls.append(args)
        raise Reached

    monkeypatch.setattr(module, "tc_report", recorder)
    q = torch.zeros(20, 8, 128, dtype=torch.float16)
    weights = torch.zeros(20, 8, dtype=torch.float16)
    keys = operands("close", 1, 256, 5)[2]
    positions = torch.arange(100, 120, dtype=torch.int32)
    valid = torch.ones(20, dtype=torch.bool)
    with pytest.raises(Reached):
        attention_ops.select_packed_positions(q, weights, keys, positions, valid, 1, None)
    assert calls == [()]
    monkeypatch.setattr(module, "_TC", False)
    calls.clear()
    with pytest.raises(Exception) as off:
        attention_ops.select_packed_positions(q, weights, keys, positions, valid, 1, None)
    assert off.type is not Reached and calls == []


def test_off_the_serving_op_is_called_and_nothing_of_v41idx3_is_touched(monkeypatch, caplog, tmp_path):
    """Unset, with the library's path pointing at nothing: the import does not look for it; the wrapper's output is
    the serving op's; its line is the serving scorer's; no v41idx3 line."""
    with caplog.at_level(logging.INFO, logger=LOG):
        module = wrapper(monkeypatch, None, tc_lib=str(tmp_path / "absent.so"))
        assert module._TC is False and module._TC_STATE["counters"] is None
        q, w, keys = operands("plain", 6, 2048, 6)
        got = through(module, q, w, keys, 2000)
        module.tc_report()
    assert torch.equal(got.view(torch.int16), serving(q, w, keys, 2000).view(torch.int16))
    assert lines(caplog, "v41idx3") == []
    assert lines(caplog, "decode scorer active") == ["V4.1 indexer: v41idx tiled decode scorer active (heads 8)"]


def test_what_refuses(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="VLLM_V41_INDEX_TC must be 0 or 1"):
        wrapper(monkeypatch, "2")
    with pytest.raises(OSError):
        wrapper(monkeypatch, "1", tc_lib=str(tmp_path / "absent.so"))
    with pytest.raises(ValueError, match="VLLM_V41_INDEX_TC_STATS"):
        wrapper(monkeypatch, "1", stats="soon")
    module = wrapper(monkeypatch, "1")
    q, w, keys = operands("close", 2, 256, 7)
    with pytest.raises(ValueError, match="eight index heads per rank"):
        module.scores(poisoned(2, 259), q[:8], w[:8], keys, torch.full((2,), 99, dtype=torch.int64), 4, 1)


def test_the_switch_without_the_tiled_scorer_stops_the_import(monkeypatch):
    """attention_ops imports index_tiled only with VLLM_V41_INDEX_TILED on; the switch of a scorer that is off
    would do nothing, so the import refuses."""
    from vllm.models.deepseek_v41 import attention_ops

    monkeypatch.delenv("VLLM_V41_INDEX_TILED", raising=False)
    monkeypatch.setenv(KNOB, "1")
    with pytest.raises(ValueError, match="VLLM_V41_INDEX_TC needs VLLM_V41_INDEX_TILED=1"):
        importlib.reload(attention_ops)
    monkeypatch.delenv(KNOB, raising=False)
    importlib.reload(attention_ops)  # and unset it imports as it did
