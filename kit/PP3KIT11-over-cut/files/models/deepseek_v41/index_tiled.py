# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_INDEX_TILED: register-tiled decode index scorer (v41idx.so).

Replaces torch.ops.v41k.index_scores_decode (the decode branch of
attention_ops.select_packed_positions) with v41idx.index_scores_tiled:

  - a persistent grid (<= 2 CTAs per SM) over the live 64-key blocks only,
    instead of capacity / 32 CTAs that mostly read positions and exit
    (25.6 / 48.9 us per call at short context, PROF1103b / lc-prof L0);
  - each thread owns one token x all heads x 4 keys, so a shared-memory load
    feeds 8-16 FMAs instead of 0.5 (497 / 1061 us per call at 300K keys).

Numerics: BITWISE. Every (row, key) dot product is the reference's single
fmaf chain over d = 0..127 on the same values (keys rounded to fp16 first),
and the relu / fp16 weight product / head-sum epilogue is the reference's.

  VLLM_V41_INDEX_TILED=0  off (default): this module is never imported.
  VLLM_V41_INDEX_TILED=1  on: decode index scores come from v41idx.so.
  VLLM_V41_INDEX_TILED_LIB  the .so; default benchmarks/deepseek_v41/v41idx/
                        build/v41idx.so beside VLLM_V41K_LIB's v41k directory.

Fail fast: with the knob on, the library loads at import (a missing .so
raises OSError at boot) and a call the kernel does not take (heads other than
4 or 8, more than 64 rows) raises; nothing falls back silently. The knobs are
read with os.getenv so envs.py stays untouched.
"""

from __future__ import annotations

import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_KNOB = os.getenv("VLLM_V41_INDEX_TILED", "0")
if _KNOB not in ("0", "1"):
    raise ValueError(f"VLLM_V41_INDEX_TILED must be 0 or 1, got {_KNOB!r}")
HEADS = (4, 8)
MAX_ROWS = 64
_STATE: dict = {"active": False}


def library_path() -> str:
    explicit = os.getenv("VLLM_V41_INDEX_TILED_LIB")
    if explicit:
        return explicit
    v41k = os.getenv("VLLM_V41K_LIB")
    if v41k:
        # .../benchmarks/deepseek_v41/<v41k dir>/build/v41k.so -> .../deepseek_v41
        root = os.path.dirname(os.path.dirname(os.path.dirname(v41k)))
    else:
        root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "benchmarks",
            "deepseek_v41",
        )
    return os.path.join(root, "v41idx", "build", "v41idx.so")


def load() -> str:
    path = library_path()
    torch.ops.load_library(path)  # raises OSError when the .so is missing
    if not hasattr(torch.ops.v41idx, "index_scores_tiled"):
        raise RuntimeError(f"VLLM_V41_INDEX_TILED=1: {path} has no v41idx ops")
    logger.info("V4.1 indexer: v41idx library loaded from %s", path)
    return path


if _KNOB == "1":
    LIBRARY = load()


def scores(out, q, w, keys, positions, heads, ratio):
    """out[t, n] for n < (positions[T - 1] + 1) // ratio, bitwise as
    torch.ops.v41k.index_scores_decode(out, q, w, keys, positions, heads,
    ratio)."""
    if heads not in HEADS or out.shape[0] * heads > MAX_ROWS:
        raise ValueError(
            "VLLM_V41_INDEX_TILED=1: the tiled scorer takes 4 or 8 index heads "
            f"per rank and at most {MAX_ROWS} rows; got heads={heads}, "
            f"tokens={out.shape[0]}"
        )
    if _TC:
        _tc_scores(out, q, w, keys, positions, heads, ratio)
        return
    if not _STATE["active"]:
        _STATE["active"] = True
        logger.info(
            "V4.1 indexer: v41idx tiled decode scorer active (heads %d)", heads
        )
    torch.ops.v41idx.index_scores_tiled(out, q, w, keys, positions, heads, ratio)


# --- VLLM_V41_INDEX_TC: the decode index scores from the tensor cores (v41idx3.so), bitwise the scores above ---
#
#   VLLM_V41_INDEX_TC=0   off (default): nothing below runs; v41idx3.so is not looked for, not loaded.
#   VLLM_V41_INDEX_TC=1   on (needs VLLM_V41_INDEX_TILED=1 and eight index heads per rank): `scores` calls
#                         torch.ops.v41idx3.index_scores_tc. Its dot products are four-product tensor-core adds; a
#                         guard in the kernel passes a (token, key) cell only where neither that add nor the serving
#                         chain can round (then both are the real sum), and recomputes by the serving chain every
#                         head it does not pass. Proof, host tests and the one-card run (2,048 generated cases and a
#                         127,677-token read's own operands bitwise, 2026-11-18): the plan of
#                         the decode-kernels work, PLAN-INDEX-TC.md, and its kernels/v41idx3/.
#   VLLM_V41_INDEX_TC_LIB the .so; default v41idx3/build/v41idx3.so beside v41idx's folder.
#   VLLM_V41_INDEX_TC_STATS  seconds between two readings of the kernel's counters (default 30; 0: at every eager
#                         indexer call). The kernel counts in card memory: calls, cells written, cells its first
#                         look did not pass, heads recomputed by the serving chain. A decode step is a graph replay
#                         and runs no Python, so the counters are READ where Python runs: at an eager indexer call
#                         (a prompt step of any request; never inside a graph capture). One line a reading,
#                         only when a counter moved. Nothing is read when the worker stops: the counts of the last
#                         decode steps show at the next request's first prompt step.
_TC_KNOB = "VLLM_V41_INDEX_TC"
_TC_VALUE = os.getenv(_TC_KNOB, "0")
if _TC_VALUE not in ("0", "1"):
    raise ValueError(f"{_TC_KNOB} must be 0 or 1, got {_TC_VALUE!r}")
_TC = _TC_VALUE == "1"
_TC_STATE: dict = {"counters": None, "read_at": 0.0, "last": None}


def tc_library_path() -> str:
    explicit = os.getenv("VLLM_V41_INDEX_TC_LIB")
    if explicit:
        return explicit
    # .../deepseek_v41/v41idx/build/v41idx.so -> .../deepseek_v41/v41idx3/build/v41idx3.so
    root = os.path.dirname(os.path.dirname(os.path.dirname(library_path())))
    return os.path.join(root, "v41idx3", "build", "v41idx3.so")


def tc_stats_seconds() -> float:
    value = os.getenv("VLLM_V41_INDEX_TC_STATS", "30")
    if not value.isdecimal():
        raise ValueError(f"VLLM_V41_INDEX_TC_STATS must be a number of seconds, got {value!r}")
    return float(value)


def tc_report(force: bool = False) -> None:
    """Read the kernel's counters and print them if one moved. Called where Python runs in the indexer (an eager
    call); does nothing inside a graph capture, before the scorer's first call, or sooner than the interval."""
    counters = _TC_STATE["counters"]
    if counters is None:
        return
    if counters.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    now = time.monotonic()
    if not force and now - _TC_STATE["read_at"] < _TC_SECONDS:
        return
    _TC_STATE["read_at"] = now
    values = counters.tolist()
    if values != _TC_STATE["last"]:
        _TC_STATE["last"] = values
        logger.info(
            "V4.1 indexer: v41idx3 counters: %d calls, %d cells, %d cells over the first look's limit, %d heads "
            "recomputed by the serving chain.",
            values[2], values[3], values[0], values[1],
        )


def _tc_scores(out, q, w, keys, positions, heads, ratio) -> None:
    if heads != 8:
        raise ValueError(f"{_TC_KNOB}=1: the tensor-core scorer takes eight index heads per rank; got {heads}")
    counters = _TC_STATE["counters"]
    if counters is None:
        if out.is_cuda and torch.cuda.is_current_stream_capturing():
            # The counters must not come from a graph's own memory pool: a replay would write over them.
            raise RuntimeError(f"{_TC_KNOB}=1: the scorer's first call is inside a graph capture")
        counters = torch.zeros(4, dtype=torch.int64, device=out.device)
        _TC_STATE["counters"] = counters
        _TC_STATE["read_at"] = time.monotonic()
        logger.info(
            "V4.1 indexer: v41idx3 tensor-core decode scorer active (heads %d); its counters are read at eager "
            "indexer calls, at most every %d s.",
            heads, int(_TC_SECONDS),
        )
    elif counters.device != out.device:
        raise RuntimeError(f"{_TC_KNOB}=1: one worker, one card: the counters are on {counters.device}, this call on {out.device}")
    else:
        tc_report()
    torch.ops.v41idx3.index_scores_tc(out, q, w, keys, positions, heads, ratio, 0, counters)


if _TC:
    if _KNOB != "1":
        raise ValueError(f"{_TC_KNOB}=1 needs VLLM_V41_INDEX_TILED=1")
    _TC_SECONDS = tc_stats_seconds()
    _TC_LIBRARY = tc_library_path()
    torch.ops.load_library(_TC_LIBRARY)  # raises OSError when the .so is missing
    if not hasattr(torch.ops.v41idx3, "index_scores_tc"):
        raise RuntimeError(f"{_TC_KNOB}=1: {_TC_LIBRARY} has no v41idx3 ops")
    logger.info("V4.1 indexer: v41idx3 library loaded from %s", _TC_LIBRARY)
