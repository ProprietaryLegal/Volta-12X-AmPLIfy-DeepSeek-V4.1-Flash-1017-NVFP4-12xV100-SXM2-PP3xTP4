# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_CARD_LIMIT_MIB: a ceiling on what one worker's card may hold.

Nothing in the engine caps the caching allocator: after warm-up the long-lived
memory is fixed, but the eager prompt-read path allocates by chunk size and
the allocator keeps what it once reserved unless a segment is wholly free
(model.py releases the cache at every forward of 2,048 rows or more; a segment
pinned by one live block stays). On the twelve-card all-NVFP4 build the middle
stage's cards crept from 31,778 to 31,836 MiB of 32,510 in half an hour
(window w3, 2026-11-17).

At the end of warm-up (graphs captured) every worker measures its card:

    used     = total - free          (the driver's figures for the card)
    outside  = used - reserved       (context, NCCL, native work areas, graphs'
                                      driver-side memory: not the allocator's)

and logs them. Unset or empty: that line only (the measurement).

Set to N (MiB): the card's used memory may reach N. The allocator's own limit
is N - outside, given to torch.cuda.set_per_process_memory_fraction. At that
limit the allocator first returns its cached blocks and retries; an allocation
that still does not fit raises torch.OutOfMemoryError in that worker: the step
fails and the engine stops with the error in its log. It is never silent, and
it happens at a chosen figure below the card instead of wherever a native
allocation outside the allocator would have failed. A worker whose reserved
memory already exceeds its limit at the end of warm-up refuses to start.

Pure functions here; ``apply`` is the one place that touches the device.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

KNOB = "VLLM_V41_CARD_LIMIT_MIB"
MIB = 1 << 20


@dataclass(frozen=True)
class CardMemory:
    """One card at the end of warm-up, in MiB (rounded down)."""

    total: int
    used: int
    reserved: int

    @property
    def outside(self) -> int:
        return self.used - self.reserved


@dataclass(frozen=True)
class Measured:
    card: CardMemory


@dataclass(frozen=True)
class Limited:
    card: CardMemory
    limit: int  # the card's used memory may reach this
    allocator_limit: int  # limit - outside

    @property
    def fraction(self) -> float:
        return self.allocator_limit * MIB / (self.card.total * MIB)

    @property
    def headroom(self) -> int:
        return self.allocator_limit - self.card.reserved


def parse(text: str) -> int | None:
    """The knob's value: None (unset or empty) or a whole number of MiB."""
    if text == "":
        return None
    if not text.isdigit() or int(text) < 1:
        raise ValueError(f"{KNOB}={text!r}: expected a whole number of MiB, or unset")
    return int(text)


def card_memory(total_bytes: int, free_bytes: int, reserved_bytes: int) -> CardMemory:
    card = CardMemory(
        total=total_bytes // MIB,
        used=(total_bytes - free_bytes) // MIB,
        reserved=reserved_bytes // MIB,
    )
    if not 0 <= card.reserved <= card.used <= card.total:
        raise RuntimeError(
            f"{KNOB}: the card's figures do not add up (total {card.total}, used "
            f"{card.used}, allocator reserved {card.reserved} MiB)"
        )
    return card


def plan(card: CardMemory, limit: int | None) -> Measured | Limited:
    if limit is None:
        return Measured(card)
    if limit > card.total:
        raise ValueError(
            f"{KNOB}={limit}: above the card's {card.total} MiB; unset it for no limit"
        )
    allocator_limit = limit - card.outside
    if allocator_limit <= card.reserved:
        raise RuntimeError(
            f"{KNOB}={limit}: at the end of warm-up the card already holds "
            f"{card.used} MiB ({card.reserved} in the allocator, {card.outside} "
            f"outside it); the allocator's limit would be {allocator_limit} MiB. "
            "This worker does not start."
        )
    return Limited(card, limit, allocator_limit)


def line(rank: str, result: Measured | Limited) -> str:
    card = result.card
    base = (
        f"V4.1 card memory after warm-up ({rank}): used {card.used} of {card.total} "
        f"MiB, allocator reserved {card.reserved}, outside the allocator {card.outside}"
    )
    if isinstance(result, Measured):
        return base + f"; no limit ({KNOB} unset)"
    return base + (
        f"; {KNOB}={result.limit}: allocator limit {result.allocator_limit} MiB "
        f"(headroom {result.headroom}); beyond it an allocation fails with "
        "torch.OutOfMemoryError"
    )


def apply(device, rank: str) -> Measured | Limited:
    """End of warm-up on one worker: measure, log, and set the limit if asked."""
    import torch

    from vllm.logger import init_logger

    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    result = plan(
        card_memory(total, free, torch.cuda.memory_reserved(device)),
        parse(os.environ.get(KNOB, "")),
    )
    if isinstance(result, Limited):
        torch.cuda.set_per_process_memory_fraction(result.fraction, device)
    init_logger(__name__).info(line(rank, result))
    return result
