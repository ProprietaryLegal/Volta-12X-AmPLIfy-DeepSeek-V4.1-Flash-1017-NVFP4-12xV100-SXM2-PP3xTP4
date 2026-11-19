# SPDX-License-Identifier: Apache-2.0
"""Host ownership contracts for one-request MRV2 steps. No device operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.group_step import StepPlan
    from vllm.v1.outputs import DraftTokenIds
    from vllm.v1.worker.gpu.input_batch import InputBatch


def drafts_for_step(
    batch: InputBatch, drafts: DraftTokenIds | None
) -> DraftTokenIds:
    """Deliver proposals only when their producing step can sample a token.

    PP may schedule the final prefill before consuming an earlier chunk's
    result. The scheduler's mutable is_prefill_chunk then describes that
    later step. Delivering the earlier chunk's placeholder proposals would
    authorize decode before the final prefill's sampled token exists.
    Use the producing batch's snapshot, and omit its row entirely so an old
    intermediate result cannot clear proposals belonging to a newer step.
    """
    from vllm.v1.outputs import DraftTokenIds

    incomplete = batch.is_incomplete_prefilling_np
    if batch.num_reqs > 1:
        # VLLM_V41_MULTI_ROWS: a group of decoding requests (no prefill);
        # every request's proposals are this step's.
        if (
            incomplete is None
            or incomplete.shape != (batch.num_reqs,)
            or bool(incomplete.any())
            or drafts is None
            or drafts.req_ids != batch.req_ids
            or len(drafts.draft_token_ids) != batch.num_reqs
        ):
            raise ValueError("MRV2 concurrency drafts need one matching group step")
        return drafts
    if (
        batch.num_reqs != 1
        or incomplete is None
        or incomplete.shape != (1,)
        or drafts is None
        or drafts.req_ids != batch.req_ids
        or len(drafts.draft_token_ids) != 1
    ):
        raise ValueError("MRV2 concurrency drafts need one matching producing step")
    if bool(incomplete[0]):
        return DraftTokenIds([], [])
    return drafts


class SlotOwners:
    """FREE -> OWNED(request) -> FREE; a live owner cannot change slots."""

    def __init__(self, capacity: int) -> None:
        from vllm.v1.core.sched.v41_multi_seq import MAX_SLOTS

        if not 1 <= capacity <= MAX_SLOTS:
            raise ValueError(
                f"MRV2 concurrency supports at most {MAX_SLOTS} resident requests"
            )
        self.capacity = capacity
        self.by_request: dict[str, int] = {}

    def claim(self, request: str, slot: int) -> None:
        if not 0 <= slot < self.capacity:
            raise ValueError("MRV2 concurrency slot outside configured capacity")
        previous = self.by_request.get(request)
        if previous is not None and previous != slot:
            raise ValueError("MRV2 concurrency changed a live request's slot")
        if any(
            owner != request and value == slot
            for owner, value in self.by_request.items()
        ):
            raise ValueError("MRV2 concurrency slot still belongs to another request")
        self.by_request[request] = slot

    def release(self, request: str) -> None:
        self.by_request.pop(request, None)

    def slot_for(self, request: str) -> int:
        if request not in self.by_request:
            raise ValueError("MRV2 concurrency request has no owned slot")
        return self.by_request[request]

    def slot_for_index(self, indices: Mapping[str, int], index: int) -> int:
        owners = [request for request, value in indices.items() if value == index]
        if len(owners) != 1:
            raise ValueError("MRV2 delayed output has no unique live request index")
        return self.slot_for(owners[0])


def step_slot(
    slots: Mapping[str, int] | None,
    scheduled: Mapping[str, int],
    capacity: int,
) -> tuple[str, int] | None:
    if slots is None or set(slots) != set(scheduled) or len(slots) > 1:
        raise ValueError("MRV2 concurrency requires one scheduled request and its slot")
    if not slots:
        return None
    request, slot = next(iter(slots.items()))
    if not 0 <= slot < capacity or scheduled[request] <= 0:
        raise ValueError("MRV2 concurrency invalid slot or token count")
    return request, slot


# VLLM_V41_MULTI_ROWS (KCAMP1106 K1): several decoding requests per step.
# The scheduler groups them (v41_multi_seq.group_size); these host contracts
# turn its output into the model's plan and the per-slot commits.


def step_slots(
    slots: Mapping[str, int] | None,
    scheduled: Mapping[str, int],
    capacity: int,
    size: int,
) -> tuple[tuple[str, int], ...]:
    """The step's (request, slot) pairs in scheduled order: none, one, or a
    group of at most ``size`` requests with distinct slots."""
    if slots is None or set(slots) != set(scheduled) or len(slots) > size:
        raise ValueError(
            f"MRV2 concurrency: {len(scheduled)} scheduled requests, slots "
            f"{slots}, at most {size} per step"
        )
    pairs = tuple((request, slots[request]) for request in scheduled)
    for request, slot in pairs:
        if not 0 <= slot < capacity or scheduled[request] <= 0:
            raise ValueError("MRV2 concurrency invalid slot or token count")
    if len({slot for _, slot in pairs}) != len(pairs):
        raise ValueError(f"MRV2 concurrency: one slot twice in a step {slots}")
    return pairs


def group_plan(
    req_ids: Sequence[str],
    num_scheduled: np.ndarray,
    slot_of: Mapping[str, int],
) -> StepPlan | None:
    """The model's plan for a multi-request step, in the batch's row order
    (req_ids / num_scheduled_tokens of the InputBatch); None for one request.
    Segments enforce 1..6 rows per request and 2..4 requests."""
    from vllm.models.deepseek_v41 import group_step, multi_rows

    if len(req_ids) != len(num_scheduled):
        raise ValueError(f"{len(req_ids)} requests with {len(num_scheduled)} counts")
    if len(req_ids) == 1:
        return None
    missing = [r for r in req_ids if r not in slot_of]
    if missing:
        raise ValueError(f"MRV2 concurrency: no slot for {missing}")
    return group_step.StepPlan(
        multi_rows.Segments(tuple(int(n) for n in num_scheduled)),
        tuple(int(slot_of[r]) for r in req_ids),
    )


def plan_commit_rows(plan: StepPlan) -> tuple[tuple[int, int], ...]:
    """(slot, row of num_sampled) for each request of a sampled step."""
    return tuple((slot, row) for row, slot in enumerate(plan.slots))


def pending_commit_rows(
    owners: SlotOwners,
    indices: Mapping[str, int],
    idx_mapping: np.ndarray,
    gen_now: np.ndarray,
    gen_at_receive: np.ndarray,
    need_sampled: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    """(slot, row) for a delayed PP receive's rows that commit: PPHandler's
    exact acceptance (same index generation, sampled count needed); a freed
    index never selects a replacement request's slot."""
    rows = []
    for row, index in enumerate(idx_mapping.tolist()):
        if gen_now[index] != gen_at_receive[row] or not need_sampled[row]:
            continue
        rows.append((owners.slot_for_index(indices, index), row))
    return tuple(rows)
