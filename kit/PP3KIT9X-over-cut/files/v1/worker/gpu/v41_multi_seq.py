# SPDX-License-Identifier: Apache-2.0
"""MRV2 adapter for V4.1's scheduler-owned request slots.

Persistent MRV2 token/draft arrays already use stable request indices. Only
private model state and target graphs need rebinding. Delayed PP commits bind
the receive's owner first, before the next scheduled request is selected.
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import replace
from vllm.logger import init_logger
from vllm.models.deepseek_v41 import multi_rows
from vllm.v1.core.sched.v41_multi_seq import group_size, multi_seq_decision
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.model_states.deepseek_v41 import DeepseekV41ModelState
from vllm.v1.worker.gpu import v41_group_graphs
from vllm.v1.worker.gpu.v41_multi_seq_state import (
    SlotOwners,
    group_plan,
    pending_commit_rows,
    plan_commit_rows,
    step_slot,
    step_slots,
)

if TYPE_CHECKING:
    from vllm.config import CompilationConfig, SchedulerConfig, VllmConfig
    from vllm.models.deepseek_v41.group_step import StepPlan
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


def single_seq_scheduler_config(
    vllm_config: VllmConfig, max_num_seqs: int = 1
) -> SchedulerConfig:
    """The scheduler config at B=max_num_seqs (1: one slot's private graph
    manager; n: a group graph manager of n slots).

    SchedulerConfig carries max_model_len and is_encoder_decoder as InitVars:
    replace() re-runs __post_init__ and cannot read them back from the
    instance, so they come from the model config (a bare replace() raises a
    pydantic ValidationError at engine start; a two-slot boot, 2026-11-04).
    """
    model_config = vllm_config.model_config
    if model_config is None:
        raise RuntimeError("MRV2 concurrency graphs require a model config")
    return replace(
        vllm_config.scheduler_config,
        max_num_seqs=max_num_seqs,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )


SLOT_CAPTURE_SIZES = tuple(range(1, 7))


def slot_compilation_config(
    shared: CompilationConfig, sizes: tuple[int, ...] = SLOT_CAPTURE_SIZES
) -> CompilationConfig:
    """A private compilation config for one slot's B=1 graph manager.

    The manager reassigns only cudagraph_capture_sizes and
    max_cudagraph_capture_size (adjust_cudagraph_sizes_for_spec_decode); it
    reads everything else. A shallow copy with its own size list isolates those
    assignments. deepcopy() must not be used: the config holds the live
    forward context (layers, KV and their GPU tensors), and copying it asked
    for 14.7-16 GiB per GPU at a two-slot boot (2026-11-04T20:33Z).
    """
    private = copy(shared)
    private.cudagraph_capture_sizes = list(sizes)
    return private


class V41MultiSeqRunner:
    # VLLM_V41_MULTI_ROWS (KCAMP1106 K1): decoding requests per forward (the
    # scheduler's group_size; 1 = one request per forward, a group runs
    # eager), this step's scheduled (request, slot) pairs and a group's plan
    # (set by plan_step, read until the next begin_step).
    group_size: int = 1
    selected: tuple[tuple[str, int], ...] = ()
    plan: StepPlan | None = None
    # VLLM_V41_GROUP_GRAPHS: one FULL graph manager per ordered slot tuple
    # (v41_group_graphs.py); empty = group steps run eager.
    group_graphs: bool = False
    group_managers: dict[tuple[int, ...], v41_group_graphs.GroupGraphManager]

    def __init__(self, runner: GPUModelRunner) -> None:
        enabled, reason = multi_seq_decision(runner.vllm_config)
        if not enabled:
            raise ValueError(
                "MRV2 concurrency adapter requires an admitted configuration"
            )
        if not isinstance(runner.model_state, DeepseekV41ModelState):
            raise ValueError("MRV2 concurrency adapter requires V4.1 model state")
        self.runner = runner
        self.state = runner.model_state
        self.state.v41_multi_seq = True
        self.capacity = runner.max_num_reqs
        self.owners = SlotOwners(self.capacity)
        self.managers: list[ModelCudaGraphManager] = []
        self.group_size = group_size(self.capacity, multi_rows.ENABLED)
        self.group_graphs = v41_group_graphs.enabled() and self.group_size > 1
        self.group_managers = {}
        self.bind(0, fresh=True)
        logger.info(
            "VLLM_V41_MULTI_SEQ=1 active (MRV2 runner): slots=%d; %s; "
            "up to %d request(s) per forward (VLLM_V41_MULTI_ROWS: groups of "
            "decoding requests run eager); private target graph pools; draft M<=5",
            self.capacity,
            reason,
            self.group_size,
        )
        if self.group_graphs:
            logger.info(
                "VLLM_V41_GROUP_GRAPHS=1 active: group steps of equal rows replay "
                "FULL graphs (%d ordered slot tuples x rows %s, private pools); "
                "unequal rows run eager",
                len(v41_group_graphs.slot_tuples(self.capacity, self.group_size)),
                v41_group_graphs.GROUP_ROWS,
            )

    def bind(self, slot: int, *, fresh: bool = False) -> None:
        self.runner.get_model().v41_bind_slot(slot, new_sequence=fresh)
        # ModelState otherwise caches the slot-0 StageCommit at construction.
        self.state._stage_commit = self.runner.get_model().model.stage_commit
        if self.managers:
            self.runner.cudagraph_manager = self.managers[slot]

    def bind_pending(self) -> None:
        handler = self.runner.pp_handler
        if handler is None or not handler.queue or handler.queue[0] is None:
            return
        pending = handler.queue[0]
        if self.group_size > 1 and pending.idx_mapping_np.shape[0] > 1:
            # A group step's receive: each committing row in its own slot
            # (postprocess_state -> commit_group).
            rows = pending_commit_rows(
                self.owners,
                self.runner.req_states.req_id_to_index,
                pending.idx_mapping_np,
                handler.req_idx_gen_np,
                pending.gen_at_receive_np,
                pending.need_sampled_mask,
            )
            if rows:
                self.state.group_commit = rows
            return
        if pending.idx_mapping_np.shape != (1,):
            raise ValueError("MRV2 concurrency received more than one PP request")
        index = int(pending.idx_mapping_np[0])
        # Match PPHandler's exact generation and sampled-mask acceptance. A
        # freed index must never select a replacement request's private state.
        if (
            handler.req_idx_gen_np[index] != pending.gen_at_receive_np[0]
            or not pending.need_sampled_mask[0]
        ):
            return
        slot = self.owners.slot_for_index(self.runner.req_states.req_id_to_index, index)
        self.bind(slot)

    def _store_worker(self):
        """VLLM_V41_STORE / VLLM_V41_STORE_DUMP: made at the first step (the KV
        caches exist by then); with both off nothing of the store loads."""
        if self._store_checked:
            return self.state.v41_store_worker
        self._store_checked = True
        from vllm.v1.core.sched.v41_multi_seq import store_dump_dir, store_requested

        if store_requested() or store_dump_dir() is not None:
            from vllm.models.deepseek_v41.store_worker import make_store_worker

            self.state.v41_store_worker = make_store_worker(
                self.runner, self.runner.get_model().model
            )
        return self.state.v41_store_worker

    _store_checked: bool = False

    def begin_step(self, output: SchedulerOutput) -> None:
        worker = self._store_worker()
        if worker is not None:
            # Block ids, DropOps, the step's save, the zeroing check: before
            # any request of this step is added (add_requests runs next).
            worker.begin_step(output)
        for request in output.finished_req_ids | (output.preempted_req_ids or set()):
            self.owners.release(request)
        self.plan = None
        self.state.group_plan = None
        self.runner.get_model().model.v41_plan = None
        if self.group_size > 1:
            self.selected = step_slots(
                output.v41_slots,
                output.num_scheduled_tokens,
                self.capacity,
                self.group_size,
            )
            if len(self.selected) > 1:
                self._begin_group(output)
                return
            selected = self.selected[0] if self.selected else None
        else:
            selected = step_slot(
                output.v41_slots, output.num_scheduled_tokens, self.capacity
            )
        if selected is None:
            return
        request, slot = selected
        self.owners.claim(request, slot)
        fresh = any(
            new.req_id == request and new.num_computed_tokens == 0
            for new in output.scheduled_new_reqs
        )
        self.bind(slot, fresh=fresh)

    def _begin_group(self, output: SchedulerOutput) -> None:
        """Several decoding requests in one forward: claim every slot (a
        group never holds a new request: the scheduler groups decoders)."""
        new = {
            r.req_id for r in output.scheduled_new_reqs if r.num_computed_tokens == 0
        }
        for request, slot in self.selected:
            if request in new:
                raise ValueError(f"MRV2 concurrency: new request {request} in a group")
            self.owners.claim(request, slot)
        # The model binds each request's slot around its per-sequence parts.
        self.bind(self.selected[0][1])

    def plan_step(self, input_batch: InputBatch) -> StepPlan | None:
        """The model's plan for this step's batch (None: one request). The
        target forward, its sampling and the drafter read it."""
        if len(self.selected) < 2:
            return None
        num_reqs = input_batch.num_reqs
        plan = group_plan(
            input_batch.req_ids[:num_reqs],
            input_batch.num_scheduled_tokens[:num_reqs],
            dict(self.selected),
        )
        if input_batch.num_tokens_after_padding != input_batch.num_tokens:
            raise ValueError("MRV2 concurrency: a group step is never padded")
        self.plan = plan
        self.state.group_plan = plan
        self.runner.get_model().model.v41_plan = plan
        return plan

    def group_graph(
        self, plan: StepPlan
    ) -> tuple[v41_group_graphs.GroupGraphManager, Any] | None:
        """VLLM_V41_GROUP_GRAPHS: the (manager, FULL descriptor) replaying
        this group step; None runs it eager (switch off, or unequal rows)."""
        if not self.group_graphs:
            return None
        return v41_group_graphs.select(plan, self.group_managers)

    def after_draft(self) -> None:
        """sample_tokens on the last PP rank, after speculator.propose wrote the
        drafter's rows of this step's forward: the store's drafter pages and the
        dump's save event of a prompt end (store_worker.StoreWorker.after_draft)."""
        worker = self._store_worker()
        if worker is not None:
            worker.after_draft()

    def commit_sampled(self) -> None:
        """Before the sampled step's postprocess: each request's row."""
        if self.plan is not None:
            self.state.group_commit = plan_commit_rows(self.plan)

    def init_graphs(self) -> None:
        runner = self.runner
        existing = runner.cudagraph_manager
        assert existing is not None
        # Keep resident storage at capacity, but graph geometry at B=1. A
        # private compilation config prevents manager normalization leaking.
        for _ in range(self.capacity):
            config = copy(runner.vllm_config)
            config.scheduler_config = single_seq_scheduler_config(config)
            config.compilation_config = slot_compilation_config(config.compilation_config)
            manager = ModelCudaGraphManager(
                config, runner.device, existing.cudagraph_mode, decode_query_len=1
            )
            # Sharing a pool would let a later slot's capture reuse an earlier
            # graph's live output/workspace addresses.
            if existing.cudagraph_mode:
                manager.pool = torch.cuda.graph_pool_handle()
            self.managers.append(manager)
        runner.cudagraph_manager = self.managers[0]
        if not self.group_graphs:
            return
        if not existing.cudagraph_mode:
            raise ValueError(
                "VLLM_V41_GROUP_GRAPHS=1 needs CUDA graphs (cudagraph mode is NONE)"
            )
        for slots in v41_group_graphs.slot_tuples(self.capacity, self.group_size):
            n = len(slots)
            config = copy(runner.vllm_config)
            config.scheduler_config = single_seq_scheduler_config(config, n)
            config.compilation_config = slot_compilation_config(
                config.compilation_config,
                tuple(n * r for r in v41_group_graphs.GROUP_ROWS),
            )
            manager = v41_group_graphs.GroupGraphManager(
                config, runner.device, existing.cudagraph_mode, slots
            )
            # A private pool per tuple, as per slot (see above).
            manager.pool = torch.cuda.graph_pool_handle()
            self.group_managers[slots] = manager

    def capture_remaining(self) -> None:
        runner = self.runner
        try:
            for slot, manager in enumerate(self.managers[1:], start=1):
                self.bind(slot, fresh=True)
                manager.capture(
                    runner.model,
                    runner.model_state,
                    runner.input_buffers,
                    runner.intermediate_tensors,
                    runner.block_tables,
                    runner.attn_groups,
                    runner.kv_cache_config,
                    has_lora=False,
                    use_aux_hidden_state_outputs=runner.use_aux_hidden_state_outputs,
                )
                torch.accelerator.synchronize()
            for manager in self.group_managers.values():
                manager.capture_group(
                    runner.model,
                    runner.model_state,
                    runner.get_model().model,
                    input_buffers=runner.input_buffers,
                    intermediate_tensors=runner.intermediate_tensors,
                    block_tables=runner.block_tables,
                    attn_groups=runner.attn_groups,
                    kv_cache_config=runner.kv_cache_config,
                    has_lora=False,
                    use_aux_hidden_state_outputs=runner.use_aux_hidden_state_outputs,
                )
                torch.accelerator.synchronize()
        finally:
            self.bind(0, fresh=True)
