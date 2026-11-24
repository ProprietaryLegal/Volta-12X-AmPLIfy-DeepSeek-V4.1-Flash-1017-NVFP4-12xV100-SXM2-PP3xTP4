# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 model state for Model Runner V2.

V4.1 keeps private per-sequence state on its first pipeline stage (the Engram
hash history and the ratio-2 compressor states). The V1 runner commits that
state after every forward; this model state does the same through the MRV2
hooks, so the model code is unchanged:

* ``prepare_attn`` passes the per-request prefill flag that ``V41CacheBuilder``
  copies into its shared device flag, and marks capture and dummy batches as
  dummy (the builder then writes False), as the V1 runner does.
* ``add_request`` rewinds a same-session prefix hit (``model.restore_prefix``,
  which also resets the PP stage-1 commit protocol) and resets that protocol
  for a request that starts at position 0.
* ``postprocess_state`` commits the sampled counts
  (``model.commit_speculative_state``). On a first stage that is not the last
  this is the "sampled counts arrived" path of ``pp_commit.StageCommit``
  (``take_sampled``); on the last stage of PP2 the model ignores it; with PP1
  it is the commit itself, and a prefill chunk selects its full length on the
  device.
* The runner calls ``pp_before_forward`` / ``pp_after_forward`` on a first
  stage that is not the last (``model_runner.py``).

With ``VLLM_V41_SNAPSHOT_HIT=1`` it also keeps one ``RingTrack`` per slot
(vllm/v1/core/sched/v41_multi_seq.py), fed with every request start and every
executed forward. It copies the slot's window rings to host memory right
after the forward that ends a request's prompt, and copies them back when a
request starts where only that snapshot holds the rows before it. A start
that neither the live rings nor the snapshot covers raises: the scheduler
ran the same arithmetic and must never have admitted it.

It also keeps the per-request inputs of V4.1 coupled drafts
(``VLLM_V41_COUPLED_DRAFTS``): the stream key on the device and a host copy of
the temperature, which the speculator reads without a sync.
"""

import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank
from vllm.distributed.ec_transfer import has_ec_transfer
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.logger import init_logger
from vllm.sampling_params import SamplingType
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.v41_multi_seq import (
    SNAPSHOT_HIT_KNOB,
    RingPath,
    RingTrack,
    snapshot_hit_requested,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)

V41_MODEL_TYPE = "deepseek_v41_text"


class _SkipCounter(Protocol):
    def step(
        self, skipped: bool, req_ids: list[str], num_scheduled: list[int]
    ) -> list[str]: ...


class _SkipHelpers(Protocol):
    """The deployed V1 helper contract (BASECHECK verifies its source)."""

    KNOB: str
    skip_step: Callable[[Iterable[int], Iterable[int], Iterable[int], int], bool]
    SkipCounter: Callable[[], _SkipCounter]

    def static_block_reasons(
        self,
        *,
        pooling: bool,
        spec_method: str | None,
        kv_transfer: bool,
        ec_transfer: bool,
    ) -> list[str]: ...

    def step_block_reasons(
        self,
        *,
        prompt_logprobs: bool,
        routed_experts: bool,
    ) -> list[str]: ...


@dataclass
class _PrefillSkip:
    """One owner for the runtime binding, block reasons and V1 accounting."""

    rows: int
    bind: Callable[[bool], None]
    helpers: _SkipHelpers
    pooling: bool
    spec_method: str | None
    report: bool
    counter: _SkipCounter
    logged: set[str] = field(default_factory=set)

    @cached_property
    def static_reasons(self) -> list[str]:
        # KV transfer initializes after load_model. Like V1, query it at the
        # first runtime plan, not while constructing the model state.
        return self.helpers.static_block_reasons(
            pooling=self.pooling,
            spec_method=self.spec_method,
            kv_transfer=has_kv_transfer_group(),
            ec_transfer=has_ec_transfer(),
        )

    def plan(
        self,
        batch: InputBatch,
        prompt: list[int],
        *,
        prompt_logprobs: bool,
        routed_experts: bool,
    ) -> bool:
        reasons = self.static_reasons or self.helpers.step_block_reasons(
            prompt_logprobs=prompt_logprobs, routed_experts=routed_experts
        )
        for reason in reasons:
            if reason not in self.logged:
                self.logged.add(reason)
                if self.report:
                    logger.info(
                        "%s: skip disabled (%s); logged once.",
                        self.helpers.KNOB,
                        reason,
                    )
        n = batch.num_reqs
        scheduled = batch.num_scheduled_tokens[:n].tolist()
        skip = not reasons and self.helpers.skip_step(
            batch.num_computed_tokens_np[:n].tolist(), scheduled, prompt, self.rows
        )
        if self.report:
            for line in self.counter.step(skip, batch.req_ids[:n], scheduled):
                logger.info("%s: %s", self.helpers.KNOB, line)
        return skip


def is_deepseek_v41(vllm_config: VllmConfig) -> bool:
    model_config = vllm_config.model_config
    text_config = getattr(model_config, "hf_text_config", None)
    return getattr(text_config, "model_type", None) == V41_MODEL_TYPE


def _language_model(model: nn.Module) -> nn.Module:
    get_language_model = getattr(model, "get_language_model", None)
    return get_language_model() if callable(get_language_model) else model


@dataclass
class V41AttnMetadata(ModelSpecificAttnMetadata):
    """The per-request prefill flag (CPU bool), read by V41CacheBuilder."""

    is_prefilling: torch.Tensor

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        return {"is_prefilling": self.is_prefilling[:num_reqs]}


class DeepseekV41ModelState(DefaultModelState):
    v41_multi_seq: bool = False
    # VLLM_V41_MULTI_ROWS: (slot, row) pairs for the next multi-row commit;
    # set by the MULTI_SEQ adapter, consumed by postprocess_state.
    group_commit: tuple[tuple[int, int], ...] | None = None
    # This step's multi-request plan (group_step.StepPlan) or None; set by
    # the MULTI_SEQ adapter, read by the DSpark speculator.
    group_plan: Any = None
    # VLLM_V41_STORE / VLLM_V41_STORE_DUMP (store_worker.StoreWorker), set by
    # the MULTI_SEQ adapter at its first step; None with both off.
    v41_store_worker: Any = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        if self.rope_state is not None or self.supports_mm_inputs:
            raise ValueError(
                "DeepSeek-V4.1 under Model Runner V2 expects 1D positions and "
                "text-only inputs; got M-RoPE or multimodal inputs."
            )
        if vllm_config.kernel_config.tp_prefill_overlap_min_tokens > 0:
            raise ValueError(
                "DeepSeek-V4.1 under Model Runner V2 does not support "
                "tp_prefill_overlap_min_tokens > 0: its cache builder admits "
                "one request and one geometry per forward. Set it to 0."
            )
        language_model = _language_model(model)
        inner = getattr(language_model, "model", None)
        self._commit = getattr(language_model, "commit_speculative_state", None)
        self._commit_group = getattr(language_model, "commit_group", None)
        self._restore_prefix = getattr(language_model, "restore_prefix", None)
        if self._commit is None or self._restore_prefix is None or inner is None:
            raise ValueError(
                "DeepSeek-V4.1 under Model Runner V2 needs the model's "
                "commit_speculative_state and restore_prefix hooks."
            )
        # None unless this is a first stage that is not the last.
        self._stage_commit = getattr(inner, "stage_commit", None)
        # VLLM_V41_SNAPSHOT_HIT: one RingTrack per slot (made at the first
        # request: the slots are created after this state), else None.
        self._v41_inner = inner
        self._snapshot_hit = snapshot_hit_requested()
        self._ring_tracks: list[RingTrack] | None = None
        self._ring_copies = [0, 0]  # snapshots, restores
        if self._snapshot_hit:
            logger.info(
                "%s=1 active (worker): ring snapshot after the forward that ends "
                "a prompt, restore when a request starts at a snapshot; a start "
                "outside both the live rings and the snapshot raises.",
                SNAPSHOT_HIT_KNOB,
            )
        self._prefill_skip: _PrefillSkip | None = None
        self._prompt_logprobs = np.zeros(self.max_num_reqs, dtype=np.bool_)
        rows = int(getattr(inner, "_v41_skip_rows", 0))
        if rows > 0:
            bind = getattr(language_model, "prefill_skip_bind", None)
            if not callable(bind):
                raise ValueError(
                    "DeepSeek-V4.1 MRV2 prefill skip needs prefill_skip_bind."
                )
            # This dependency belongs to the deployed V1/model skip feature.
            # Import only when enabled; the original port predates that file.
            helpers = cast(
                _SkipHelpers, import_module("vllm.models.deepseek_v41.prefill_skip")
            )
            spec = vllm_config.speculative_config
            report = (
                get_pp_group().is_last_rank and get_tensor_model_parallel_rank() == 0
            )
            self._prefill_skip = _PrefillSkip(
                rows=rows,
                bind=bind,
                helpers=helpers,
                pooling=self.model_config.runner_type == "pooling",
                spec_method=spec.method if spec is not None else None,
                report=report,
                counter=helpers.SkipCounter(),
            )
            if getattr(inner, "_v41_skip_eligible", False):
                logger.info(
                    "%s=%d MRV2 skip active: per-forward prefill_skip_bind; "
                    "V1 static/step guards and chunk boundaries apply.",
                    helpers.KNOB,
                    rows,
                )

        # Coupled-draft inputs, indexed by request state index.
        self.keys_np = np.zeros(self.max_num_reqs, dtype=np.int64)
        self.keys_gpu = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self._keys_dirty = False
        # Host copy for the drafter's all-greedy check (the sampler, which
        # has its own, exists only on the last PP rank).
        self.temperature_np = np.zeros(self.max_num_reqs, dtype=np.float32)
        self._engine_seed = getattr(self.model_config, "seed", None) or 0
        logger.info(
            "DeepSeek-V4.1 Model Runner V2 state active: prefill flag through "
            "V4.1 attention metadata, sampled-count commit in postprocess_state, "
            "prefix restore on add_request (PP stage-1 commit protocol: %s).",
            "on" if self._stage_commit is not None else "not this stage",
        )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        params = new_req_data.sampling_params
        # Every stage sees new/resumed requests, unlike the last-stage sampler.
        # Replacing this slot also clears the prior request's logprob flag.
        self._prompt_logprobs[req_index] = (
            params is not None and params.prompt_logprobs is not None
        )
        worker = self.v41_store_worker
        if self._snapshot_hit:
            if worker is None or not self._store_start(worker, new_req_data.req_id):
                self._ring_start(new_req_data.num_computed_tokens)
        if new_req_data.num_computed_tokens > 0:
            # A same-session prefix hit (or a resumed request): rewind the
            # private state to the cached position. prefill_token_ids holds
            # the prompt plus any outputs of a resumed request.
            assert new_req_data.prefill_token_ids is not None
            self._restore_prefix(
                new_req_data.num_computed_tokens, new_req_data.prefill_token_ids
            )
            if worker is not None:
                # VLLM_V41_STORE_DUMP: the start event, before the first forward.
                slot = self._ring_slots().current
                worker.started(
                    slot, self._ring_tracks[slot], new_req_data.req_id,
                    new_req_data.num_computed_tokens,
                )
        elif self._stage_commit is not None:
            # A forward armed for the previous request is dropped: this one
            # starts at position 0, which resets the device protocol.
            self._stage_commit.reset()

        params = new_req_data.sampling_params
        if params is None:
            self.temperature_np[req_index] = 0.0
            return
        from vllm.v1.spec_decode import coupled_draw

        # The V1 runner's key (_fill_coupled_keys): the request seed when
        # given, else the engine seed and the request id.
        self.keys_np[req_index] = coupled_draw.request_key(
            params.seed, new_req_data.req_id, self._engine_seed
        )
        self._keys_dirty = True
        # V1 stores 0 for any greedy request (temperature below the epsilon).
        greedy = params.sampling_type == SamplingType.GREEDY
        self.temperature_np[req_index] = 0.0 if greedy else params.temperature

    # -- VLLM_V41_SNAPSHOT_HIT ---------------------------------------------
    def _ring_slots(self) -> Any:
        slots = getattr(self._v41_inner, "v41_slots", None)
        if slots is None:
            raise RuntimeError(
                f"{SNAPSHOT_HIT_KNOB}=1: the model has no slots (VLLM_V41_MULTI_SEQ)"
            )
        if self._ring_tracks is None:
            # The scheduler's value: both read VLLM_V41_PREFILL_SKIP_ROWS
            # through prefill_skip.skip_rows() (model.py: _v41_skip_rows).
            rows = int(getattr(self._v41_inner, "_v41_skip_rows", 0))
            self._ring_tracks = [RingTrack(rows) for _ in range(slots.num_slots)]
        return slots

    def _ring_start(self, at: int) -> None:
        """A request starts at ``at`` in the bound slot (the runner bound the
        step's slot before add_requests). RingTrack.start raises when neither
        the live rings nor the snapshot hold the rows before ``at``."""
        slots = self._ring_slots()
        assert self._ring_tracks is not None
        slot = slots.current
        track = self._ring_tracks[slot]
        if track.start(at) is not RingPath.RESTORE:
            return
        began = time.perf_counter()
        slots.restore_rings(slot)
        self._ring_copies[1] += 1
        if self._ring_copies[1] == 1:
            logger.info(
                "%s first restore: slot %d at %d snapshot_end %d copy_ms %.3f",
                SNAPSHOT_HIT_KNOB,
                slot,
                at,
                track.snap_end,
                (time.perf_counter() - began) * 1e3,
            )

    def _store_start(self, worker: Any, req_id: str) -> bool:
        """VLLM_V41_STORE: every start joins the slot's pending save copies;
        a store hit starts the ring track at H and restores the slot (the
        runner bound it before add_requests). True: restored."""
        slots = self._ring_slots()
        assert self._ring_tracks is not None
        slot = slots.current
        return worker.start(slot, self._ring_tracks[slot], req_id)

    def _ring_forward(self, input_batch: InputBatch, req_states: RequestState) -> None:
        """After one executed forward: move the bound slot's ring positions,
        and copy its rings to host memory if the forward ended the prompt."""
        slots = self._ring_slots()
        assert self._ring_tracks is not None
        if input_batch.num_reqs != 1:
            raise RuntimeError(
                f"{SNAPSHOT_HIT_KNOB}=1: {input_batch.num_reqs} requests in one forward"
            )
        slot = slots.current
        index = int(input_batch.idx_mapping_np[0])
        start = int(input_batch.num_computed_tokens_np[0])
        prompt_len = int(req_states.prompt_len.np[index])
        ended = self._ring_tracks[slot].forward(
            start, int(input_batch.num_scheduled_tokens[0]), prompt_len
        )
        worker = self.v41_store_worker
        if not ended:
            if worker is not None:
                worker.after_forward(slot, input_batch.req_ids[0], False, prompt_len, start)
            return
        first = not slots.has_ring_snapshot(slot)
        if first and self.device.type == "cuda":
            # Time the copy alone: let the forward's kernels finish first.
            torch.cuda.synchronize(self.device)
        began = time.perf_counter()
        slots.snapshot_rings(slot)
        self._ring_copies[0] += 1
        if first:
            logger.info(
                "%s first snapshot: slot %d end %d rings %d host_bytes %d copy_ms %.3f",
                SNAPSHOT_HIT_KNOB,
                slot,
                self._ring_tracks[slot].snap_end,
                len(slots.rings(slot)),
                slots.ring_host_bytes(),
                (time.perf_counter() - began) * 1e3,
            )
        if worker is not None:
            # VLLM_V41_STORE: the step's SaveOp (or SkipSaveOp) after the
            # snapshot; VLLM_V41_STORE_DUMP: the save event.
            worker.after_forward(slot, input_batch.req_ids[0], True, prompt_len, start)

    @contextmanager
    def prefill_skip_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        cudagraph_mode: CUDAGraphMode,
        *,
        for_capture: bool = False,
    ) -> Iterator[None]:
        """Around one forward: the prefill-skip binding, then (after a forward
        that returned) VLLM_V41_SNAPSHOT_HIT's ring positions and snapshot."""
        with self._prefill_skip_bound(
            input_batch, req_states, cudagraph_mode, for_capture=for_capture
        ):
            yield
        if not self._snapshot_hit or input_batch.is_dummy_batch or for_capture:
            return
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return
        self._ring_forward(input_batch, req_states)

    @contextmanager
    def _prefill_skip_bound(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        cudagraph_mode: CUDAGraphMode,
        *,
        for_capture: bool = False,
    ) -> Iterator[None]:
        """UNBOUND -> bound for one runtime forward -> UNBOUND, even on error.

        Capture/profile and rows=0 never call the binding. FULL decode replays
        are bound False. A skip must execute eagerly: capturing the Python
        layer switch would bake an incomplete forward into future replays.
        """
        state = self._prefill_skip
        if state is None or input_batch.is_dummy_batch or for_capture:
            yield
            return
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            yield
            return
        indices = input_batch.idx_mapping_np[: input_batch.num_reqs]
        active_indices = list(req_states.req_id_to_index.values())
        if self.v41_multi_seq:
            active_indices = indices.tolist()
        skip = state.plan(
            input_batch,
            req_states.prompt_len.np[indices].tolist(),
            prompt_logprobs=bool(self._prompt_logprobs[active_indices].any()),
            routed_experts=self.model_config.enable_return_routed_experts,
        )
        if skip and cudagraph_mode != CUDAGraphMode.NONE:
            raise ValueError("DeepSeek-V4.1 MRV2 skip requires eager prefill.")
        try:
            state.bind(skip)
            yield
        finally:
            state.bind(False)

    def apply_staged_writes(self) -> None:
        super().apply_staged_writes()
        if self._keys_dirty:
            async_copy_to_gpu(self.keys_np, out=self.keys_gpu)
            self._keys_dirty = False

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        if isinstance(num_sampled, int):
            # A non-last stage with no sampled counts for this step (prefill
            # chunks or finishing rows). A prefill forward committed itself in
            # pp_after_forward; an armed decode forward stays armed and is
            # committed by pp_before_forward or dropped with its request.
            return
        rows, self.group_commit = self.group_commit, None
        if idx_mapping.shape[0] != 1:
            # VLLM_V41_MULTI_ROWS: several requests' counts; the MULTI_SEQ
            # adapter names each committing row's slot (v41_multi_seq.py).
            if rows is None:
                raise ValueError(
                    "DeepSeek-V4.1 commits the private state of one request per "
                    f"step; got {idx_mapping.shape[0]} rows without a group "
                    "commit (VLLM_V41_MULTI_SEQ + VLLM_V41_MULTI_ROWS)."
                )
            self._commit_group(rows, num_sampled)
            return
        if rows is not None:
            raise ValueError(f"DeepSeek-V4.1 group commit {rows} for one request")
        self._commit(num_sampled[:1])

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_id: int = 0,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(
            input_batch.is_prefilling_np[: input_batch.num_reqs]
        )
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            max_seq_len = input_batch.capture_context_len or self.max_model_len
        else:
            max_seq_len = int(seq_lens_cpu_upper_bound[:num_reqs].max().item())
        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            model_specific_attn_metadata=V41AttnMetadata(is_prefilling),
            for_cudagraph_capture=for_capture,
            prefix_anchor_lens=input_batch.prefix_anchor_lens,
            is_dummy_batch=input_batch.is_dummy_batch or for_capture,
            ubatch_id=ubatch_id,
        )
