# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-schema DeepSeek V4.1 activation transport over existing PP rails."""

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import blake2b
from math import prod

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator, Handle
from vllm.logger import init_logger
from vllm.models.deepseek_v41.contract import INDEX_KEY_WIDTH, KV_ROW_WIDTH, StagePlan
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

SWITCH = "VLLM_V41_PP_LEAN_HANDOFF"
logger = init_logger(__name__)
_OUTPUT = ModelRunnerOutput | AsyncModelRunnerOutput | None
_RunModel = Callable[
    [SchedulerOutput, IntermediateTensors | None], IntermediateTensors | _OUTPUT
]


def lean_handoff_enabled(value: str | None) -> bool:
    if value in (None, "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{SWITCH} must be 0, 1, or unset, got {value!r}")


@dataclass(frozen=True)
class ForwardStep:
    ordinal: int
    rows: int
    schedule_digest: int

    @classmethod
    def from_schedule(
        cls, ordinal: int, schedule: SchedulerOutput, max_rows: int
    ) -> "ForwardStep | None":
        counts = schedule.num_scheduled_tokens
        rows = schedule.total_num_scheduled_tokens
        slots = schedule.v41_slots
        drafts = schedule.scheduled_spec_decode_tokens
        if (
            not 0 <= rows <= max_rows
            or rows != sum(counts.values())
            or any(count <= 0 for count in counts.values())
            or not drafts.keys() <= counts.keys()
            or (slots is not None and slots.keys() != counts.keys())
        ):
            raise RuntimeError(
                f"{SWITCH}: invalid scheduled row or request layout at step {ordinal}"
            )
        if rows == 0:
            return None

        # The workers share the same scheduler RPC. Check the ordered layout
        # as well as the ordinal, so equal-width requests cannot trade slots.
        # Token values never enter the control record or its digest.
        layout = tuple(
            (
                request,
                count,
                None if slots is None else slots[request],
                len(drafts.get(request, ())),
            )
            for request, count in counts.items()
        )
        digest = blake2b(repr(layout).encode(), digest_size=8).digest()
        return cls(ordinal, rows, int.from_bytes(digest, "little", signed=True))

    def control(self) -> torch.Tensor:
        return torch.tensor(
            (self.ordinal, self.rows, self.schedule_digest),
            dtype=torch.int64,
            device="cpu",
        )

    def check_control(self, control: torch.Tensor) -> None:
        ordinal, rows, digest = control.tolist()
        if ordinal != self.ordinal:
            raise RuntimeError(
                f"{SWITCH}: sequence mismatch: expected {self.ordinal}, received {ordinal}"
            )
        if rows != self.rows:
            raise RuntimeError(
                f"{SWITCH}: rows mismatch at step {ordinal}: expected {self.rows}, received {rows}"
            )
        if digest != self.schedule_digest:
            raise RuntimeError(f"{SWITCH}: schedule mismatch at step {ordinal}")


@dataclass(frozen=True)
class TensorField:
    name: str
    tail: tuple[int, ...]
    dtype: torch.dtype

    def shape(self, step: ForwardStep) -> tuple[int, ...]:
        return (step.rows, *self.tail)


@dataclass(frozen=True)
class PendingReceive:
    step: ForwardStep
    schema: tuple[TensorField, ...]
    device: torch.device
    pp: GroupCoordinator
    tp: GroupCoordinator
    control: torch.Tensor
    work: dist.Work

    @classmethod
    def post(
        cls,
        step: ForwardStep,
        schema: tuple[TensorField, ...],
        device: torch.device,
        pp: GroupCoordinator,
        tp: GroupCoordinator,
    ) -> "PendingReceive":
        control = torch.empty(3, dtype=torch.int64, device="cpu")
        work = dist.irecv(
            control, src=pp.ranks[pp.rank_in_group - 1], group=pp.cpu_group
        )
        return cls(step, schema, device, pp, tp, control, work)

    def complete(self) -> dict[str, torch.Tensor]:
        self.work.wait()
        self.step.check_control(self.control)
        shards = {
            field.name: torch.empty(
                self.step.rows * prod(field.tail) // self.tp.world_size,
                dtype=field.dtype,
                device=self.device,
            )
            for field in self.schema
        }
        for handle in self.pp.irecv_tensor_dict_static(shards):
            handle.wait()
        return {
            field.name: self.tp.all_gather(shards[field.name], dim=0).reshape(
                field.shape(self.step)
            )
            for field in self.schema
        }


class LeanIntermediateTensors(IntermediateTensors):
    """An incoming hand-off awaiting the consumer's actual forward row count."""

    def __init__(self, incoming: PendingReceive) -> None:
        super().__init__({})
        self._incoming: PendingReceive | None = incoming

    @property
    def consumed(self) -> bool:
        return self._incoming is None

    def for_rows(self, rows: int) -> dict[str, torch.Tensor]:
        incoming = self._incoming
        if incoming is None:
            raise RuntimeError(f"{SWITCH}: hand-off receive was already consumed")
        if rows != incoming.step.rows:
            raise RuntimeError(
                f"{SWITCH}: runner rows mismatch at step {incoming.step.ordinal}: "
                f"scheduled {incoming.step.rows}, runner prepared {rows}"
            )
        self.tensors = incoming.complete()
        self._incoming = None
        return self.tensors


@dataclass(frozen=True)
class PendingSend:
    control: torch.Tensor
    control_work: dist.Work
    shards: dict[str, torch.Tensor]
    payload_work: list[Handle]

    def wait(self) -> None:
        self.control_work.wait()
        for handle in self.payload_work:
            handle.wait()


@dataclass(frozen=True)
class Ready:
    next_ordinal: int


@dataclass(frozen=True)
class Running:
    ordinal: int


@dataclass(frozen=True)
class Sending:
    next_ordinal: int
    send: PendingSend
    can_overlap: bool


@dataclass(frozen=True)
class Failed:
    pass


@dataclass(frozen=True)
class Closed:
    pass


class LeanHandoff:
    """Owns one worker's forward PP traffic and pending send lifetime."""

    def __init__(
        self,
        pp: GroupCoordinator,
        tp: GroupCoordinator,
        stage: StagePlan,
        device: torch.device,
        hidden_size: int,
        max_rows: int,
    ) -> None:
        cuts = {2: ((0, 20), (20, 40)), 3: ((0, 14), (14, 28), (28, 40))}
        if pp.world_size not in cuts:
            raise ValueError(
                f"{SWITCH}: PP group has {pp.world_size} stages, requires 2 or 3"
            )
        if stage.stages != pp.world_size:
            raise ValueError(
                f"{SWITCH}: stage plan has {stage.stages} stages, "
                f"PP group requires {pp.world_size}"
            )
        if not 0 <= pp.rank_in_group < pp.world_size:
            raise ValueError(
                f"{SWITCH}: PP rank is {pp.rank_in_group}, "
                f"requires 0 through {pp.world_size - 1}"
            )
        if stage.rank != pp.rank_in_group:
            raise ValueError(
                f"{SWITCH}: stage plan rank is {stage.rank}, "
                f"PP group requires rank {pp.rank_in_group}"
            )
        cut = (stage.first, stage.stop)
        expected_cut = cuts[pp.world_size][pp.rank_in_group]
        if cut != expected_cut:
            raise ValueError(
                f"{SWITCH}: stage {stage.rank} cut is {cut}, requires {expected_cut}"
            )
        if tp.world_size != 4:
            raise ValueError(
                f"{SWITCH}: TP group size is {tp.world_size}, requires 4"
            )
        if pp.use_cpu_custom_send_recv:
            raise ValueError(
                f"{SWITCH}: PP use_cpu_custom_send_recv is "
                f"{pp.use_cpu_custom_send_recv}, requires False"
            )
        if hidden_size <= 0:
            raise ValueError(
                f"{SWITCH}: model hidden width is {hidden_size}, requires at least 1"
            )
        if max_rows <= 0:
            raise ValueError(
                f"{SWITCH}: row budget is {max_rows}, requires at least 1"
            )
        fields = {
            "hidden": TensorField("hidden", (4, hidden_size), torch.float32),
            "pre": TensorField("pre", (4,), torch.float32),
            "kv_rows": TensorField("kv_rows", (KV_ROW_WIDTH,), torch.float16),
            "index_keys": TensorField("index_keys", (INDEX_KEY_WIDTH,), torch.float16),
        }
        self.receive_schema = tuple(fields[key] for key in stage.receives)
        self.send_schema = tuple(fields[key] for key in stage.sends)
        self.device = device
        self.pp = pp
        self.tp = tp
        self.max_rows = max_rows
        self._state: Ready | Running | Sending | Failed | Closed = Ready(0)
        logger.info(
            "%s=1 active (stage %d, TP rank %d)",
            SWITCH,
            pp.rank_in_group,
            tp.rank_in_group,
        )

    def _post_send(
        self, step: ForwardStep, tensors: dict[str, torch.Tensor]
    ) -> PendingSend:
        expected = tuple(field.name for field in self.send_schema)
        if tuple(tensors) != expected:
            raise RuntimeError(
                f"{SWITCH}: outgoing keys at step {step.ordinal} must be {expected}"
            )
        shards = {}
        for field in self.send_schema:
            tensor = tensors[field.name]
            shape = field.shape(step)
            if field.name == "hidden" and tensor.ndim == len(shape):
                if tensor.shape[-1] != shape[-1]:
                    raise RuntimeError(
                        f"{SWITCH}: outgoing hidden at step {step.ordinal}: "
                        f"hand-off hidden width {shape[-1]}, "
                        f"model hidden width {tensor.shape[-1]}"
                    )
            if (
                tensor.shape != shape
                or tensor.dtype != field.dtype
                or tensor.device != self.device
                or not tensor.is_contiguous()
            ):
                raise RuntimeError(
                    f"{SWITCH}: outgoing {field.name} violates the tensor schema "
                    f"at step {step.ordinal}, rows {step.rows}"
                )
            shards[field.name] = tensor.view(self.tp.world_size, -1)[
                self.tp.rank_in_group
            ]
        control = step.control()
        work = dist.isend(
            control,
            dst=self.pp.ranks[self.pp.rank_in_group + 1],
            group=self.pp.cpu_group,
        )
        return PendingSend(
            control, work, shards, self.pp.isend_tensor_dict_static(shards)
        )

    def execute(
        self,
        schedule: SchedulerOutput,
        run_model: _RunModel,
        may_overlap_next: Callable[[int, bool], bool],
    ) -> _OUTPUT:
        previous = self._state
        if isinstance(previous, Failed):
            raise RuntimeError(f"{SWITCH}: hand-off has failed")
        if isinstance(previous, Closed):
            raise RuntimeError(f"{SWITCH}: hand-off is closed")
        if isinstance(previous, Running):
            raise RuntimeError(
                f"{SWITCH}: reentrant execution at step {previous.ordinal}"
            )

        ordinal = previous.next_ordinal
        self._state = Running(ordinal)
        try:
            pending = previous if isinstance(previous, Sending) else None
            if pending is not None and not pending.can_overlap:
                pending.send.wait()
                pending = None

            step = ForwardStep.from_schedule(ordinal, schedule, self.max_rows)
            next_ordinal = ordinal + 1
            incoming = None
            if step is not None and self.receive_schema:
                incoming = LeanIntermediateTensors(
                    PendingReceive.post(
                        step, self.receive_schema, self.device, self.pp, self.tp
                    )
                )
            output = run_model(schedule, incoming)
            if incoming is not None and not incoming.consumed:
                raise RuntimeError(
                    f"{SWITCH}: step {ordinal} left its receive unconsumed"
                )

            if step is not None and self.send_schema:
                if not isinstance(output, IntermediateTensors):
                    raise RuntimeError(
                        f"{SWITCH}: step {ordinal} produced no outgoing tensors"
                    )
                if pending is not None:
                    pending.send.wait()
                overlap = may_overlap_next(step.rows, False)
                self._state = Sending(
                    next_ordinal, self._post_send(step, output.tensors), overlap
                )
                return None

            if isinstance(output, IntermediateTensors):
                raise RuntimeError(
                    f"{SWITCH}: step {ordinal} produced unexpected outgoing tensors"
                )
            self._state = (
                Ready(next_ordinal)
                if pending is None
                else Sending(next_ordinal, pending.send, pending.can_overlap)
            )
            return output
        except BaseException:
            self._state = Failed()
            raise

    def close(self) -> None:
        previous = self._state
        if isinstance(previous, Running):
            raise RuntimeError(f"{SWITCH}: cannot close during step {previous.ordinal}")
        self._state = Closed()
        if isinstance(previous, Sending):
            previous.send.wait()
