# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_MULTI_SEQ: the V4.1 model's per-sequence device state, per slot.

Everything below continues one sequence from step to step, so each running
request needs its own copy (vllm/v1/core/sched/v41_multi_seq.py):

- the window rings of the ring layers and compressed anchors (attention);
- the ratio-2 compressor states (incomplete pair, candidates, metadata);
- the Engram hash history (first stage);
- the pending commit buffers (the forward's length, position, prefill flag);
- the first stage's commit protocol (``StageCommit``, PP only);
- the packed long-context KV rows (host arena, hot ring, GPU index keys) and
  the selected-row buffer: a multi-request step fills it per request at an
  indexer layer and reads it again at the group's later ratio layers.

The rest is shared: weights, per-step scratch (the store's staging buffer,
prefill only; the index-reduce bank, consumed within one indexer call;
``stage_commit_count``; the shared attention metadata), Engram IO and
everything the vLLM block pool owns per request (paged window and draft
caches).

Binding swaps the module dictionary entries that hold these tensors
(``_buffers``, ``__dict__``); it costs a few microseconds of host time and
runs only when the step's slot differs from the bound one. Eager code then
reads the bound slot; CUDA graphs bake pointers, so each slot replays graphs
captured while it was bound (the runner keys them by slot).

VLLM_V41_SNAPSHOT_HIT: ``snapshot_rings`` copies one slot's window rings to
host memory and ``restore_rings`` copies them back into the same tensors (the
pointers the graphs baked stay valid). The model state decides when
(vllm/v1/worker/gpu/model_states/deepseek_v41.py).

VLLM_V41_PARTIAL_PREFIX: ``checkpoint_rings`` copies one slot's rings to a
host checkpoint keyed by the position the forward ended at,
``restore_checkpoint`` copies one back, ``drop_checkpoints_after`` frees the
checkpoints past a request's start. The model state decides when, from its
RingTrack (RULE-h3.md); a restore of a position no checkpoint holds raises.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

from .pp_commit import StageCommit

if TYPE_CHECKING:
    from .model import DeepseekV41Model

logger = init_logger(__name__)

_C2_FIELDS = ("partial", "candidates", "metadata", "position")
_HASH_FIELDS = ("history", "candidates", "metadata")
_PENDING_FIELDS = ("pending_length", "pending_position", "pending_prefill")


def copy_rings_to_host(rings: list[torch.Tensor], host: torch.Tensor) -> None:
    """Every ring's rows into host[i]; each copy waits for the kernels that
    wrote the ring."""
    for i, ring in enumerate(rings):
        host[i].copy_(ring)


def copy_rings_from_host(rings: list[torch.Tensor], host: torch.Tensor) -> None:
    """host[i] back into every ring, in place."""
    for i, ring in enumerate(rings):
        ring.copy_(host[i])


def import_rings(rings: list[torch.Tensor], saved: list[torch.Tensor]) -> None:
    """VLLM_V41_STORE: another conversation's saved rings into a slot's ring
    tensors, in place (the pointers CUDA graphs baked stay valid)."""
    if len(saved) != len(rings) or any(s.shape != r.shape for s, r in zip(saved, rings)):
        raise ValueError(
            f"V4.1 store: {len(saved)} saved rings for {len(rings)} ring tensors"
        )
    for ring, data in zip(rings, saved):
        ring.copy_(data)


def _fresh(tensor: torch.Tensor) -> torch.Tensor:
    # Created right after load, before any forward: slot 0 holds the
    # constructors' initial values, which every new slot starts from.
    return tensor.detach().clone()


class V41Slots:
    def __init__(self, model: DeepseekV41Model, num_slots: int, group: Any):
        from vllm.model_executor.kernels.deepseek_v41_compression import V41C2State

        from vllm.v1.core.sched.v41_multi_seq import MAX_SLOTS

        if not 1 <= num_slots <= MAX_SLOTS:
            raise ValueError(f"V4.1 slots: {num_slots} (supported 1..{MAX_SLOTS})")
        self.num_slots = num_slots
        self.current = 0
        # (namespace dict, key, the value of every slot); bind() writes
        # namespace[key] = values[slot].
        self._entries: list[tuple[dict, str, list[Any]]] = []
        self.device_bytes = 0
        self.host_bytes = 0
        counts = {"ring": 0, "c2": 0, "hash": 0, "pending": 0, "commit": 0}

        def add(namespace: dict, key: str, make) -> None:
            first = namespace[key]
            values = [first] + [make(first) for _ in range(num_slots - 1)]
            for value in values[1:]:
                if isinstance(value, torch.Tensor):
                    self.device_bytes += value.numel() * value.element_size()
            self._entries.append((namespace, key, values))

        # VLLM_V41_STORE: the module of each ring, in module order (its
        # target layer is read only by the store, ring_layers()).
        self._ring_modules: list[torch.nn.Module] = []
        for module in model.modules():
            if "window_ring" in module._buffers:
                add(module._buffers, "window_ring", _fresh)
                counts["ring"] += 1
                self._ring_modules.append(module)
            if isinstance(module, V41C2State):
                for name in _C2_FIELDS:
                    add(module._buffers, name, _fresh)
                counts["c2"] += 1
        hash_state = model.hash_state
        self._hash_ns = None if hash_state is None else hash_state._buffers
        if hash_state is not None:
            for name in _HASH_FIELDS:
                add(hash_state._buffers, name, _fresh)
            counts["hash"] = 1
        for name in _PENDING_FIELDS:
            add(model._buffers, name, _fresh)
        counts["pending"] = len(_PENDING_FIELDS)
        if model.__dict__.get("stage_commit") is not None:
            add(model.__dict__, "stage_commit", lambda _: StageCommit())
            counts["commit"] = 1
        store = model.layers[model.start_layer].attn.shared.store
        self._store_ns = None if store is None else store.__dict__
        self.store_slots = 0
        if store is not None:
            states = [store.v41_slot_state()] + [
                store.v41_add_slot(group) for _ in range(num_slots - 1)
            ]
            for name in store.V41_SLOT_FIELDS:
                self._entries.append((store.__dict__, name, [s[name] for s in states]))
            for state in states[1:]:
                self.host_bytes += state["host_cpu"].numel()
                for table in ("hot", "index"):
                    self.device_bytes += sum(t.numel() for t in state[table].values())
                self.device_bytes += state["selected"].numel()
            self.store_slots = num_slots
        self.counts = counts
        # VLLM_V41_SNAPSHOT_HIT: per slot, the host copy of its rings (None
        # until the slot's first snapshot).
        self._ring_host: list[torch.Tensor | None] = [None] * num_slots
        # VLLM_V41_PARTIAL_PREFIX: per slot, position -> the host copy of its
        # rings taken right after the forward that ended there.
        self._ring_checkpoints: list[dict[int, torch.Tensor]] = [
            {} for _ in range(num_slots)
        ]
        logger.info(
            "V4.1 slots (VLLM_V41_MULTI_SEQ): %d slots; per slot %d window "
            "rings, %d C2 states, hash state %s, pending buffers, stage "
            "commit %s, packed KV %s; extra GPU %.1f MiB, extra host %.0f MiB",
            num_slots,
            counts["ring"],
            counts["c2"],
            "yes" if counts["hash"] else "no",
            "yes" if counts["commit"] else "no",
            "yes" if store is not None else "no",
            self.device_bytes / 2**20,
            self.host_bytes / 2**20,
        )

    # -- VLLM_V41_SNAPSHOT_HIT -------------------------------------------
    def rings(self, slot: int) -> list[torch.Tensor]:
        """The slot's window rings, in module order."""
        return [
            values[slot] for _, key, values in self._entries if key == "window_ring"
        ]

    def ring_host_bytes(self) -> int:
        """Host memory of the ring snapshots of all slots (0 before the first)."""
        return sum(
            host.numel() * host.element_size()
            for host in self._ring_host
            if host is not None
        )

    def has_ring_snapshot(self, slot: int) -> bool:
        return self._ring_host[slot] is not None

    def snapshot_rings(self, slot: int) -> None:
        """Copy the slot's rings to host memory (pageable: nothing is pinned
        or registered with the driver)."""
        rings = self.rings(slot)
        host = self._ring_host[slot]
        if host is None:
            host = torch.empty((len(rings), *rings[0].shape), dtype=rings[0].dtype)
            self._ring_host[slot] = host
        copy_rings_to_host(rings, host)

    def store_fields(self, slot: int) -> dict[str, Any]:
        """VLLM_V41_STORE: the slot's packed-store fields (V41_SLOT_FIELDS),
        whichever slot is bound."""
        if self._store_ns is None:
            raise RuntimeError("V4.1 slots: no packed store")
        return {
            key: values[slot]
            for namespace, key, values in self._entries
            if namespace is self._store_ns
        }

    def diag_values(self, slot: int) -> dict[str, torch.Tensor]:
        """VLLM_V41_STORE_DUMP diagnostics: the slot's ratio-2 compressor
        states (in module order) and Engram hash state."""
        out: dict[str, torch.Tensor] = {}
        c2 = 0
        for namespace, key, values in self._entries:
            if namespace is self._hash_ns:
                out[f"hash.{key}"] = values[slot]
            elif key in _C2_FIELDS and namespace is not self._store_ns:
                if key == _C2_FIELDS[0]:
                    c2 += 1
                out[f"c2.{c2 - 1}.{key}"] = values[slot]
        return out

    def ring_layers(self) -> list[int]:
        """VLLM_V41_STORE: the target layer of each ring, in rings() order."""
        return [module.layer for module in self._ring_modules]

    def ring_snapshot(self, slot: int) -> torch.Tensor:
        """VLLM_V41_STORE: the slot's host ring snapshot (taken at the
        forward that ended the prompt), rings in ring_layers order."""
        host = self._ring_host[slot]
        if host is None:
            raise RuntimeError(f"V4.1 slots: slot {slot} has no ring snapshot")
        return host

    def restore_rings(self, slot: int) -> None:
        """Copy the slot's last snapshot back into its rings."""
        host = self._ring_host[slot]
        if host is None:
            raise RuntimeError(f"V4.1 slots: slot {slot} has no ring snapshot")
        copy_rings_from_host(self.rings(slot), host)

    # -- VLLM_V41_PARTIAL_PREFIX ------------------------------------------
    def checkpoint_positions(self, slot: int) -> list[int]:
        return sorted(self._ring_checkpoints[slot])

    def checkpoint_host_bytes(self) -> int:
        """Host memory of the ring checkpoints of all slots."""
        return sum(
            host.numel() * host.element_size()
            for checkpoints in self._ring_checkpoints
            for host in checkpoints.values()
        )

    def checkpoint_rings(self, slot: int, at: int) -> None:
        """Copy the slot's rings to a new host checkpoint at ``at`` (pageable
        memory, as the snapshot). A second checkpoint at the same position
        is an error: the track drops a checkpoint before its rows are rewritten."""
        checkpoints = self._ring_checkpoints[slot]
        if at in checkpoints:
            raise RuntimeError(f"V4.1 slots: slot {slot} already holds a checkpoint at {at}")
        rings = self.rings(slot)
        host = torch.empty((len(rings), *rings[0].shape), dtype=rings[0].dtype)
        copy_rings_to_host(rings, host)
        checkpoints[at] = host

    def restore_checkpoint(self, slot: int, at: int) -> None:
        """Copy the slot's checkpoint at ``at`` back into its rings."""
        host = self._ring_checkpoints[slot].get(at)
        if host is None:
            raise RuntimeError(
                f"V4.1 slots: slot {slot} has no ring checkpoint at {at} "
                f"(held: {self.checkpoint_positions(slot)})"
            )
        copy_rings_from_host(self.rings(slot), host)

    def clear_checkpoints(self, slot: int) -> None:
        """Free every checkpoint of the slot (another conversation's rows
        take it: a store restore)."""
        self._ring_checkpoints[slot].clear()

    def drop_checkpoints_after(self, slot: int, at: int) -> list[int]:
        """Free the slot's checkpoints past ``at`` (a request starts there and
        rewrites the rows from it). Returns the positions dropped."""
        checkpoints = self._ring_checkpoints[slot]
        dropped = sorted(position for position in checkpoints if position > at)
        for position in dropped:
            del checkpoints[position]
        return dropped

    def bind(self, slot: int) -> None:
        if slot == self.current:
            return
        if not 0 <= slot < self.num_slots:
            raise ValueError(f"V4.1 slots: slot {slot} outside 0..{self.num_slots - 1}")
        for namespace, key, values in self._entries:
            namespace[key] = values[slot]
        self.current = slot
