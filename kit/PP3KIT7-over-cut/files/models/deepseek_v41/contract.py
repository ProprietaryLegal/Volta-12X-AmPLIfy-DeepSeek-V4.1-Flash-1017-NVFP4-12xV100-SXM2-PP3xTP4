# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject configurations whose semantics differ from the qualified kernels."""

_COMMON = {
    "moe_intermediate_size": 2304,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1280,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "sliding_window": 128,
    "rope_theta": 10000,
    "rms_norm_eps": 1e-20,
    "hc_mult": 4,
    "hc_eps": 1e-6,
    "hc_sinkhorn_iters": 20,
    "n_shared_experts": 1,
    "hidden_act": "silu",
    "swiglu_limit": 10.0,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "attention_bias": False,
}


def validate_runtime_geometry(config, *, draft=False):
    for name, expected in _COMMON.items():
        if getattr(config, name, None) != expected:
            raise ValueError(f"V4.1 qualified runtime requires {name}={expected!r}")
    ratios = tuple(config.compress_ratios)
    if draft:
        if ratios != (0, 0, 0):
            raise ValueError("V4.1 draft requires exactly three uncompressed layers")
        return
    if (
        ratios[:40] != (0, 0) + (2,) * 18 + (1,) * 20
        or any(ratios[40:])
        or getattr(config, "compress_rope_theta", None) != 160000
    ):
        raise ValueError("V4.1 target compression or rotary schedule differs")
    rope = config.rope_scaling or {}
    expected_rope = dict(
        rope_type="yarn",
        factor=16,
        beta_fast=32,
        beta_slow=1,
        original_max_position_embeddings=65536,
    )
    if any(rope.get(k) != value for k, value in expected_rope.items()):
        raise ValueError("V4.1 target requires the checkpoint's compressed YaRN")


def pp_split_problem(config, split):
    """Why a pipeline boundary before target layer ``split`` is not admissible.

    Returns None when admissible. Only the four-stream HC residual may cross a
    stage boundary (the KV20 rule):

    - every layer reads the KV and indexer caches of the nearest source at or
      below it, so the second stage must start at a KV and an index source;
    - Engram, its hash state and the ratio-2 compressor states are committed
      once per step from the accepted counts, so they stay on the first stage.

    For the V4.1 target (KV sources 2/8/14/20, Engram 1/14, ratio 2 on layers
    2-19) the only admissible split is 20.
    """
    layers = config.num_hidden_layers
    if not 0 < split < layers:
        return f"split must be inside 1..{layers - 1}"
    if split not in tuple(config.kv_source_layer_ids):
        return "not a KV source layer (KV would cross the boundary)"
    if split not in tuple(config.index_source_layer_ids):
        return "not an index source layer (index keys would cross the boundary)"
    if any(layer >= split for layer in config.engram_layer_ids):
        return "Engram layers and their hash state must be on the first stage"
    if any(ratio == 2 for ratio in tuple(config.compress_ratios)[split:layers]):
        return "ratio-2 compressor state must be on the first stage"
    return None


# --- Three stages (PP3 x TP4, twelve cards) ---------------------------------
#
# The split rule above ("only the four-stream HC residual may cross") admits
# one boundary, 20. A third stage needs two more things to cross, both taken
# from 1Cat-vLLM PR #1031 (cuts 14 and 28, a mirror of KV source 20):
#
# - into a stage that starts above a KV source it reads (28-39 read source 20):
#   that source's compressed latent and index key of every forward row, in
#   row order, written into the stage's own packed store (a mirror). Only a
#   ratio-1 source can be mirrored: it emits one row per token and holds no
#   committed state. The stage must start at an index source, so no top-k
#   selection crosses.
# - into a stage whose first layer is an Engram layer (14): that layer's
#   looked-up Engram rows. The hash state and the sidecar reader stay on the
#   first stage; nothing else of Engram crosses.
#
# The ratio-2 compressor state of a KV source lives with its layer; a stage
# that is not the last commits it through pp_commit.StageCommit, whichever
# stage it is.

from dataclasses import dataclass  # noqa: E402

# The stage starts this build is qualified for. Anything else is refused:
# other splits may be derivable by the rules below but were never run.
QUALIFIED_STARTS = ((0,), (0, 20), (0, 14, 28))
# Diagnostic layouts, each carrying ONE of the three-stage mechanisms, to find which one a fault lives in. Refused
# unless VLLM_V41_PP_DIAGNOSTIC_LAYOUT=1 is set: they are for test arms, never for serving.
#   (0, 20, 28)  the mirror of KV source 20 only (no Engram hand-over, no commit on a middle stage)
#   (0, 14, 20)  the Engram hand-over and the middle-stage commit only (no mirror)
DIAGNOSTIC_STARTS = ((0, 20, 28), (0, 14, 20))
DIAGNOSTIC_KNOB = "VLLM_V41_PP_DIAGNOSTIC_LAYOUT"

KV_ROW_WIDTH = 512
INDEX_KEY_WIDTH = 128


@dataclass(frozen=True)
class StagePlan:
    """What one pipeline stage owns and what crosses its two boundaries.

    ``mirror_source`` is the KV source below ``first`` whose rows this stage
    receives and keeps in its own store (None: every source it reads is its
    own). ``export_source`` is the KV source of this stage whose rows it
    sends to the next stage. ``upstream_engram`` is this stage's first layer
    when that layer is an Engram layer of a stage after the first: its Engram
    is applied by the stage before, at the end of that stage's forward, on the
    residual the layer would receive (the first stage holds the Engram sidecar
    reader and the hash state; the layer's Engram weights are built and loaded
    there, and nothing of Engram crosses the boundary). ``tail_engram`` is the
    next stage's such layer, on the stage that applies it. ``commits`` says
    the stage holds committed per-step state (Engram hash history on the first
    stage, ratio-2 compressor states wherever their source layers are).
    """

    rank: int
    stages: int
    first: int
    stop: int
    own_sources: tuple[int, ...]
    mirror_source: int | None
    export_source: int | None
    upstream_engram: int | None
    tail_engram: int | None
    commits: bool

    @property
    def is_first(self) -> bool:
        return self.rank == 0

    @property
    def is_last(self) -> bool:
        return self.rank == self.stages - 1

    @property
    def holds_embedding(self) -> bool:
        """The token embedding table is needed where tokens enter (the first
        stage) and where the drafter runs (the last); a stage between them
        never reads it and does not load it."""
        return self.is_first or self.is_last

    @property
    def store_sources(self) -> tuple[int, ...]:
        """Sources this stage's packed store holds rows for."""
        mirrored = () if self.mirror_source is None else (self.mirror_source,)
        return tuple(sorted(mirrored + self.own_sources))

    @property
    def receives(self) -> tuple[str, ...]:
        """Hand-off keys this stage receives, in schema order."""
        if self.is_first:
            return ()
        keys = ["hidden", "pre"]
        if self.mirror_source is not None:
            keys += ["kv_rows", "index_keys"]
        return tuple(keys)

    @property
    def sends(self) -> tuple[str, ...]:
        """Hand-off keys this stage sends: the next stage's ``receives``."""
        if self.is_last:
            return ()
        keys = ["hidden", "pre"]
        if self.export_source is not None:
            keys += ["kv_rows", "index_keys"]
        return tuple(keys)


def stage_plans(config, starts) -> tuple[StagePlan, ...]:
    """The plan of every stage for first layers ``starts``; raises ValueError
    naming the reason when the layout is not one this build runs."""
    starts = tuple(int(s) for s in starts)
    layers = config.num_hidden_layers
    if len(starts) == 2 and starts[0] == 0:
        problem = pp_split_problem(config, starts[1])
        if problem is not None:
            raise ValueError(f"V4.1 pipeline split at layer {starts[1]}: {problem}")
    if starts not in QUALIFIED_STARTS:
        import os

        diagnostic = os.environ.get(DIAGNOSTIC_KNOB, "0")
        if diagnostic not in ("0", "1"):
            raise ValueError(f"{DIAGNOSTIC_KNOB} requires 0 or 1, got {diagnostic!r}")
        if not (diagnostic == "1" and starts in DIAGNOSTIC_STARTS):
            raise ValueError(
                f"V4.1 pipeline stages starting at layers {starts} are not qualified "
                f"(qualified: {QUALIFIED_STARTS}; with {DIAGNOSTIC_KNOB}=1 also the "
                f"diagnostic layouts {DIAGNOSTIC_STARTS})"
            )
    kv_sources = tuple(config.kv_source_layer_ids)
    index_sources = tuple(config.index_source_layer_ids)
    engram_layers = tuple(config.engram_layer_ids)
    ratios = tuple(config.compress_ratios)
    stops = starts[1:] + (layers,)
    stage_of = {
        layer: rank
        for rank, (first, stop) in enumerate(zip(starts, stops))
        for layer in range(first, stop)
    }
    if sorted(stage_of) != list(range(layers)):
        raise ValueError(f"V4.1 stage starts {starts} do not cover layers 0..{layers - 1}")
    raw = []
    for rank, (first, stop) in enumerate(zip(starts, stops)):
        own = tuple(s for s in kv_sources if first <= s < stop)
        read = {
            max(s for s in kv_sources if s <= layer)
            for layer in range(first, stop)
            if ratios[layer]
        }
        foreign = sorted(read - set(own))
        if len(foreign) > 1:
            raise ValueError(
                f"V4.1 stage {rank} (layers {first}-{stop - 1}) reads KV sources "
                f"{foreign} of earlier stages; one mirror per stage"
            )
        mirror = foreign[0] if foreign else None
        if mirror is not None:
            if ratios[mirror] != 1:
                raise ValueError(
                    f"V4.1 stage {rank}: KV source {mirror} has ratio "
                    f"{ratios[mirror]}; only a ratio-1 source can be mirrored"
                )
            if stage_of[mirror] != rank - 1:
                raise ValueError(
                    f"V4.1 stage {rank}: KV source {mirror} is on stage "
                    f"{stage_of[mirror]}; a mirror is fed by the stage just before"
                )
            if first not in index_sources:
                raise ValueError(
                    f"V4.1 stage {rank} starts at layer {first}, which is not an "
                    f"index source: its top-k selection would cross the boundary"
                )
        for layer in range(first, stop):
            if ratios[layer]:
                index = max(s for s in index_sources if s <= layer)
                if stage_of[index] != rank:
                    raise ValueError(
                        f"V4.1 layer {layer} reads index source {index} of another stage"
                    )
        engram = [e for e in engram_layers if first <= e < stop]
        imported = None
        if rank > 0 and engram:
            if engram != [first]:
                raise ValueError(
                    f"V4.1 stage {rank}: Engram layers {engram} must be the stage's "
                    f"first layer ({first}): the first stage applies it at its end"
                )
            if rank != 1:
                raise ValueError(
                    f"V4.1 stage {rank}: only the first stage (Engram reader, hash state) "
                    "can apply the next stage's Engram"
                )
            imported = first
        commits = rank == 0 or any(ratios[s] == 2 for s in own)
        raw.append((rank, first, stop, own, mirror, imported, commits))
    if raw[-1][6] and len(raw) > 1:
        raise ValueError(
            "V4.1: the last pipeline stage holds committed state; it samples and "
            "must not also run a stage commit"
        )
    plans = []
    for i, (rank, first, stop, own, mirror, imported, commits) in enumerate(raw):
        following = raw[i + 1] if i + 1 < len(raw) else None
        plans.append(
            StagePlan(
                rank=rank,
                stages=len(raw),
                first=first,
                stop=stop,
                own_sources=own,
                mirror_source=mirror,
                export_source=None if following is None else following[4],
                upstream_engram=imported,
                tail_engram=None if following is None else following[5],
                commits=commits,
            )
        )
    return tuple(plans)


def stage_keeps_weight(plan: StagePlan, name: str) -> bool:
    """Whether this stage loads the checkpoint tensor ``name`` (its module
    path, e.g. ``model.layers.14.engram.wkv.weight``), as far as the Engram
    placement decides it. On the stage that applies layer L's Engram at its
    end (``tail_engram`` = L) the model holds, at ``layers.L``, a module with
    that Engram only: of layer L's tensors it loads the Engram's and no other.
    On the stage whose first layer is L (``upstream_engram`` = L) layer L is
    built without an Engram: its Engram tensors are not loaded there. Names
    keep their checkpoint form, so everything keyed by name (quantized
    layouts, packed results, dense preparation) finds them as on one stage.
    Every other name: True (the loader's missing-layer rule decides)."""
    import re

    match = re.match(r"^(?:.*\.)?layers\.(\d+)\.(.+)$", name)
    if match is None:
        return True
    layer, rest = int(match.group(1)), match.group(2)
    if layer == plan.tail_engram:
        return rest.startswith("engram.")
    if layer == plan.upstream_engram and rest.startswith("engram."):
        return False
    return True
