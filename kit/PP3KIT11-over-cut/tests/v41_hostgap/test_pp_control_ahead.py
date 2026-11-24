# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP3/TP4 switch, ordering and byte preservation over real Gloo."""

from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed import parallel_state
from vllm.models.deepseek_v41.contract import StagePlan
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.v41_store_files import fingerprint_inputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu import v41_pp_handoff


CONTROL = "VLLM_V41_PP_CONTROL_AHEAD"
TIMING = "VLLM_V41_PP_HOSTGAP_TIMING"
KIT10_REPLAY = os.environ.get("K11_KIT10_REPLAY") == "1"


def test_boot_switches(monkeypatch):
    options = v41_pp_handoff.handoff_options
    monkeypatch.delenv(CONTROL, raising=False)
    monkeypatch.delenv(TIMING, raising=False)
    assert options(True) == v41_pp_handoff.HandoffOptions(control_ahead=True)
    with pytest.raises(ValueError, match="requires VLLM_V41_PP_LEAN_HANDOFF=1"):
        options(False)
    monkeypatch.setenv(CONTROL, "0")
    assert options(True) == options(False) == v41_pp_handoff.HandoffOptions()
    for name in (CONTROL, TIMING):
        for invalid in ("", "yes", "2", " 1"):
            monkeypatch.setenv(name, invalid)
            with pytest.raises(ValueError, match=name):
                options(True)
        monkeypatch.setenv(name, "1")
        with pytest.raises(ValueError, match="requires VLLM_V41_PP_LEAN_HANDOFF=1"):
            options(False)
        assert options(True) == v41_pp_handoff.HandoffOptions(
            control_ahead=name == CONTROL, timing=name == TIMING)
        monkeypatch.setenv(name, "0")


@pytest.mark.parametrize("value", ("0", "1"))
def test_transport_switches_preserve_store_environment(tmp_path, value):
    config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path)),
        speculative_config=None,
        load_config=SimpleNamespace(model_loader_extra_config={}),
        compute_hash=lambda: "same config",
    )
    adopted_path = Path(os.environ["K11_ADOPTED_ENV"])
    adopted = dict(line.split("=", 1) for line in adopted_path.read_text().splitlines())
    assert not any(name.startswith("VLLM_V41_MHC_") for name in adopted)
    assert CONTROL not in adopted and TIMING not in adopted
    baseline = fingerprint_inputs(config, {}, adopted)
    assert baseline["env"]["VLLM_V41_PP_LEAN_HANDOFF"] == "1"
    if not KIT10_REPLAY:
        assert fingerprint_inputs(config, {}, {**adopted, CONTROL: value, TIMING: value}) == baseline
    # Run this same test on the retained kit 10 and compare the resulting receipt.
    if value == "0" and os.environ.get("K11_FINGERPRINT_RECEIPT"):
        Path(os.environ["K11_FINGERPRINT_RECEIPT"]).write_text(
            json.dumps(baseline, sort_keys=True).replace(str(tmp_path), "<test-model>") + "\n"
        )


# The campaign's 14/14/12 cut and 5120-wide activation. Both StagePlan
# schemas are exercised, including the optional mirrored-cache fields.
_STEPS = (1, 6, 0, 4_092, 1, 0, 24, 6, 4_096)
_FIELDS = {
    "hidden": ((4, 5120), torch.int32, torch.float32),
    "pre": ((4,), torch.int32, torch.float32),
    "kv_rows": ((512,), torch.int16, torch.float16),
    "index_keys": ((128,), torch.int16, torch.float16),
}


def _payload(keys, rows, step, stage, rails):
    result = {}
    for name in keys:
        tail, integer, floating = _FIELDS[name]
        bits = torch.empty((rows, *tail), dtype=integer)
        width = bits.numel() // 4
        for rail in rails:
            shard = bits.view(-1)[width * rail:width * (rail + 1)]
            shard.fill_(step * 100 + stage * 10 + rail)
            shard[0] = -(1 << (8 * bits.element_size() - 1))
            if width > 1:
                shard[1] = 0x7FC00037 if integer == torch.int32 else 0x7E37
        result[name] = bits.view(floating)
    return result


def _stage_plan(stage):
    return StagePlan(
        rank=stage, stages=3, first=(0, 14, 28)[stage],
        stop=(14, 28, 40)[stage], own_sources=(),
        mirror_source=24 if stage == 2 else None,
        export_source=24 if stage == 1 else None,
        upstream_engram=None, tail_engram=None, commits=False,
    )


def _schedule(rows):
    if not rows:
        return SchedulerOutput.make_empty()
    counts = {"slot-0": min(rows, 6), "slot-1": rows - 6} if rows == 24 else {"slot-0": rows}
    slots = {request: index for index, request in enumerate(counts)}
    cached = replace(
        CachedRequestData.make_empty(),
        req_ids=list(counts),
        num_output_tokens=[1] * len(counts) if rows <= 24 else [0] * len(counts),
    )
    return replace(
        SchedulerOutput.make_empty(), scheduled_cached_reqs=cached,
        num_scheduled_tokens=counts, total_num_scheduled_tokens=rows, v41_slots=slots,
    )


def _run_pipeline(rank, rendezvous, mode, receipt_dir):
    from vllm.utils.system_utils import decorate_logs
    from vllm.v1.worker.gpu_worker import Worker

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=12,
                            timeout=timedelta(seconds=45))
    stage, rail = divmod(rank, 4)
    decorate_logs(f"Worker_PP{stage}_TP{rail}")
    ahead = mode in ("ahead", "default")
    if mode == "default" or KIT10_REPLAY:
        os.environ.pop(CONTROL, None)
    else:
        os.environ[CONTROL] = "1" if ahead else "0"
    if mode in ("ahead", "timing"):
        os.environ[TIMING] = "1"
    else:
        os.environ.pop(TIMING, None)
    digest = sha256()
    pp = parallel_state.GroupCoordinator(
        [[s * 4 + t for s in range(3)] for t in range(4)],
        rank, "gloo", False, group_name="l7_pipeline",
    )
    tp = parallel_state.GroupCoordinator(
        [list(range(s * 4, (s + 1) * 4)) for s in range(3)],
        rank, "gloo", True, group_name="l7_tensor",
    )
    parallel_state._PP, parallel_state._TP = pp, tp
    plan = _stage_plan(stage)
    worker = object.__new__(Worker)
    worker.profiler = None
    worker._v41_lean_used = False
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=6),
        parallel_config=SimpleNamespace(pipeline_parallel_size=3,
                                        distributed_executor_backend="mp"),
    )
    worker.use_v2_model_runner = True
    handoff_args = (pp, tp, plan, torch.device("cpu"), 5120, 4096)
    if KIT10_REPLAY:
        worker._v41_lean_handoff = v41_pp_handoff.LeanHandoff(*handoff_args)
    else:
        options = v41_pp_handoff.handoff_options(True)
        assert options.control_ahead == ahead
        assert options.timing == (mode in ("ahead", "timing"))
        worker._v41_lean_handoff = v41_pp_handoff.LeanHandoff(
            *handoff_args, options=options)

    def execute(schedule, incoming):
        rows = schedule.total_num_scheduled_tokens
        if not rows:
            assert incoming is None
            return ModelRunnerOutput([], {})
        if stage:
            assert incoming is not None
            actual = incoming.for_rows(rows)
            expected = _payload(plan.receives, rows, step, stage - 1, range(4))
            assert tuple(actual) == tuple(expected)
            for name, value in expected.items():
                actual_bytes = actual[name].view(torch.uint8)
                assert torch.equal(actual_bytes, value.view(torch.uint8))
                digest.update(actual_bytes.numpy().tobytes())
        if stage == 2:
            return None
        return IntermediateTensors(_payload(plan.sends, rows, step, stage, (rail,)))

    worker.model_runner = SimpleNamespace(execute_model=execute)
    for step, rows in enumerate(_STEPS + (6,) * 64):
        out = worker.execute_model(_schedule(rows))
        if rows:
            assert out is None
        else:
            assert isinstance(out, ModelRunnerOutput)
    worker._v41_lean_handoff.close()
    dist.barrier()
    if ahead and stage < 2:
        # Check ordering without waiting twice on a Gloo receive. This fault
        # is injected after the peer validates the early header but before
        # any CUDA payload could be submitted.
        schedule = _schedule(6)
        if stage == 0:
            handoff = v41_pp_handoff.LeanHandoff(
                pp, tp, plan, torch.device("cpu"), 5120, 4096,
                options=v41_pp_handoff.HandoffOptions(control_ahead=True),
            )
            worker._v41_lean_handoff = handoff

            def fail_in_forward(schedule, incoming):
                assert incoming is None
                mark = torch.empty(1, dtype=torch.int32)
                dist.recv(mark, src=rank + 4)
                assert mark.item() == 1
                raise RuntimeError("forward failed after early control")

            worker.model_runner.execute_model = fail_in_forward
            with pytest.raises(RuntimeError, match="forward failed after early control"):
                worker.execute_model(schedule)
            with pytest.raises(RuntimeError, match="hand-off has failed"):
                worker.execute_model(schedule)
            handoff.close()
        else:
            expected = v41_pp_handoff.ForwardStep.from_schedule(0, schedule, 4096)
            assert expected is not None
            header = torch.empty(3, dtype=torch.int64)
            dist.recv(header, src=rail, group=pp.cpu_group)
            expected.check_control(header)
            dist.send(torch.ones(1, dtype=torch.int32), dst=rail)
    dist.barrier()
    dist.destroy_process_group()
    if receipt_dir:
        (Path(receipt_dir) / f"{mode}-{rank}.sha256").write_text(digest.hexdigest() + "\n")


@pytest.mark.parametrize("mode", ("off", "timing", "ahead", "default"))
def test_worker_pp3_real_gloo(tmp_path: Path, mode: str):
    # The real CPU communicator performs the gather as well as PP transport.
    import vllm.platforms as platforms
    from vllm.platforms.cpu import CpuPlatform
    from vllm.v1.worker import gpu_worker  # noqa: F401

    assert dist.is_gloo_available()
    original = platforms.current_platform
    platforms.current_platform = CpuPlatform()
    try:
        torch.set_num_threads(1)
        mp.start_processes(
            _run_pipeline, args=((tmp_path / "gloo").as_uri(), mode, os.environ.get("K11_TRANSPORT_RECEIPTS")),
            nprocs=12, start_method="fork",
        )
    finally:
        platforms.current_platform = original
