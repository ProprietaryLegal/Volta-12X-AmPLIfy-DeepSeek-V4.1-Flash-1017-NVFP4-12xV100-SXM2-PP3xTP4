#!/usr/bin/env python3
"""allreduce_probe.py CARDS [SECONDS=420]: step 0 of the chunk-boundary question (PLAN section 9, "The chunk-boundary
reading"; owner's order of 2026-11-17). Four ranks on ONE board, no model: each rank holds its own random FP16
tensor and the four are summed through the engine's own entry, tensor_model_parallel_all_reduce, set up as a
worker sets it up (init_batch_invariance, set_custom_all_reduce(True), init_distributed_environment,
ensure_model_parallel_initialized(4, 1)) under the serving build's environment and inside a server configuration
whose ring provider is off as in the build's profile (kernel-config sm70_ring enabled: false). The probe REFUSES
to judge unless its tensor-parallel group dispatches exactly as the serving boot logs it: ['CUSTOM', 'PYNCCL']
(run/state/P3o2-S.log, 2026-11-17 10:03Z, group tp:0). Does a row's sum depend on how many rows are reduced with
it, or on where the row lies in the buffer?

  case SIZE    [8188, 5120] whole, against its first 8,124 rows reduced alone: rows 0 to 8,123 compared
  case SMALL   [819, 5120] whole (8,386,560 B, under the custom path's 8 MiB), against its first 755 rows alone
  case OFFSET  rows 64 to 8,187 of the big tensor reduced alone, against the same rows inside the whole: same
               size as SIZE's part, other offset in the buffer
Per call the backend the communicator recorded (custom, pynccl, ...). Per case: elements that differ, rows with a
difference, first and last such row. Beside them: the whole sum against the four ranks added in FP32 and rounded
once (what the custom path computes), and the same call twice (must be equal: the noise floor).

EXPECTED IF THE HYPOTHESIS IS RIGHT (written before the first run): SIZE different, SMALL equal, OFFSET
different; the big calls on pynccl, the small ones on custom; the big whole NOT equal to FP32-once, the small one
equal. If SIZE and OFFSET are both equal, the four-rank sum is not where a re-cut prompt read gets its other
rounding, and the hypothesis is refuted for this entry.

Random numbers only (seeded, the same on every rank); no model, no text. A rank that does not finish within the
limit is killed by its exact PID. Exit 0 the probe ran to its end (whatever it found), 1 it did not."""
import multiprocessing as mp
import os
import socket
import sys
import time

BIG, SMALL, COLS, SHIFT = 8188, 819, 5120, 64


def rank_main(r, n, port, q):
    try:
        import torch
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import parallel_state as ps
        from vllm.distributed.communication_op import tensor_model_parallel_all_reduce as all_reduce
        from vllm.model_executor.layers.batch_invariant import init_batch_invariance

        cfg = VllmConfig()
        cfg.kernel_config.sm70_ring.enabled = False       # the build's profile: kernel-config sm70_ring enabled: false
        ctx = set_current_vllm_config(cfg)
        ctx.__enter__()
        init_batch_invariance()
        ps.set_custom_all_reduce(True)                    # the profile: no-disable-custom-all-reduce
        ps.init_distributed_environment(n, r, f"tcp://127.0.0.1:{port}", r, "nccl")
        ps.ensure_model_parallel_initialized(n, 1)
        group = ps.get_tp_group()
        comm = group.device_communicator
        trace = comm._collective_trace
        trace.enabled = True
        dev = torch.device(f"cuda:{r}")
        # the dispatch order, by the communicator's own conditions (cuda_communicator.py, its "Using ... backends" line)
        order = []
        if comm.ring_comm.status["enabled"]:
            order.append("SM70_RING")
        if comm.host_reduce_comm is not None and comm.host_reduce_comm.status["enabled"]:
            order.append("HOST_REDUCE")
        if comm.qr_comm is not None and not comm.qr_comm.disabled:
            order.append("QUICK_REDUCE")
        if comm.fi_ar_comm is not None and not comm.fi_ar_comm.disabled:
            order.append("FLASHINFER")
        if comm.use_custom_allreduce and comm.ca_comm is not None and not comm.ca_comm.disabled:
            order.append("CUSTOM")
        if comm.symm_mem_comm is not None and not comm.symm_mem_comm.disabled:
            order.append("SYMM_MEM")
        if comm.pynccl_comm is not None and not comm.pynccl_comm.disabled:
            order.append("PYNCCL")
        ca = comm.ca_comm
        q.put((r, "INFO", f"group {group.unique_name} world {group.world_size}; dispatch order {order}; custom path "
                          f"limit {getattr(ca, 'dispatch_max_size', None)} B of {getattr(ca, 'max_size', None)}; long-prefill "
                          f"fused norm {comm.use_sm70_tp4_long_prefill_fused_norm}"))
        if order != ["CUSTOM", "PYNCCL"]:
            raise RuntimeError(f"this probe's group dispatches {order}, the serving boot's ['CUSTOM', 'PYNCCL']: it is "
                               "not the server's all-reduce and nothing is judged")
        # every rank builds all four ranks' tensors (same seeds), keeps its own for the sum and all for the reference
        parts = []
        for k in range(n):
            g = torch.Generator().manual_seed(20261117 + k)
            parts.append(torch.randn(BIG, COLS, generator=g).half().to(dev))
        mine = parts[r]
        ref = torch.zeros(BIG, COLS, dtype=torch.float32, device=dev)
        for p in parts:
            ref += p.float()
        ref = ref.half()                      # the four ranks added in FP32, rounded once
        del parts

        def reduce(x):
            trace.seen.clear()
            out = all_reduce(x.clone().contiguous()).clone()   # the big path may reduce in place: give it a copy
            torch.cuda.synchronize(dev)
            took = sorted({key[1] for key in trace.seen}) or ["(nothing recorded)"]
            return out, "+".join(took)

        def diff(a, b):
            ne = a != b
            rows = ne.any(dim=1).nonzero().flatten()
            n_el = int(ne.sum().item())
            if n_el == 0:
                return "EQUAL (0 elements differ)"
            return (f"DIFFERENT: {n_el} of {a.numel()} elements ({100.0 * n_el / a.numel():.2f} %), {rows.numel()} of "
                    f"{a.shape[0]} rows, first row {int(rows[0])}, last row {int(rows[-1])}")

        whole, b_whole = reduce(mine)
        again, _ = reduce(mine)
        part, b_part = reduce(mine[: BIG - SHIFT])
        tail, b_tail = reduce(mine[SHIFT:])
        s_whole, b_s_whole = reduce(mine[:SMALL])
        s_part, b_s_part = reduce(mine[: SMALL - SHIFT])
        lines = [
            f"backends: [{BIG}, {COLS}] {b_whole}; [{BIG - SHIFT}, {COLS}] rows 0.. {b_part}; [{BIG - SHIFT}, {COLS}] rows "
            f"{SHIFT}.. {b_tail}; [{SMALL}, {COLS}] {b_s_whole}; [{SMALL - SHIFT}, {COLS}] {b_s_part}",
            f"REPEAT  [{BIG}, {COLS}] twice: {diff(whole, again)}",
            f"SIZE    rows 0 to {BIG - SHIFT - 1}: whole against the first {BIG - SHIFT} rows alone: {diff(whole[: BIG - SHIFT], part)}",
            f"SMALL   rows 0 to {SMALL - SHIFT - 1}: whole [{SMALL}] against the first {SMALL - SHIFT} rows alone: {diff(s_whole[: SMALL - SHIFT], s_part)}",
            f"OFFSET  rows {SHIFT} to {BIG - 1}: inside the whole against reduced alone: {diff(whole[SHIFT:], tail)}",
            f"FP32-ONCE  [{BIG}, {COLS}] against the ranks added in FP32 and rounded once: {diff(whole, ref)}",
            f"FP32-ONCE  [{SMALL}, {COLS}]: {diff(s_whole, ref[:SMALL])}",
            f"checksum of the whole on this rank: {float(whole.float().sum().item()):.6e}",
        ]
        for line in lines:
            q.put((r, "LINE", line))
        del whole, again, part, tail, s_whole, s_part, ref, mine
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()
        torch.cuda.empty_cache()
        q.put((r, "DONE", ""))
    except Exception as e:  # reported by the parent; the probe then did not run to its end
        q.put((r, "ERROR", repr(e)[:600]))


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    cards = sys.argv[1]
    limit = float(sys.argv[2]) if len(sys.argv) == 3 else 420.0
    n = len(cards.split(","))
    if n != 4:
        print("allreduce_probe: four cards of one board, e.g. 0,1,2,3")
        sys.exit(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cards
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps_ = [ctx.Process(target=rank_main, args=(r, n, port, q)) for r in range(n)]
    for p in ps_:
        p.start()
    state = {r: "started" for r in range(n)}
    out = {r: [] for r in range(n)}
    end = time.time() + limit
    while time.time() < end and not all(v in ("DONE", "ERROR") for v in state.values()):
        try:
            r, kind, text = q.get(timeout=1.0)
        except Exception:
            if all(not p.is_alive() for p in ps_):
                break
            continue
        if kind in ("DONE", "ERROR"):
            state[r] = kind
        if text:
            out[r].append(f"{kind} {text}")
    for r in range(n):
        print(f"rank {r} card {cards.split(',')[r]} pid {ps_[r].pid}: {state[r]}", flush=True)
    for line in out[0]:
        print(f"rank 0: {line}", flush=True)
    for r in range(1, n):
        for line in out[r]:
            if line.startswith(("ERROR", "INFO")):
                print(f"rank {r}: {line}", flush=True)
        print(f"rank {r}: its lines {'are the same as rank 0\'s' if [x for x in out[r] if x.startswith('LINE')] == [x for x in out[0] if x.startswith('LINE')] else 'DIFFER from rank 0\'s: ' + ' | '.join(x for x in out[r] if x.startswith('LINE'))[:1500]}", flush=True)
    for p in ps_:
        p.join(20)
        if p.is_alive():
            p.terminate(); p.join(5)
            if p.is_alive():
                p.kill(); p.join(5)
    ok = all(v == "DONE" for v in state.values())
    print("ALLREDUCE PROBE", "RAN TO ITS END" if ok else "DID NOT FINISH", cards, flush=True)
    sys.exit(0 if ok else 1)
