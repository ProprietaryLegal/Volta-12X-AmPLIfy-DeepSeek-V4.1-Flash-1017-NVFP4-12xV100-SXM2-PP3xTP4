"""tools/pipe_witness.py on log lines in the server's own form.

The line prefixes and the first-skipped-step line are copied from window s4's server log (run/state/P3s4-F4c.log,
numbers only). The two active lines have never been printed by a server: their text is the engine's own, as its CPU
tests assert it word for word (engine branch pp3-12card-pipe: tests/models/test_deepseek_v41_mid_prefill_skip.py,
test_construction_on_each_stage; tests/v1/core/test_v41_prompt_end_hold.py runs the scheduler that logs the other).
Expected verdicts come from the tool's rule, not from its output.
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("pipe_witness", Path(__file__).resolve().parents[1] / "tools" / "pipe_witness.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)

BOOT = [
    "(EngineCore pid=66845) INFO 11-17 13:33:05 [v41_multi_seq.py:1205] VLLM_V41_PREFIX_LOG=1 active (scheduler): one line per admitted request",
    "(Worker_PP1_TP0 pid=70960) INFO 11-17 13:34:55 [gguf_loader.py:717] PP stage 1/3 loads 351 of 1008 GGUF tensors",
    "(Worker_PP2_TP0 pid=70964) INFO 11-17 13:34:55 [model.py:1047] VLLM_V41_PREFILL_SKIP_ROWS=3072 active: prefill steps ending >= 3072 rows before the prompt end skip layers 28-39 on this stage.",
]
SKIP2 = "(Worker_PP2_TP0 pid=70964) INFO 11-17 13:36:37 [model.py:1606] VLLM_V41_PREFILL_SKIP_ROWS: first skipped prefill step (layers > 20)."
SKIP1 = SKIP2.replace("Worker_PP2_TP0 pid=70964", "Worker_PP1_TP0 pid=70960")


def mid_line(t, proc=None):
    return (f"({proc or f'Worker_PP1_TP{t}'} pid=7096{t}) INFO 11-17 13:34:40 [model.py:1668] VLLM_V41_PP_MID_PREFILL_SKIP=1 active "
            f"(stage 1 of 3, TP rank {t}): prefill steps ending >= 3072 rows before the prompt end stop after layer 20's "
            "attention on this stage too (layers 21-27 are skipped, their window rings zeroed; the hand-off carries the rows "
            "of KV source 20).")


def hold_line(n, proc="EngineCore"):
    return (f"({proc} pid=66845) INFO 11-17 13:33:05 [v41_multi_seq.py:1608] VLLM_V41_MULTI_SEQ_PROMPT_END_HOLD={n} active "
            f"(scheduler): the step that ends a request's prompt opens {n} decode step(s) for it; while a running request "
            "has opening steps left, no prompt chunk of a request that is already running is scheduled. Admissions and "
            "decode steps are scheduled as before.")


def turn_sched(n, proc="EngineCore"):
    return (f"({proc} pid=66845) INFO 11-17 13:33:05 [v41_multi_seq.py:1690] VLLM_V41_PP_TURN_CHUNKS={n} active (scheduler): "
            f"a request's last min(rows, 3072) prompt rows are read in {n} chunk(s) of a multiple of 64 rows, the rows before "
            "them in chunks of at most 4092 ending 3072 rows before the prompt's end; a request in those last rows reads "
            "before a request that is not; a request with fewer than 2048 rows to read is cut as before.")


def turn_worker(n, t, stage=1):
    return (f"(Worker_PP{stage}_TP{t} pid=7096{t}) INFO 11-17 13:36:30 [gpu_worker.py:1492] VLLM_V41_PP_TURN_CHUNKS={n} active "
            "(worker): this stage awaits its pipeline hand-off before its next step.")


def verdict(lines, mid, hold, after_read=False, turn=0):
    fails, unread, _ = W.read(lines, mid, hold, after_read, turn)
    return "FAIL" if fails else "INCONCLUSIVE" if unread else "PASS"


def test_both_off():
    assert verdict(BOOT, 0, 0) == "PASS"
    assert verdict(BOOT + [mid_line(0)], 0, 0) == "FAIL"       # a boot without the switch must not name it
    assert verdict(BOOT + [hold_line(128)], 0, 0) == "FAIL"


def test_mid_on_needs_one_line_from_each_middle_worker():
    on = BOOT + [mid_line(t) for t in range(4)]
    assert verdict(on, 1, 0) == "PASS"
    assert verdict(BOOT + [mid_line(t) for t in range(3)], 1, 0) == "FAIL"            # a worker without it
    assert verdict(on + [mid_line(2)], 1, 0) == "FAIL"                                # one twice
    assert verdict(on + [mid_line(1, proc="Worker_PP2_TP1")], 1, 0) == "FAIL"         # a worker of another stage
    assert verdict(BOOT + [mid_line(t).replace(f"TP rank {t}", "TP rank 0") for t in range(4)], 1, 0) == "FAIL"
    assert verdict(BOOT, 1, 0) == "FAIL"


def test_a_line_glued_behind_another_processes_unfinished_line_is_its_writers():
    glued = "Capturing CUDA graphs (FULL):  50%|     | 3/6 [00:01<00:01,  2.9it/s]" + mid_line(3)
    assert verdict(BOOT + [mid_line(t) for t in range(3)] + [glued], 1, 0) == "PASS"


def test_hold_on_needs_the_engine_cores_line_with_its_count():
    assert verdict(BOOT + [hold_line(128)], 0, 128) == "PASS"
    assert verdict(BOOT + [hold_line(64)], 0, 128) == "FAIL"
    assert verdict(BOOT + [hold_line(128), hold_line(128)], 0, 128) == "FAIL"
    assert verdict(BOOT + [hold_line(128, proc="Worker_PP0_TP0")], 0, 128) == "FAIL"
    assert verdict(BOOT, 0, 128) == "FAIL"
    assert verdict(BOOT + [mid_line(t) for t in range(4)] + [hold_line(128)], 1, 128) == "PASS"


def test_after_a_read():
    assert verdict(BOOT + [SKIP2], 0, 0, True) == "PASS"
    assert verdict(BOOT + [SKIP2, SKIP1], 0, 0, True) == "FAIL"       # the middle stage skipped without the switch
    on = BOOT + [mid_line(t) for t in range(4)]
    assert verdict(on + [SKIP2, SKIP1], 1, 0, True) == "PASS"
    assert verdict(on + [SKIP2], 1, 0, True) == "FAIL"                # the switch is on and the middle stage did not skip
    assert verdict(on, 1, 0, True) == "INCONCLUSIVE"                  # no prompt with skipped chunks was read
    assert verdict(on + [SKIP2, SKIP2, SKIP1], 1, 0, True) == "FAIL"


def test_not_a_boots_log():
    assert verdict(["nothing of a server"], 0, 0) == "INCONCLUSIVE"
    assert verdict([BOOT[0]], 0, 0) == "INCONCLUSIVE"                 # no middle-stage worker's line


def test_turn_chunks_need_the_schedulers_line_and_after_a_read_the_middle_workers():
    """The lines' text is the engine's (v41_multi_seq.py and gpu_worker.py at c3ea24c and after); no server has
    printed them yet."""
    workers = [turn_worker(3, t) for t in range(4)]
    assert verdict(BOOT + [turn_sched(3)], 0, 0, turn=3) == "PASS"                      # before any read: no worker line yet
    assert verdict(BOOT + [turn_sched(3)] + workers[:2], 0, 0, turn=3) == "PASS"        # some workers have handed off
    assert verdict(BOOT + [turn_sched(3), SKIP2] + workers, 0, 0, after_read=True, turn=3) == "PASS"
    assert verdict(BOOT + [turn_sched(3), SKIP2] + workers[:3], 0, 0, after_read=True, turn=3) == "FAIL"   # one worker silent after a read
    assert verdict(BOOT + [turn_sched(3), SKIP2], 0, 0, after_read=True, turn=3) == "FAIL"   # no worker awaits: the buffer is unguarded
    assert verdict(BOOT + [turn_sched(2), SKIP2] + workers, 0, 0, after_read=True, turn=3) == "FAIL"       # another n
    assert verdict(BOOT + [turn_sched(3), SKIP2] + [turn_worker(2, t) for t in range(4)], 0, 0, after_read=True, turn=3) == "FAIL"
    assert verdict(BOOT + [turn_sched(3), turn_sched(3)], 0, 0, turn=3) == "FAIL"       # doubled
    assert verdict(BOOT + [turn_sched(3, proc="Worker_PP0_TP0")], 0, 0, turn=3) == "FAIL"
    assert verdict(BOOT + [turn_sched(3), SKIP2] + workers + [turn_worker(3, 0, stage=0)], 0, 0, after_read=True, turn=3) == "FAIL"
    assert verdict(BOOT, 0, 0, turn=3) == "FAIL"                                        # the switch asked, no line
    # a boot without the switch holds no line that names it
    assert verdict(BOOT + [turn_sched(3)], 0, 0) == "FAIL"
    assert verdict(BOOT + workers[:1], 0, 0) == "FAIL"
    # with the other two switches
    both = BOOT + [mid_line(t) for t in range(4)] + [hold_line(128), turn_sched(3), SKIP2, SKIP1] + workers
    assert verdict(both, 1, 128, after_read=True, turn=3) == "PASS"
    assert verdict(both, 1, 128, after_read=True) == "FAIL"


def test_usage_of_turn(capsys, tmp_path):
    log = tmp_path / "x.log"
    log.write_text("\n".join(BOOT + [turn_sched(2)]) + "\n")
    assert W.main([str(log), "--mid", "0", "--hold", "0", "--turn", "2"]) == 0
    assert "PIPE WITNESS mid=0 hold=0 turn=2: PASS" in capsys.readouterr().out
    assert W.main([str(log), "--mid", "0", "--hold", "0"]) == 1
