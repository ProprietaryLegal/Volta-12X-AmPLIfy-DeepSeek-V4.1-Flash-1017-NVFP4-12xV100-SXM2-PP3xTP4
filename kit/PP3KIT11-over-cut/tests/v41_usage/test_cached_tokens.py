# SPDX-License-Identifier: Apache-2.0
"""Real scheduler and prefix pool, with the CPU test supplying a sampled token."""

import asyncio
import copy
import json
import logging
import time

import pytest

import torch
from transformers import LlamaConfig

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_manager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched import scheduler as scheduler_module
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

BLOCK_SIZE = 16
PUBLIC_PROMPT = b"We the People of the United States, in Order to form a more perfect Union. "


@pytest.fixture
def scheduler_replies(tmp_path, monkeypatch, caplog):
    # Use the CPU fixture pattern of test_v41_multi_seq_engine_loop_slots.
    # Only the serving configuration check (DSpark, GPU runner) is bypassed.
    monkeypatch.setenv("VLLM_V41_MULTI_SEQ", "1")
    monkeypatch.setenv("VLLM_V41_PREFIX_LOG", "1")
    monkeypatch.setenv("VLLM_V41_SNAPSHOT_HIT", "1")
    monkeypatch.setattr(kv_cache_manager, "_V41_SESSION_PREFIX", True)
    monkeypatch.setattr(
        scheduler_module, "multi_seq_decision", lambda config: (True, "CPU test")
    )
    LlamaConfig().save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), skip_tokenizer_init=True, max_model_len=1024,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=512, max_model_len=1024,
            enable_chunked_prefill=True, async_scheduling=False, is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(block_size=BLOCK_SIZE, enable_prefix_caching=True),
    )
    config.cache_config.num_gpu_blocks = 64
    cache = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], FullAttentionSpec(
            block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32,
        ))],
    )
    scheduler = scheduler_module.Scheduler(
        vllm_config=config, kv_cache_config=cache,
        structured_output_manager=StructuredOutputManager(config),
        block_size=BLOCK_SIZE, log_stats=True,
    )
    assert scheduler.v41_multi_seq is not None
    assert scheduler.v41_multi_seq.slots.snapshot_hits
    init_none_hash(sha256)
    hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
    prompt = list((PUBLIC_PROMPT * 4)[:257])
    processor = OutputProcessor(None, log_stats=False)
    replies = []
    for request_id, expected_hit in (("cold", 0), ("resend", 256)):
        core_request = EngineCoreRequest(
            request_id=request_id, prompt_token_ids=prompt.copy(),
            sampling_params=SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False),
            pooling_params=None, mm_features=None, arrival_time=time.monotonic(),
            lora_request=None, cache_salt=None, data_parallel_rank=None,
            external_req_id=request_id,
        )
        processor.add_request(core_request, prompt=None)
        request = Request.from_engine_core_request(core_request, hasher)
        scheduler.add_request(request)
        assert request.num_computed_tokens == 0
        with caplog.at_level(logging.INFO, logger="vllm.v1.core.sched.v41_multi_seq"):
            scheduled = scheduler.schedule()
        (admitted,) = scheduled.scheduled_new_reqs
        assert admitted.num_computed_tokens == expected_hit
        assert scheduled.num_scheduled_tokens == {request_id: len(prompt) - expected_hit}
        assert request.prefill_stats is not None
        assert request.prefill_stats.num_cached_tokens == expected_hit
        assert request.prefill_stats.num_local_cached_tokens == expected_hit
        assert request.prefill_stats.num_external_cached_tokens == 0
        assert request.prefill_stats.num_prompt_tokens == len(prompt)
        (line,) = (line for line in caplog.messages
                    if line.startswith(f"VLLM_V41_PREFIX_LOG request {request_id} "))
        decision = "hit" if expected_hit else "cold"
        assert f" hit {expected_hit} decision {decision} " in line

        outputs = scheduler.update_from_output(
            scheduled,
            ModelRunnerOutput(
                req_ids=[request_id], req_id_to_index={request_id: 0},
                sampled_token_ids=[[ord(".")]], logprobs=None,
                prompt_logprobs_dict={}, pooler_output=[],
            ),
        )
        (output,) = outputs[0].outputs
        assert output.prefill_stats is not None
        assert output.prefill_stats.num_cached_tokens == expected_hit
        assert request.is_finished() and not scheduler.running
        (reply,) = processor.process_outputs(
            [output], engine_core_timestamp=time.monotonic()
        ).request_outputs
        assert isinstance(reply, RequestOutput)
        assert reply.num_cached_tokens == expected_hit
        print(f"{request_id}: scheduled and logged {expected_hit}, "
              f"prefill and RequestOutput {reply.num_cached_tokens}, prompt {len(prompt)}")
        replies.append((reply, expected_hit))
    return replies


class CPUAnthropicServing(AnthropicServingMessages):
    """Feed a CPU RequestOutput into the real serving generators."""

    def __init__(self, output: RequestOutput, enabled: bool):
        self.output = output
        self.enable_prompt_tokens_details = enabled
        self.enable_force_include_usage = False
        self.enable_log_outputs = False
        self.request_logger = None
        self.use_harmony = False
        self.enable_auto_tools = False
        self.tool_parser = None
        self.parser_cls = None
        self.response_role = "assistant"
        self.tool_call_id_type = "random"
        self.system_fingerprint = None
        self.return_tokens_as_token_ids = False
        self.stop_reason_map = {"stop": "end_turn", "length": "max_tokens"}

    async def create_chat_completion(self, request, raw_request=None):
        async def sampled():
            yield self.output

        args = (
            request, sampled(), self.output.request_id, "cpu-model", [], None,
            RequestResponseMetadata(request_id=self.output.request_id),
        )
        if request.stream:
            return self.chat_completion_stream_generator(*args)
        return await self.chat_completion_full_generator(*args)


def stream_data(sse):
    return json.loads(sse.split("data: ", 1)[1])


def test_scheduler_prefix_count(scheduler_replies):
    assert [hit for _, hit in scheduler_replies] == [0, 256]


@pytest.mark.parametrize("stream", [False, True], ids=["full", "sse"])
@pytest.mark.parametrize("expected_hit", [256, 0], ids=["hit", "cold"])
@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_anthropic_cache_usage(scheduler_replies, stream, expected_hit, enabled):
    reply, log_hit = next(row for row in scheduler_replies if row[1] == expected_hit)
    serving = CPUAnthropicServing(reply, enabled)
    request = AnthropicMessagesRequest(
        model="cpu-model", max_tokens=1, stream=stream,
        messages=[{"role": "user", "content": PUBLIC_PROMPT.decode() * 4}],
    )

    async def collect():
        response = await serving.create_messages(request)
        if not stream:
            return [("message", response.model_dump(exclude_none=True)["usage"], 1)]
        events = [stream_data(event) async for event in response]
        assert events[-1]["type"] == "message_stop"
        (start,) = [event for event in events if event["type"] == "message_start"]
        (delta,) = [event for event in events if event["type"] == "message_delta"]
        return [("message_start", start["message"]["usage"], 0),
                ("message_delta", delta["usage"], 1)]

    whole_prompt = len(reply.prompt_token_ids)
    for event, usage, output_count in asyncio.run(collect()):
        if enabled:
            assert usage.get("cache_read_input_tokens") == log_hit, (event, usage)
            assert usage["input_tokens"] + usage["cache_read_input_tokens"] == whole_prompt
        else:
            assert "cache_read_input_tokens" not in usage
            assert usage["input_tokens"] == whole_prompt
        assert usage["output_tokens"] == output_count


@pytest.mark.parametrize("stream", [False, True], ids=["full", "sse"])
@pytest.mark.parametrize("count", [None, 0, 256])
@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_openai_chat_cache_usage(scheduler_replies, stream, count, enabled):
    output = copy.copy(scheduler_replies[1][0])
    output.num_cached_tokens = count
    serving = CPUAnthropicServing(output, enabled)
    request = ChatCompletionRequest(
        model="cpu-model", stream=stream,
        stream_options={"include_usage": True} if stream else None,
        messages=[{"role": "user", "content": PUBLIC_PROMPT.decode() * 4}],
    )

    async def collect():
        response = await serving.create_chat_completion(request)
        if not stream:
            return response.usage.model_dump(exclude_none=True)
        events = [event async for event in response]
        assert events.pop() == "data: [DONE]\n\n"
        (usage,) = [chunk["usage"] for chunk in map(stream_data, events)
                    if not chunk["choices"]]
        return usage

    usage = asyncio.run(collect())
    assert usage["prompt_tokens"] == len(output.prompt_token_ids)
    assert usage["completion_tokens"] == 1
    if enabled:
        assert usage["prompt_tokens_details"]["cached_tokens"] == (count or 0)
    else:
        assert "prompt_tokens_details" not in usage
