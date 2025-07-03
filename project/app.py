import os
import sys
import json
import asyncio
import torch
from typing import Union, Literal, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from collections.abc import AsyncGenerator
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.utils import with_cancellation
from vllm.utils import random_uuid
from vllm.version import __version__ as VLLM_VERSION
from project.pipeline import load_pretrained_tokenizer


# Create API
app = FastAPI(default_response_class=ORJSONResponse)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.state.enable_server_load_tracking = False  # <-- Fix crash from vLLM utils

# Load vllm engine
engine_args = AsyncEngineArgs(
    model=os.getenv("MODEL_NAME"),
    quantization="bitsandbytes",
    dtype="half",
    tensor_parallel_size=torch.cuda.device_count(),
    trust_remote_code=True,
    gpu_memory_utilization=0.8,  # Réduisez à 0.7-0.8
    # enforce_eager=True,  # Évite les optimisations qui peuvent crasher
    disable_custom_all_reduce=True,  # Plus stable
)
engine = AsyncLLMEngine.from_engine_args(engine_args)
print(f"✅ VLLM engine (version {VLLM_VERSION}) loaded with args {engine_args}")

# Load tokenizer
tokenizer = load_pretrained_tokenizer(model_name=os.getenv("MODEL_NAME"))


# Chat message schema
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


# Generate request schema
class RequestData(BaseModel):
    prompts: list[Union[str, list[ChatMessage]]]
    use_chatml: bool = False
    use_stream: bool = False
    sampling_params: Dict[str, Any] = {}


@app.get("/")
async def root():
    """Home route"""
    return ORJSONResponse({"message": "LLM inference API"})


@app.post("/generate")
async def generate(request_data: RequestData, request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompts: the list of texts to use for the generation (str or list of chat message).
    - use_chatml: whether to use chatml format for prompts
    - use_stream: whether to stream the results or not.
    - sampling_params: the sampling parameters.
    """
    return await _generate(request_data=request_data, raw_request=request)


@with_cancellation
async def _generate(request_data: RequestData, raw_request: Request) -> Response:
    prompts = request_data.prompts
    use_chatml = request_data.use_chatml
    use_stream = request_data.use_stream
    sampling_params = request_data.sampling_params

    assert isinstance(prompts, list)
    print(f"Start generate n_prompts={len(prompts)}, use_chatml={use_chatml}, use_stream={use_stream}")

    # Format prompts using chat template if needed #TODO: external func ?
    formatted_prompts = [
        (
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            if isinstance(p, list) and use_chatml
            else p
        )
        for p in prompts
    ]

    # Setup sampling params
    full_sampling_params = {
        "seed": 0,
        "skip_special_tokens": True,
        "max_tokens": 1024,
        "temperature": 0,
        **sampling_params,
    }
    print(f"SamplingParams={full_sampling_params}")

    # Streaming mode (one prompt at a time)
    async def generate_stream() -> AsyncGenerator[bytes, None]:
        yield b'{"completion": ['  # start the array
        for i, prompt in enumerate(formatted_prompts):
            request_id = random_uuid()
            final_output = None
            async for output in engine.generate(prompt, SamplingParams(**full_sampling_params), request_id):
                final_output = output
            if final_output and final_output.outputs:
                print(f"Completion done for request_id={request_id} (index={i})")
                text = json.dumps(final_output.outputs[0].text)  # ensure proper escaping
                if i > 0:
                    yield b","  # Add comma before each item except first
                yield text.encode("utf-8")  # Stream the actual string

        yield b"]}"  # Close array and object

    if use_stream:
        return StreamingResponse(generate_stream(), media_type="application/json")

    # Non streaming mode (batched prompts)
    async def generate_one(prompt: str) -> str:
        request_id = random_uuid()
        final_output = None
        try:
            async for output in engine.generate(prompt, SamplingParams(**full_sampling_params), request_id):
                final_output = output
        except asyncio.CancelledError:
            print(f"[CANCELLED] Generation was cancelled for request_id={request_id}")
            return Response(status_code=499)

        assert final_output is not None
        print(f"Completion done for request_id={request_id}")
        return final_output.outputs[0].text

    outputs = []
    for prompt in formatted_prompts:
        text = await generate_one(prompt)
        outputs.append(text)
    results = {"completion": outputs}
    print(f"/generate response size: {sys.getsizeof(results) / 1024:.2f} KB")
    return results
