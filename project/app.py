import os
import asyncio
import torch
from typing import Union, Literal, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.version import __version__ as VLLM_VERSION
from project.pipeline import load_pretrained_tokenizer
from project.logger import get_logger

logger = get_logger(__name__)


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


# Create API
app = FastAPI(default_response_class=ORJSONResponse)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Load vllm engine
engine = LLM(
    model=os.getenv("MODEL_NAME"),
    quantization="bitsandbytes",
    dtype="half",
    tensor_parallel_size=torch.cuda.device_count(),
    trust_remote_code=True,
    gpu_memory_utilization=0.8,  # Réduisez à 0.7-0.8
    enforce_eager=True,  # Évite les optimisations qui peuvent crasher
    disable_custom_all_reduce=True,  # Plus stable
    disable_log_stats=False,
)
logger.info(f"✅ VLLM engine (version {VLLM_VERSION}) loaded!")

# Load tokenizer
tokenizer = load_pretrained_tokenizer(model_name=os.getenv("MODEL_NAME"))

# Thread pool to avoid blocking
executor = ThreadPoolExecutor(max_workers=2)

@app.get("/")
async def root():
    """Home route"""
    return {"message": "LLM inference API (sync vLLM)"}


@app.post("/generate")
async def generate(request_data: RequestData) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompts: the list of texts to use for the generation (str or list of chat message).
    - use_chatml: whether to use chatml format for prompts
    - use_stream: whether to stream the results or not.
    - sampling_params: the sampling parameters.
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            _generate_batch,
            request_data.prompts,
            request_data.use_chatml,
            request_data.sampling_params,
        )
        return {"completion": result}
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_batch(
    prompts: list[Union[str, list[ChatMessage]]],
    use_chatml: bool,
    sampling_params: Dict[str, Any],
) -> list[str]:
    logger.info(f"🔄 Running sync generation on {len(prompts)} prompts (chatml={use_chatml})")

    # Format prompts
    formatted_prompts = [
        (
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            if isinstance(p, list) and use_chatml
            else p
        )
        for p in prompts
    ]

    # Create SamplingParams
    full_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        seed=0,
        skip_special_tokens=True,
        **sampling_params,
    )
    outputs = engine.generate(formatted_prompts, full_params)
    completions = [output.outputs[0].text for output in outputs]

    logger.info(f"✅ Completed {len(completions)} prompts.")
    return completions
