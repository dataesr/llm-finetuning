import os
import json
import asyncio
import importlib
import torch
from uuid import uuid4
import time
from typing import Literal, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status as fastapi_status
from fastapi.responses import ORJSONResponse, JSONResponse, Response
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.version import __version__ as VLLM_VERSION
from project.model.config import model_get_config, model_get_instruction
from project.logger import get_logger

logger = get_logger(__name__)

FOLDER = "completions"

class RequestData(BaseModel):
    """Generate request inputs"""
    prompts: list[str]
    chat_template_params: Dict[str, Any] = {}
    sampling_params: Dict[str, Any] = {}

class Task(BaseModel):
    """Generate task class"""
    status: Literal["queued", "running", "done", "error"]
    completions: Optional[list[str]] = None
    error: Optional[str] = None
    queued_at: float
    running_at: Optional[float] = None
    done_at: Optional[float] = None

class TaskStore:
    """Generate task store class"""

    def __init__(self):
        self._store: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
        self._save_on_dir = FOLDER
        if not os.path.isdir(FOLDER):
            logger.warning("⚠️ TaskStore saving directory not found: saving to disk disabled")
            self._save_on_dir = None

    async def _update(self, task_id: str, **kwargs):
        async with self._lock:
            task = self._store.get(task_id)
            if not task:
                raise KeyError(f"Task {task_id} not found")
            updated = task.model_copy(update=kwargs)
            self._store[task_id] = updated

    async def create(self) -> str:
        async with self._lock:
            task_id = str(uuid4())
            self._store[task_id] = Task(
                status="queued",
                queued_at=time.time(),
            )
            return task_id

    async def set_error(self, task_id: str, error: str):
        await self._update(task_id, status="error", error=error)

    async def set_running(self, task_id: str):
        await self._update(task_id, status="running", running_at=time.time())

    async def set_done(self, task_id: str, completions: list):
        await self._update(task_id, status="done", done_at=time.time(), completions=completions)
        if self._save_on_dir:
            file_path = f"{self._save_on_dir}/{task_id}.json"
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(completions, file)
            logger.debug(f"{os.listdir(FOLDER)}")
            logger.debug(f"💾 Task {task_id} completions saved to {file_path}")

    async def get(self, task_id: str) -> Task:
        async with self._lock:
            task = self._store.get(task_id)
            if not task:
                raise KeyError(f"Task {task_id} not found")
            return task

    # async def cleanup(self, older_than_secs: int = 60 * 10):
    #     async with self._lock:
    #         now = time.time()
    #         expired = [tid for tid, t in self._store.items() if t.done_at and (now - t.done_at) > older_than_secs]
    #         for tid in expired:
    #             del self._store[tid]
    #             logger.debug(f"🗑️ Task {tid} expired and removed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🟧 Initializing application...")

    model_name = os.getenv("MODEL_NAME")
    assert model_name is not None

    # Get model pipeline
    config = model_get_config(model_name)
    app.state.pipeline = importlib.import_module(f"project.pipeline.{config}")

    # Get model instruction
    app.state.model_instruction = model_get_instruction(model_name)

    # Initialize vllm engine
    app.state.engine = LLM(
        model=model_name,
        quantization="bitsandbytes",
        dtype="half",
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        disable_log_stats=False,
    )
    logger.info(f"✅ vLLM engine {VLLM_VERSION} loaded")
    app.state.engine_lock = asyncio.Lock()

    # Initialize tokenizer
    tokenizer = get_tokenizer(model_name, trust_remote_code=True)
    if not tokenizer:
        logger.error(f"❌ Tokenizer not loaded!")
    if not tokenizer.chat_template:
        logger.debug(f"No chat template found for {model_name} tokenizer, applying default..")
        tokenizer.chat_template = app.state.pipeline.default_chat_template
    app.state.tokenizer = tokenizer
    logger.info(f"✅ Tokenizer loaded")

    # Initialize task store
    app.state.task_store = TaskStore()

    # Initialize cleaning function
    # async def cleanup_task_store():
    #     while True:
    #         await app.state.task_store.cleanup(older_than_secs=600)
    #         await asyncio.sleep(60)

    # asyncio.create_task(cleanup_task_store())
    logger.info(f"✅ Task store initialized")

    yield

    # Optional cleanup logic here
    logger.info("🟥 Shutting down app.")


# Create API
app = FastAPI(lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/")
async def root() -> JSONResponse:
    """Application default route"""
    return JSONResponse(content={"application": "LLM inference API (sync vLLM)"})


@app.get("/health")
async def health() -> JSONResponse:
    """Application health route"""
    try:
        if app.state.engine and app.state.tokenizer and app.state.task_store:
            return JSONResponse(
                content={"status": "healthy"},
                status_code=fastapi_status.HTTP_200_OK,
            )
    except Exception:
        return JSONResponse(
            content={"status": "unhealthy"},
            status_code=fastapi_status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@app.post("/generate")
async def generate(request_data: RequestData) -> JSONResponse:
    """Create a generation task for the request.

    The request should be a JSON object with the following fields:
    - prompts: the list of texts to use for the generation
    - sampling_params: the sampling parameters.

    It returns a JSON object:
    - task_id: task unique identifier
    - status: task current status
    """
    task_id = await app.state.task_store.create()

    async def generate_task(task_id: str, request_data: RequestData):
        try:
            logger.info(f"➡️ New generation task {task_id} added ({len(request_data.prompts)} prompts)")
            if request_data.sampling_params:
                logger.debug(f"Generation with custom params: {request_data.sampling_params}")

            async with app.state.engine_lock:
                await app.state.task_store.set_running(task_id)
                completions = await asyncio.to_thread(
                    _generate,
                    app.state.engine,
                    app.state.tokenizer,
                    request_data.prompts,
                    request_data.chat_template_params,
                    request_data.sampling_params,
                )
            await app.state.task_store.set_done(task_id, completions=completions)
            logger.info(f"✅ Generation task {task_id} done")
        except Exception as e:
            await app.state.task_store.set_error(task_id, error=str(e))
            logger.error(f"❌ Generation task {task_id} failed: {e}")

    # Create task
    asyncio.create_task(generate_task(task_id, request_data))

    # Return task id
    task_data = await app.state.task_store.get(task_id)
    return JSONResponse({"task_id": task_id, "status": task_data.status})


# Get task status and completions
@app.get("/generate/{task_id}")
async def get_generate_task(task_id: str) -> ORJSONResponse:
    """
    Get a generation task data.

    The request route should contains:
    - task_id: task identifier

    It returns a ORJSON object with the following fields:
    - task_id: task identifier
    - status: task status
    - completions: list of generated completions
    - error: error message
    - queued_at: task creation time
    - running_at: task started time
    - done_at: task done time
    """
    try:
        task = await app.state.task_store.get(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return ORJSONResponse(content=task.model_dump())


def _apply_chat_template(tokenizer, prompts: list[str], chat_template_params: Dict[str, Any]) -> list[str]:
    logger.debug(f"Formatting {len(prompts)} prompts")
    if chat_template_params:
        logger.debug(f"Custom formatting params: {chat_template_params}")

    # Get instruction param
    instruction = chat_template_params.pop("instruction") if "instruction" in chat_template_params else None
    instruction = instruction or app.state.model_instruction

    # Format prompts
    formatted_prompts = [
        tokenizer.apply_chat_template(
            app.state.pipeline.construct_one_conversation(system=instruction, user=prompt),
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_params,
        )
        for prompt in prompts
    ]
    logger.debug(f"Formatted prompts: {formatted_prompts[0]}")

    return formatted_prompts


def _generate(
    engine: LLM, tokenizer, prompts: list[str], chat_template_params: Dict[str, Any], sampling_params: Dict[str, Any]
) -> list[str]:
    logger.debug(f"Running generation on {len(prompts)} prompts...")

    # Format prompts
    formatted_prompts = _apply_chat_template(tokenizer, prompts, chat_template_params=chat_template_params)

    # Sampling params
    max_length = tokenizer.model_max_length
    truncate_length = max_length if isinstance(max_length, int) and max_length < 100_000 else 8192
    full_params = {
        "seed": 0,
        "temperature": 0,
        "max_tokens": 1024,
        "skip_special_tokens": True,
        "truncate_prompt_tokens": truncate_length,
        **sampling_params,
    }

    # Generate outputs
    outputs = engine.generate(formatted_prompts, SamplingParams(**full_params))
    completions = [output.outputs[0].text for output in outputs]
    logger.debug(f"Generated {len(completions)} completions")

    return completions
