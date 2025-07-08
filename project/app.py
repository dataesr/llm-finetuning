import os
import json
import asyncio
import torch
from uuid import uuid4
import time
from typing import Union, Literal, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.version import __version__ as VLLM_VERSION
from project.pipeline import load_pretrained_tokenizer
from project.logger import get_logger

logger = get_logger(__name__)

FOLDER = "completions"

# Chat message schema
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


# Generate request schema
class RequestData(BaseModel):
    prompts: list[Union[str, list[ChatMessage]]]
    use_chatml: bool = False
    sampling_params: Dict[str, Any] = {}


# Generate task schema
class Task(BaseModel):
    status: Literal["queued", "running", "done", "error"]
    completions: Optional[list[str]] = None
    error: Optional[str] = None
    queued_at: float
    running_at: Optional[float] = None
    done_at: Optional[float] = None


# Generate task store schema
class TaskStore:

    def __init__(self, save_dir: str = None):
        self._store: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
        self._save_on_dir = save_dir
        if save_dir is None or not os.path.isdir(str(save_dir)):
            logger.warning("TaskStore saving directory not found: saving to disk disabled")
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
            logger.debug(f"💾 Task {task_id}  completions saved to {file_path}")

    async def get(self, task_id: str) -> Task:
        async with self._lock:
            task = self._store.get(task_id)
            if not task:
                raise KeyError(f"Task {task_id} not found")
            return task

    async def cleanup(self, older_than_secs: int = 60 * 10):
        async with self._lock:
            now = time.time()
            expired = [tid for tid, t in self._store.items() if t.done_at and (now - t.done_at) > older_than_secs]
            for tid in expired:
                del self._store[tid]
                logger.debug(f"🗑️ Task {tid} expired and removed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🟧 Initializing application...")

    model_name = os.getenv("MODEL_NAME")
    assert model_name is not None

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
    logger.info(f"✅ vLLM engine loaded")
    app.state.engine_lock = asyncio.Lock()

    # Initialize tokenizer
    app.state.tokenizer = load_pretrained_tokenizer(model_name=model_name)

    # Initialize task store
    app.state.task_store = TaskStore(save_dir=FOLDER)

    # Initialize cleaning function
    async def cleanup_task_store():
        while True:
            await app.state.task_store.cleanup(older_than_secs=600)
            await asyncio.sleep(60)

    asyncio.create_task(cleanup_task_store())
    logger.info(f"✅ Task store initialized")

    yield

    # Optional cleanup logic here
    logger.info("🟥 Shutting down app.")


# Create API
app = FastAPI(lifespan=lifespan, default_response_class=ORJSONResponse)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/")
async def root():
    """Home route"""
    return {"message": "LLM inference API (sync vLLM)"}


# @app.get("/stream-ping")
# async def stream_ping():
#     async def generator():
#         for i in range(90):
#             yield f"ping {i}\n".encode()
#             await asyncio.sleep(1)

#     return StreamingResponse(generator(), media_type="text/plain")


@app.post("/generate")
async def generate(request_data: RequestData) -> Response:
    """Create a generation task for the request.

    The request should be a JSON object with the following fields:
    - prompts: the list of texts to use for the generation (str or list of chat message).
    - use_chatml: whether to use chatml format for prompts
    - use_stream: whether to stream the results or not.
    - sampling_params: the sampling parameters.
    """
    task_id = await app.state.task_store.create()

    async def generate_task(task_id: str, request_data: RequestData):
        try:
            logger.info(
                f"➡️ New generation task {task_id} added (prompts={len(request_data.prompts)}, use_chatml={request_data.use_chatml})"
            )
            if request_data.sampling_params:
                logger.debug(f"Generation with custom params: {request_data.sampling_params}")

            async with app.state.engine_lock:
                await app.state.task_store.set_running(task_id)
                completions = await asyncio.to_thread(
                    _generate,
                    app.state.engine,
                    app.state.tokenizer,
                    request_data.prompts,
                    request_data.use_chatml,
                    request_data.sampling_params,
                )
            await app.state.task_store.set_done(task_id, completions=completions)
            logger.info(f"✅ Generation task {task_id} done.")
        except Exception as e:
            await app.state.task_store.set_error(task_id, error=str(e))
            logger.error(f"❌ Generation task {task_id} failed: {e}")

    # Create task
    asyncio.create_task(generate_task(task_id, request_data))

    # Return task id
    return {"task_id": task_id, "status": "queued"}


# Get task status and completions
@app.get("/generate/{task_id}")
async def get_task(task_id: str):
    try:
        task = await app.state.task_store.get(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


def _generate(
    engine: LLM,
    tokenizer,
    prompts: list[Union[str, list[ChatMessage]]],
    use_chatml: bool,
    sampling_params: Dict[str, Any],
) -> list[str]:
    logger.debug(f"Running generation on {len(prompts)} prompts...")

    # Format prompts
    formatted_prompts = [
        (
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            if isinstance(p, list) and use_chatml
            else p
        )
        for p in prompts
    ]

    # Sampling params
    full_params = {"seed": 0, "temperature": 0, "max_tokens": 1024, "skip_special_tokens": True, **sampling_params}

    # Generate outputs
    outputs = engine.generate(formatted_prompts, SamplingParams(**full_params))
    completions = [output.outputs[0].text for output in outputs]
    logger.debug(f"Generated {len(completions)} completions")

    return completions
