import redis
from fastapi import APIRouter, Request
from .service import fine_tune
from app.redis import service as redis_service
from app.logging import get_logger

router = APIRouter()

logger = get_logger(__name__)

@router.get("/test")
def test():
    fine_tune("meta-llama/Llama-3.2-1B", "test.json", "test_fine_tuned", None)
    return {"ok": True}


@router.post("/ft")
async def run_ft(request: Request):
    args = await request.json()
    model_name = args.get("model_name")
    dataset_name = args.get("dataset_name")
    output_model_name = args.get("output_model_name")
    huggingface_hub = args.get("huggingface_hub")
    # assert(args.get('PUBLIC_API_PASSWORD') == PUBLIC_API_PASSWORD)
    q = redis_service.get_queue()
    logger.debug("Starting task fine_tune")
    logger.debug(args)
    task = q.enqueue(fine_tune, model_name, dataset_name, output_model_name, huggingface_hub)
    response = {"status": "success", "data": {"task_id": task.get_id()}}
    return response
