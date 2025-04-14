import redis
from rq import Queue
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from app._utils.packages import get_installed_versions
from app.redis import service as redis_service
from app.datasets.route import router as datasets_router
from app.llm.route import router as llm_router

REDIS_URL = "redis://redis:6379/0"

router = APIRouter()

@router.get("/")
def home():
    return {"application": "LLM Fine-Tuning"}


@router.get("/versions")
def versions():
    versions = get_installed_versions()
    return jsonable_encoder(versions)


@router.get("/redis")
def redis_jobs():
    q = redis_service.get_queue()
    return {"Current jobs in 'llm' queue": q.count}


# Add datasets route (for test only)
router.include_router(datasets_router, prefix="/datasets")

# Add llm route
router.include_router(llm_router, prefix="/llm")
