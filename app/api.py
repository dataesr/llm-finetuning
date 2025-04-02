import sys
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from app._utils.packages import get_installed_versions
from app.datasets.route import router as datasets_router
from app.llm.route import router as llm_router

router = APIRouter()

@router.get("/")
def home():
    return {"application": "LLM Fine-Tuning"}


@router.get("/versions")
def versions():
    versions = get_installed_versions()
    return jsonable_encoder(versions)


# Add datasets route (for test only)
router.include_router(datasets_router, prefix="/datasets")

# Add llm route
router.include_router(llm_router, prefix="/llm")
