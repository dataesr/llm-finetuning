import sys
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from ._utils.packages import get_installed_versions
from .datasets.route import router as datasets_router

router = APIRouter()


@router.get("/")
def home():
    return {"application": "LLM Fine-Tuning"}


@router.get("/versions")
def versions():
    versions = get_installed_versions()
    return jsonable_encoder(versions)


# Add datasets route
router.include_router(datasets_router, prefix="/datasets")
