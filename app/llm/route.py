from fastapi import APIRouter
from .service import fine_tune

router = APIRouter()


@router.get("/test")
def test():
    fine_tune("meta-llama/Llama-3.2-1B", "test.json", "test_fine_tuned")
    return {"ok": True}
