from fastapi import APIRouter
from .service import load

router = APIRouter()


@router.get("/test")
def test():
    test = load("test.json")
    if test is None:
        return {"error": "dataset not loaded"}
    test_dict = test.to_dict()
    return test_dict
