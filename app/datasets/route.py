from fastapi import APIRouter
from .service import load

router = APIRouter()


@router.get("/test")
def test():
    test = load("abstract_ouvrirlascience.json")
    if not test:
        return {"error": "dataset not loaded"}
    test_dict = test.to_dict(orient="records")
    return test_dict
