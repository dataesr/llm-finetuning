from fastapi import APIRouter
from .service import fine_tune, push_model

router = APIRouter()


@router.get("/test")
def test():
    fine_tune("meta-llama/Llama-3.2-1B", "test.json", "test_fine_tuned", None)
    return {"ok": True}


@router.post("/ft")
def run_ft(args: dict):
    model_name = args.get("model_name")
    dataset_name = args.get("dataset_name")
    output_model_name = args.get("output_model_name")
    huggingface_hub = args.get("huggingface_hub")
    fine_tune(model_name, dataset_name, output_model_name, huggingface_hub)
