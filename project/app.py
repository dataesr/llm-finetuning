import os
from fastapi import FastAPI
from pydantic import BaseModel
from project.pipeline import load_vllm_engine, model_predict

# Create API
app = FastAPI()

# Load model and tokenizer
vllm_engine, tokenizer = load_vllm_engine(model_name=os.getenv("MODEL_NAME"))

# Input schema
class Input(BaseModel):
    messages: str | object
    use_chatml: bool = False


@app.get("/")
def root():
    return {"message": "LLM inference API"}


@app.post("/predict")
def predict(input: Input):
    prediction = model_predict(engine=vllm_engine, tokenizer=tokenizer, inputs=input.messages, use_chatml=input.use_chatml)
    return {"prediction": prediction}
