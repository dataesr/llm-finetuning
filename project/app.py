import os
import sys
import json
from typing import Union, Literal
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from project.pipeline import load_vllm_engine, model_predict

# Create API
app = FastAPI(default_response_class=ORJSONResponse)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Load model and tokenizer
vllm_engine, tokenizer = load_vllm_engine(model_name=os.getenv("MODEL_NAME"))


# Chat message schema
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


# Input schema
class Input(BaseModel):
    messages: list[Union[str, list[ChatMessage]]]
    use_chatml: bool = False


@app.get("/")
async def root():
    return {"message": "LLM inference API"}


@app.post("/predict")
async def predict(input: Input):
    prediction = model_predict(engine=vllm_engine, tokenizer=tokenizer, inputs=input.messages, use_chatml=input.use_chatml)
    response = {"prediction": prediction}
    print(f"/predict response size: {sys.getsizeof(response) / 1024:.2f} KB")
    return response
