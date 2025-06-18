import os
from fastapi import FastAPI
from pydantic import BaseModel
from script.pipeline import load_pretrained_model, predict

# Create API
app = FastAPI()

# Load model and tokenizer
model, tokenizer = load_pretrained_model(model_name=os.getenv("MODEL_NAME"))


# Input schema
class Input(BaseModel):
    input: str | object
    use_chatml: bool = False


@app.get("/")
def root():
    return {"message": "LLM inference API"}


@app.post("/predict")
def predict(input: Input):
    prediction = predict(model=model, tokenizer=tokenizer, input=input.input, use_chatml=input.use_chatml)
    return {"prediction": prediction}
