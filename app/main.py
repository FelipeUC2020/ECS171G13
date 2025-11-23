from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import numpy as np
import os
from .model_loader import get_model  # Import model loader

app = FastAPI()

# Device for model inference
device = torch.device('cpu')  # Use CPU for simplicity

# Select and load the model (change 'cnn' to other keys for different models)
MODEL_KEY = 'cv72-24'  # Change this to switch models, e.g., 'rnn' if added
model = get_model(MODEL_KEY)
print("Open the web app at http://127.0.0.1:8000")

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Template system
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/models")
async def get_models():
    # TODO: Return list of available models for selection in UI
    from .model_loader import MODELS
    return JSONResponse({"models": list(MODELS.keys())})

@app.post("/predict")
async def predict(payload: dict):
    # TODO: Implement input validation (e.g., check if input is list of 72 lists of 4 floats)
    model_key = payload.get("model", MODEL_KEY)  # TODO: Support model selection
    input_data = payload["input"]  # e.g., [[hour1_ch1, hour1_ch2, ...], [hour2_ch1, ...], ...]
    model = get_model(model_key)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 72, 4)
    input_tensor = input_tensor.permute(0, 2, 1)  # Permute to (1, 4, 72) for CNN
    
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy().flatten().tolist()  # 24-hour forecast
    
    return JSONResponse({"prediction": prediction})
