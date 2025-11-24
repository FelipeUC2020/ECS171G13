from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from data_cleanup import DataProcessor
from model_loader import ModelLoader
from llm_recommendations import get_recommendations
import random

app = FastAPI()

# CORS for frontend at localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals 
X_TEST_CH = None
CNN_MODEL = None

@app.on_event("startup")
def _startup():
    global X_TEST_CH, CNN_MODEL
    processor = DataProcessor(input_steps=24*3, output_steps=24)
    (_, _), (_, _), (X_test, _) = processor.load_and_process_data()
    channel_indices = [4, 5, 6, 7]
    X_TEST_CH = X_test[:, :, channel_indices]
    CNN_MODEL = ModelLoader("CNN")

def _prepare_test_sample():
    # sample a random entry from the test data
    idx = random.randint(0, len(X_TEST_CH) - 1)
    return X_TEST_CH[idx]

@app.get("/run")
def run_pipeline():
    x = _prepare_test_sample()
    preds = CNN_MODEL.get_predictions(x)
    recs = get_recommendations(preds.tolist())
    return {"input": x.tolist(), "predictions": preds.tolist(), "llm_recommendations": recs}

@app.get("/")
def root():
    return {"status": "ok", "message": "API is running"}