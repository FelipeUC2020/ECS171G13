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
LSTM_MODEL = None

@app.on_event("startup")
def _startup():
    global X_test, y_test, CNN_MODEL, LSTM_MODEL
    processor = DataProcessor(input_steps=24, output_steps=24)
    (_, _), (_, _), (X_test, y_test) = processor.load_and_process_data()
    CNN_MODEL = ModelLoader("CNN")
    LSTM_MODEL = ModelLoader("LSTM")

def _prepare_test_sample():
    # sample a random entry from the test data
    idx = random.randint(0, len(X_test) - 1)
    return X_test[idx], y_test[idx]

@app.get("/run")
def run_pipeline():
    x, y = _prepare_test_sample()
    cnn_preds = CNN_MODEL.get_predictions(x)
    lstm_preds = LSTM_MODEL.get_predictions(x)
    # recs = get_recommendations(cnn_preds.tolist())
    return {
        "input": x.tolist(),
        "cnn_predictions": cnn_preds.tolist(),
        "lstm_predictions": lstm_preds.tolist(), 
        "label": y.tolist()
    }

@app.get("/")
def root():
    return {"status": "ok", "message": "API is running"}