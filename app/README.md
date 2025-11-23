# App Development Guide

## Overview

This is a FastAPI web application for CNN-based multi-step energy forecasting. It loads a pre-trained CNN model and provides a web interface for predictions.

## Running the App

1. Ensure you have Python and dependencies installed (see root `requirements.txt`).

2. From the project root directory, run:

   ```bash
   uvicorn app.main:app --reload
   ```

3. Open `http://127.0.0.1:8000` in your browser to access the web interface.

## TODOs

- [ ] Add support for RNN/LSTM models (placeholder in model_loader.py).
- [ ] Implement input validation for the prediction endpoint (TODO in main.py).
- [ ] Add error handling and user-friendly messages in the HTML/JS (TODOs in script.js).
- [ ] Improve UI with charts for predictions (TODO in index.html and script.js).
- [x] Support multiple model selection in the web interface (implemented).
