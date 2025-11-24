#!/usr/bin/env bash
source venv/bin/activate
exec uvicorn --host 127.0.0.1 --port 8000 app:app --env-file .env --reload
