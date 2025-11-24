import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_api_key: Optional[str] = os.getenv("GPT_KEY")
if not _api_key:
    raise RuntimeError("Missing GPT_KEY in environment. Set it in your .env file.")

_client = OpenAI(api_key=_api_key)

def _ask_gpt(prompt: str, model: str = "gpt-4o-mini") -> str:
    resp = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""

def get_recommendations(data):
    prompt = f"""
    You are a power consuption expert analyst, your job is to read a time series of power consumption data and 
    generate recommendations for the user to reduce their power consumption. Pay attention to spikes and drops in the data.
    Hypothesize possible causes for the spikes and drops and derive the recommendations based on them.
    
    Data: {data}
    """ 
    return _ask_gpt(prompt)
