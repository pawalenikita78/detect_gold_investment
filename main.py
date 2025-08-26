from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mistralai import Mistral
from dotenv import load_dotenv
import os

# Load .env -> environment
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise RuntimeError("Set MISTRAL_API_KEY in your environment or .env")

# Mistral client + model
client = Mistral(api_key=api_key)
MODEL = "mistral-small-latest"  # try large if you prefer

app = FastAPI()

class UserMessage(BaseModel):
    message: str


@app.post("/detect_gold_intent")
async def detect_gold_intent(data: UserMessage):
    # Build a system + user prompt
    messages = [
        {"role": "system", "content": "You are an AI financial assistant. Detect if the user is asking about GOLD INVESTMENT. If yes, reply with a helpful answer about gold investments. If not, say it’s about something else."},
        {"role": "user", "content": data.message},
    ]

    # Call Mistral API for intent detection
    response = client.chat.complete(
        model="mistral-small",
        messages=messages,
        temperature=0.3
    )

    reply_text = response.choices[0].message.content

    # Detect intent
    if "gold" in reply_text.lower():
        # Use Mistral model to reply specifically about gold investment
        gold_messages = [
            {"role": "system", "content": "You are a financial assistant. Provide a detailed, helpful answer about gold investment based on the user's query."},
            {"role": "user", "content": data.message},
        ]
        gold_response = client.chat.complete(
            model="mistral-small",
            messages=gold_messages,
            temperature=0.7
        )
        gold_reply = gold_response.choices[0].message.content
        intent = "gold_investment"
        return {"intent": intent, "reply": gold_reply}
    else:
        intent = "other"
        return {"intent": intent, "reply": reply_text}



