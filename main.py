from fastapi import FastAPI
app = FastAPI()
from services.chat_service import ChatService

@app.get("/")
async def root():
    return {"This is a rex agent service for all types of agents !"}

@app.post("/healthz")
async def healthcheck():
    return {"status": "healthy"}

@app.get("/chat")
async def chat():
    return None

