from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"This is a rex agent service for all types of agents !"}

@app.post("/healthz")
async def healthcheck():
    return {"status": "healthy"}
