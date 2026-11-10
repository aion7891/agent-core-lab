from fastapi import FastAPI

app = FastAPI(title="agent-core-lab")


@app.get("/")
async def root():
    return {"message": "This is a rex agent service for all types of agents!"}


@app.get("/healthz")
async def healthcheck():
    return {"status": "healthy"}
