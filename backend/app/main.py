from fastapi import FastAPI

from backend.app.api import journals

app = FastAPI(title="MindAura.AI API")
app.include_router(journals.router)


@app.get("/health")
def health():
    """
    Liveness check that never touches the ai/ pipeline or triggers any
    model loading, so it stays instant regardless of whether anything has
    been analyzed yet.
    """
    return {"status": "ok"}
