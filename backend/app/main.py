"""
MedTrust AI - Hospital Telehealth & AI Clinical Case Sheet Platform
Main FastAPI application configuring CORS, Database initialization,
REST API routers, WebSocket live transcript broadcasting, and Static Frontend serving.
"""

import os
import json
import logging
from typing import Dict, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.database import init_db
from backend.app.api import auth, patients, consultations, casesheets, journals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medtrust")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing MedTrust Clinical Database and seed data...")
    init_db()
    logger.info("Database initialized successfully.")
    yield

app = FastAPI(
    title="MedTrust AI - Clinical Telehealth & Case Sheet Platform",
    description="Hospital platform for live telehealth, Google Meet integration, real-time transcription, and 17-section AI Case Sheet generation.",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API Routers
app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(consultations.router)
app.include_router(casesheets.router)
app.include_router(journals.router)


# --- Real-Time WebSocket Connection Manager for Consultations ---
class ConsultationConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, consultation_id: str, websocket: WebSocket):
        await websocket.accept()
        if consultation_id not in self.active_connections:
            self.active_connections[consultation_id] = set()
        self.active_connections[consultation_id].add(websocket)
        logger.info(f"WebSocket client connected to consultation {consultation_id}")

    def disconnect(self, consultation_id: str, websocket: WebSocket):
        if consultation_id in self.active_connections:
            self.active_connections[consultation_id].discard(websocket)
            if not self.active_connections[consultation_id]:
                del self.active_connections[consultation_id]
        logger.info(f"WebSocket client disconnected from consultation {consultation_id}")

    async def broadcast(self, consultation_id: str, message: dict):
        if consultation_id in self.active_connections:
            dead_sockets = set()
            for connection in self.active_connections[consultation_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    dead_sockets.add(connection)
            for dead in dead_sockets:
                self.active_connections[consultation_id].discard(dead)


ws_manager = ConsultationConnectionManager()


@app.websocket("/ws/consultation/{consultation_id}")
async def consultation_websocket_endpoint(websocket: WebSocket, consultation_id: str):
    await ws_manager.connect(consultation_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            event_type = data.get("type", "transcript_turn")
            
            # Broadcast to all participants (doctor, student, patient)
            await ws_manager.broadcast(consultation_id, {
                "type": event_type,
                "data": data.get("data", {}),
                "sender": data.get("sender", "anonymous")
            })
    except WebSocketDisconnect:
        ws_manager.disconnect(consultation_id, websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(consultation_id, websocket)


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "MedTrust AI Hospital System",
        "version": "2.0.0",
        "modules": [
            "Google Meet Spaces & Calendar API",
            "Real-time Audio Waveform & Speech-to-Text",
            "17-Section Clinical Case Sheet Extractor",
            "Doctor Electronic Sign-Off & Verification",
            "Multilingual Summaries (EN, TA, HI, TE, ML, KN) with TTS Audio",
            "Patient Management & Auto-Age Calculator",
            "Multi-Role Access (Doctor, Student, Patient)"
        ]
    }


# Static Frontend Files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Frontend index.html under construction"}
