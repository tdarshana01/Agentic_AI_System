from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json

from query.prompt_logic import analyze_incident_logic

app = FastAPI()
clients: List[WebSocket] = []


# === Models ===
class Incident(BaseModel):
    incident_id: str
    machine: str
    timestamp: str
    type: str
    measured_value: Optional[str] = None
    threshold: Optional[str] = None
    status: str


class IncidentRequest(BaseModel):
    new_incident: Incident


# === REST API ===
@app.post("/analyze_incident/")
async def analyze_incident(request: IncidentRequest):
    new_incident = request.new_incident.dict()   # ✅ convert Pydantic → dict
    result = analyze_incident_logic(new_incident)

    # Push to connected WS clients
    for ws in clients.copy():
        try:
            await ws.send_text(result["ai_answer"])
        except:
            clients.remove(ws)

    return {
        "query": new_incident,
        "retrieved_context": result["retrieved_context"],
        "ai_answer": result["ai_answer"],
    }


# === WebSocket ===
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            await asyncio.sleep(1)  # keep alive
    except WebSocketDisconnect:
        clients.remove(ws)
