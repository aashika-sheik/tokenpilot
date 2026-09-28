"""Small telemetry ingestion API for company-managed AI gateways.
Run separately from Streamlit:
    uvicorn api:app --host 0.0.0.0 --port 8000
Set TOKENPILOT_INGEST_API_KEY in the environment.
"""
from __future__ import annotations
import os
from datetime import datetime
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from app import SessionLocal, UsageEvent, ORG_ID, audit, calculate_cost, classify_complexity

app = FastAPI(title="TokenPilot Telemetry API", version="1.0.0")
INGEST_KEY = os.getenv("TOKENPILOT_INGEST_API_KEY")

class Event(BaseModel):
    request_id: str = Field(min_length=1, max_length=100)
    timestamp: datetime
    developer_email: str
    team: str
    provider: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    task_type: str
    task_complexity: str | None = None
    repository: str | None = None
    latency_ms: float | None = None
    status: str = "success"


def auth(x_api_key: str | None):
    if not INGEST_KEY or not x_api_key or x_api_key != INGEST_KEY:
        raise HTTPException(status_code=401, detail="Invalid ingestion credentials")

@app.get("/health")
def health():
    return {"status": "ok", "service": "tokenpilot-telemetry"}

@app.post("/v1/telemetry")
def ingest(event: Event, x_api_key: str | None = Header(default=None)):
    auth(x_api_key)
    with SessionLocal() as s:
        exists = s.query(UsageEvent).filter(UsageEvent.organization_id == ORG_ID, UsageEvent.request_id == event.request_id).first()
        if exists:
            return {"status": "duplicate", "request_id": event.request_id}
        ic, oc, total = calculate_cost(event.model, event.input_tokens, event.output_tokens)
        obj = UsageEvent(
            organization_id=ORG_ID, request_id=event.request_id, timestamp=event.timestamp.replace(tzinfo=None),
            developer_email=event.developer_email, team=event.team, provider=event.provider, model=event.model,
            input_tokens=event.input_tokens, output_tokens=event.output_tokens, task_type=event.task_type,
            task_complexity=event.task_complexity or classify_complexity(event.task_type, event.input_tokens, event.output_tokens),
            repository=event.repository, latency_ms=event.latency_ms, status=event.status,
            input_cost=ic, output_cost=oc, total_cost=total,
        )
        s.add(obj); s.commit(); audit(s, ORG_ID, "telemetry-api", "telemetry_received", {"request_id": event.request_id})
    return {"status": "accepted", "request_id": event.request_id, "cost": total}

@app.post("/v1/telemetry/batch")
def ingest_batch(events: list[Event], x_api_key: str | None = Header(default=None)):
    auth(x_api_key)
    accepted = 0
    with SessionLocal() as s:
        existing = {x for x in s.query(UsageEvent.request_id).filter(UsageEvent.organization_id == ORG_ID).all()}
        existing = {x[0] for x in existing}
        for event in events:
            if event.request_id in existing: continue
            ic, oc, total = calculate_cost(event.model, event.input_tokens, event.output_tokens)
            s.add(UsageEvent(
                organization_id=ORG_ID, request_id=event.request_id, timestamp=event.timestamp.replace(tzinfo=None),
                developer_email=event.developer_email, team=event.team, provider=event.provider, model=event.model,
                input_tokens=event.input_tokens, output_tokens=event.output_tokens, task_type=event.task_type,
                task_complexity=event.task_complexity or classify_complexity(event.task_type, event.input_tokens, event.output_tokens),
                repository=event.repository, latency_ms=event.latency_ms, status=event.status,
                input_cost=ic, output_cost=oc, total_cost=total,
            ))
            accepted += 1
        s.commit(); audit(s, ORG_ID, "telemetry-api", "telemetry_batch_received", {"accepted": accepted})
    return {"status": "accepted", "accepted": accepted, "received": len(events)}
