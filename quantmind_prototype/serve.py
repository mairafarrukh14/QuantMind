"""QuantMind serving backend (Work Order W7).

A small FastAPI service that reads only the locked ``results/frontend_payload.json``
-- nothing is trained or recomputed at request time -- and exposes it to the
dashboard. The same JSON is what ``export_frontend_data.py`` writes into the static
``frontend/data.js``, so the live API and the no-server static build cannot diverge:
one file produces both. The static dashboard is also mounted here, so a single
command serves the whole thing.

Endpoints:
  GET  /api/portfolio      equity curve + headline metrics
  GET  /api/allocations    per-day weights + current allocation
  GET  /api/explanations   global feature importance + per-holding drivers
  POST /api/chat           scope-restricted Q&A through the audited rationale layer
  GET  /                   the static dashboard (reads the same payload)

Run:  uvicorn serve:app --reload   (from the quantmind_prototype directory)
"""
from __future__ import annotations

import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import config, rationale

PAYLOAD_JSON = os.path.join(config.RESULTS_DIR, "frontend_payload.json")
FRONTEND_DIR = os.path.join(os.path.dirname(config.ROOT), "frontend")

app = FastAPI(title="QuantMind API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


def _payload() -> dict:
    """Load the one locked payload. Read fresh so a re-export is picked up."""
    if not os.path.exists(PAYLOAD_JSON):
        return {}
    with open(PAYLOAD_JSON) as fh:
        return json.load(fh)


@app.get("/api/health")
def health():
    p = _payload()
    return {"ok": bool(p), "source": p.get("meta", {}).get("data_source"),
            "test_days": p.get("meta", {}).get("test_days")}


@app.get("/api/portfolio")
def portfolio():
    p = _payload()
    return {"meta": p.get("meta", {}), "kpis": p.get("kpis", {}),
            "equity": p.get("equity", {})}


@app.get("/api/allocations")
def allocations():
    p = _payload()
    return {"allocation_ts": p.get("allocation_ts", {}),
            "current_allocation": p.get("current_allocation", {})}


@app.get("/api/explanations")
def explanations():
    p = _payload()
    return {"feature_importance": p.get("feature_importance", []),
            "recommendations": p.get("recommendations", []),
            "rationale": p.get("rationale", "")}


class ChatIn(BaseModel):
    question: str


@app.post("/api/chat")
def chat(msg: ChatIn):
    return rationale.chat_answer(msg.question, _payload())


# Mount the static dashboard last, at root, so /api/* takes precedence.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
