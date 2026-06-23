# =============================================================================
# api.py — SST-Net FastAPI Backend
#
# Runs alongside the Streamlit dashboard as a separate process.
# Exposes REST endpoints for detection, health, and model info.
#
# Start with:
#   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
#
# Docs auto-generated at:
#   http://localhost:8000/docs     ← Swagger UI
#   http://localhost:8000/redoc    ← ReDoc
# =============================================================================

import io
import os
import time
import logging
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils.preprocessing import load_scaler, preprocess, build_features, FEATURE_COLS
from utils.mitre_mapper import map_dataframe, tactic_summary, technique_summary

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("sst_net_api")

# =============================================================================
# Model architecture (must match training exactly)
# =============================================================================
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=17, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, hidden_dim=128):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc          = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        x = self.input_layer(x).unsqueeze(1)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))


# =============================================================================
# App startup — load model and scaler once
# =============================================================================
app = FastAPI(
    title="SST-Net Botnet Detection API",
    description=(
        "Self-Supervised Transformer for early botnet detection in IoT networks. "
        "Upload a CTU-13 network flow CSV to get threat predictions, "
        "confidence scores, and MITRE ATT&CK mappings."
    ),
    version="1.0.0",
    contact={
        "name":  "Poorvi Prahlad Purohit",
        "email": "poorvi@example.com",
    },
)

# Allow Streamlit dashboard to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Global model/scaler (loaded once at startup) ──────────────────────────────
_model:  Optional[TransformerClassifier] = None
_scaler = None
_device: str = "cpu"
_startup_time: float = time.time()


@app.on_event("startup")
def startup_event():
    global _model, _scaler, _device

    logger.info("SST-Net API starting up...")

    # ── Load scaler ───────────────────────────────────────────────────────────
    try:
        _scaler = load_scaler()
        logger.info(f"Scaler loaded: {type(_scaler).__name__} | features={_scaler.n_features_in_}")
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        raise RuntimeError(f"Scaler load failed: {e}")

    # ── Load model ────────────────────────────────────────────────────────────
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model  = TransformerClassifier(input_dim=17)

    model_path = "transformer_classifier.pt"
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}")

    try:
        _model.load_state_dict(
            torch.load(model_path, map_location=_device), strict=False
        )
        _model.to(_device)
        _model.eval()
        logger.info(f"Model loaded on {_device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model load failed: {e}")

    logger.info("SST-Net API ready ✅")


# =============================================================================
# Pydantic response schemas
# =============================================================================
class HealthResponse(BaseModel):
    status:       str
    uptime_sec:   float
    model_loaded: bool
    scaler_type:  str
    device:       str
    timestamp:    str


class DetectionSummary(BaseModel):
    total_flows:      int
    threats_detected: int
    benign_flows:     int
    risk_score_pct:   float
    avg_confidence:   float
    threshold_used:   float
    processing_ms:    float


class ThreatFlow(BaseModel):
    flow_index:    int
    confidence:    float
    severity:      str
    attack_ids:    str
    attack_tactics: str


class DetectionResponse(BaseModel):
    summary:      DetectionSummary
    top_threats:  list[ThreatFlow]
    tactic_counts: dict
    technique_counts: dict


# =============================================================================
# Endpoints
# =============================================================================

# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Check if the API is running and model is loaded."""
    return HealthResponse(
        status       = "healthy" if _model is not None else "degraded",
        uptime_sec   = round(time.time() - _startup_time, 1),
        model_loaded = _model is not None,
        scaler_type  = type(_scaler).__name__ if _scaler else "not loaded",
        device       = _device,
        timestamp    = datetime.utcnow().isoformat() + "Z",
    )


# ── GET /model/info ───────────────────────────────────────────────────────────
@app.get("/model/info", tags=["Model"])
def model_info():
    """Return model architecture details and feature list."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    total_params = sum(p.numel() for p in _model.parameters())

    return {
        "architecture":  "TransformerClassifier",
        "input_dim":     17,
        "d_model":       128,
        "num_heads":     4,
        "num_layers":    2,
        "total_params":  total_params,
        "device":        _device,
        "feature_cols":  FEATURE_COLS,
        "scaler":        type(_scaler).__name__,
        "classes":       ["BENIGN", "BOTNET"],
    }


# ── POST /detect ──────────────────────────────────────────────────────────────
@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect(
    file:      UploadFile = File(..., description="CTU-13 network flow CSV file"),
    threshold: float      = Query(0.5, ge=0.0, le=1.0,
                                  description="Classification threshold (0–1). "
                                              "Lower = more sensitive."),
    max_rows:  int        = Query(10000, ge=1, le=100000,
                                  description="Max rows to process"),
):
    """
    Upload a CTU-13 CSV and get botnet detection results.

    Returns:
    - Detection summary (threats, risk score, confidence)
    - Top 20 threat flows with confidence and MITRE ATT&CK mapping
    - Tactic and technique breakdown
    """
    if _model is None or _scaler is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    t_start = time.time()

    # ── Read CSV ──────────────────────────────────────────────────────────────
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8", errors="replace")),
                         on_bad_lines="skip", nrows=max_rows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty")

    # ── Normalize hex ports ───────────────────────────────────────────────────
    from utils.preprocessing import _safe_port
    for col in ["Sport", "Dport"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_port)

    # ── Preprocess + Inference ────────────────────────────────────────────────
    try:
        X = preprocess(df, _scaler)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    X_tensor = torch.tensor(X).float().to(_device)

    with torch.no_grad():
        logits = _model(X_tensor)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    preds     = (probs > threshold).astype(int)
    n_threats = int(preds.sum())
    n_benign  = len(preds) - n_threats
    risk_pct  = round(float(n_threats / len(preds) * 100), 2) if len(preds) > 0 else 0.0
    avg_conf  = round(float(probs[probs > threshold].mean() * 100), 1) if n_threats > 0 else 0.0

    processing_ms = round((time.time() - t_start) * 1000, 1)
    logger.info(
        f"detect | rows={len(df)} | threats={n_threats} | "
        f"risk={risk_pct}% | {processing_ms}ms"
    )

    # ── MITRE ATT&CK mapping on threat flows ──────────────────────────────────
    threat_df = df[probs > threshold].copy().reset_index(drop=True)
    threat_df["_confidence"] = probs[probs > threshold]

    top_threats: list[ThreatFlow] = []
    tactic_counts:    dict = {}
    technique_counts: dict = {}

    if n_threats > 0:
        mapped = map_dataframe(threat_df, max_rows=min(n_threats, 500))

        t_sum    = tactic_summary(mapped)
        tech_sum = technique_summary(mapped)

        tactic_counts    = dict(zip(t_sum["tactic"], t_sum["count"].astype(int)))
        technique_counts = dict(zip(tech_sum["id"],  tech_sum["count"].astype(int)))

        for i, row in mapped.head(20).iterrows():
            conf = float(row.get("_confidence", 0))
            sev  = (
                "CRITICAL" if conf >= 0.85 else
                "HIGH"     if conf >= 0.70 else
                "MEDIUM"   if conf >= 0.50 else "LOW"
            )
            top_threats.append(ThreatFlow(
                flow_index    = int(i),
                confidence    = round(conf, 4),
                severity      = sev,
                attack_ids    = str(row.get("attack_ids",    "T1071")),
                attack_tactics= str(row.get("attack_tactics","Command and Control")),
            ))

    return DetectionResponse(
        summary=DetectionSummary(
            total_flows      = len(preds),
            threats_detected = n_threats,
            benign_flows     = n_benign,
            risk_score_pct   = risk_pct,
            avg_confidence   = avg_conf,
            threshold_used   = threshold,
            processing_ms    = processing_ms,
        ),
        top_threats      = top_threats,
        tactic_counts    = tactic_counts,
        technique_counts = technique_counts,
    )


# ── POST /predict/flow ────────────────────────────────────────────────────────
@app.post("/predict/flow", tags=["Detection"])
async def predict_single_flow(flow: dict):
    """
    Predict botnet probability for a single network flow (JSON).

    Send a JSON object with network flow features.
    Returns botnet probability and verdict.

    Example body:
    {
        "Duration": 0.5,
        "Proto": "tcp",
        "Sport": 12345,
        "Dport": 6667,
        "TotPkts": 10,
        "TotBytes": 5000,
        "SrcBytes": 4800,
        "Dir": "->",
        "State": "CON",
        "sTos": 0,
        "dTos": 0
    }
    """
    if _model is None or _scaler is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        df   = pd.DataFrame([flow])
        X    = preprocess(df, _scaler)
        X_t  = torch.tensor(X).float().to(_device)

        with torch.no_grad():
            logits = _model(X_t)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        prob_botnet = float(probs[1])
        verdict = (
            "MALICIOUS"  if prob_botnet >= 0.85 else
            "SUSPICIOUS" if prob_botnet >= 0.50 else
            "BENIGN"
        )

        return {
            "prob_benign":  round(float(probs[0]), 4),
            "prob_botnet":  round(prob_botnet, 4),
            "verdict":      verdict,
            "confidence_pct": round(prob_botnet * 100, 1),
        }

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")


# ── GET /features ─────────────────────────────────────────────────────────────
@app.get("/features", tags=["Model"])
def get_features():
    """Return the list of features the model expects."""
    return {"feature_cols": FEATURE_COLS, "count": len(FEATURE_COLS)}
