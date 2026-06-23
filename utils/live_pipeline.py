# =============================================================================
# utils/live_pipeline.py
# Live Detection Pipeline — connects FlowBuffer (capture/replay) to the
# SST-Net model for real-time scoring, alerting, and rolling history.
# =============================================================================

import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from utils.preprocessing import preprocess, build_features, _safe_port
from utils.mitre_mapper import map_dataframe


class LiveDetectionPipeline:
    """Maintains rolling history of scored flows from the live FlowBuffer.

    Call `process_batch()` on each Streamlit rerun to drain the buffer,
    run inference, and update the rolling history + alert log.
    """

    def __init__(self, model, scaler, device: str = "cpu", history_size: int = 500):
        self.model        = model
        self.scaler       = scaler
        self.device       = device
        self.history_size = history_size

        # Rolling history of all scored flows (deque auto-evicts oldest)
        self.history: deque = deque(maxlen=history_size)
        # Only flows that crossed the alert threshold
        self.alerts:  deque = deque(maxlen=history_size)

        self.total_processed = 0
        self.total_alerts    = 0
        self.last_update     = None

    def process_batch(self, flows: list, threshold: float = 0.5) -> dict:
        """Process a batch of raw flow dicts through the full pipeline.

        Parameters
        ----------
        flows     : list of dicts (raw flow records from FlowBuffer.drain())
        threshold : classification threshold

        Returns
        -------
        dict with keys: n_processed, n_alerts, new_alerts (list of dicts)
        """
        if not flows:
            return {"n_processed": 0, "n_alerts": 0, "new_alerts": []}

        df = pd.DataFrame(flows)

        # Normalize hex ports
        for col in ["Sport", "Dport"]:
            if col in df.columns:
                df[col] = df[col].apply(_safe_port)

        # ── Preprocess + inference ────────────────────────────────────────────
        try:
            X = preprocess(df, self.scaler)
        except Exception as e:
            return {"n_processed": 0, "n_alerts": 0, "new_alerts": [], "error": str(e)}

        X_tensor = torch.tensor(X).float().to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        # ── Build history records ─────────────────────────────────────────────
        now = datetime.now()
        new_alerts = []

        for i, (_, row) in enumerate(df.iterrows()):
            prob   = float(probs[i])
            is_threat = prob > threshold

            record = {
                "timestamp":   now.strftime("%H:%M:%S"),
                "src_addr":    row.get("SrcAddr", "—"),
                "dst_addr":    row.get("DstAddr", "—"),
                "sport":       row.get("Sport",   "—"),
                "dport":       row.get("Dport",   "—"),
                "proto":       row.get("Proto",   "—"),
                "confidence":  round(prob, 4),
                "verdict":     "THREAT" if is_threat else "BENIGN",
                "severity":    (
                    "CRITICAL" if prob >= 0.85 else
                    "HIGH"     if prob >= 0.70 else
                    "MEDIUM"   if prob >= 0.50 else "LOW"
                ),
            }

            self.history.append(record)

            if is_threat:
                self.alerts.append(record)
                new_alerts.append(record)
                self.total_alerts += 1

        self.total_processed += len(df)
        self.last_update = now

        return {
            "n_processed": len(df),
            "n_alerts":    len(new_alerts),
            "new_alerts":  new_alerts,
        }

    def get_history_df(self) -> pd.DataFrame:
        """Return rolling history as a DataFrame (most recent first)."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(list(self.history))[::-1].reset_index(drop=True)

    def get_alerts_df(self) -> pd.DataFrame:
        """Return alert log as a DataFrame (most recent first)."""
        if not self.alerts:
            return pd.DataFrame()
        return pd.DataFrame(list(self.alerts))[::-1].reset_index(drop=True)

    def get_stats(self) -> dict:
        """Summary stats for the live dashboard panel."""
        recent_threats = sum(1 for r in self.history if r["verdict"] == "THREAT")
        recent_total   = len(self.history)
        live_risk_pct  = (recent_threats / recent_total * 100) if recent_total > 0 else 0.0

        return {
            "total_processed": self.total_processed,
            "total_alerts":    self.total_alerts,
            "live_risk_pct":   round(live_risk_pct, 2),
            "history_size":    recent_total,
            "last_update":     self.last_update.strftime("%H:%M:%S") if self.last_update else "—",
        }

    def reset(self):
        self.history.clear()
        self.alerts.clear()
        self.total_processed = 0
        self.total_alerts    = 0
        self.last_update     = None


# =============================================================================
# Singleton accessor — persists across Streamlit reruns via session_state
# =============================================================================
def get_pipeline(model, scaler, device: str, st_session_state: dict) -> LiveDetectionPipeline:
    """Get or create the LiveDetectionPipeline, stored in Streamlit session_state
    so it survives reruns within the same browser session.
    """
    if "_live_pipeline" not in st_session_state:
        st_session_state["_live_pipeline"] = LiveDetectionPipeline(model, scaler, device)
    return st_session_state["_live_pipeline"]
