# =============================================================================
# utils/preprocessing.py
# =============================================================================

import os
import json
import hashlib
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_MODELS_DIR   = os.path.join(_PROJECT_ROOT, "models")

SCALER_PATH = os.path.join(_MODELS_DIR, "scaler.pkl")
COLS_PATH   = os.path.join(_MODELS_DIR, "feature_columns.json")

FEATURE_COLS: list[str] = [
    'Duration', 'Proto', 'Sport', 'Dir', 'Dport', 'State', 'sTos', 'dTos',
    'TotPkts', 'TotBytes', 'SrcBytes', 'BytesPerSec', 'PktsPerSec',
    'AvgPktSize', 'SrcByteRatio', 'Sport_is_priv', 'Dport_is_priv',
]

_LOG_COLS: list[str] = [
    'TotBytes', 'TotPkts', 'SrcBytes',
    'BytesPerSec', 'PktsPerSec', 'AvgPktSize',
]


def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(str(s).encode("utf-8")).hexdigest(), 16) % 1000


def _safe_port(v) -> int:
    try:
        s = str(v).strip()
        return int(s, 16) if s.startswith(('0x', '0X')) else int(float(s))
    except (ValueError, TypeError):
        return 0


def load_scaler() -> StandardScaler:
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at '{SCALER_PATH}'.\n"
            "Run export_scaler.py in your Colab training notebook to generate "
            "models/scaler.pkl, then place it in this project's models/ folder."
        )

    scaler = joblib.load(SCALER_PATH)

    if not isinstance(scaler, StandardScaler):
        raise TypeError(
            f"Expected StandardScaler in models/scaler.pkl but got "
            f"{type(scaler).__name__}. "
            "Re-run export_scaler.py in the training notebook."
        )

    if scaler.n_features_in_ != len(FEATURE_COLS):
        raise ValueError(
            f"Scaler was fitted on {scaler.n_features_in_} features but "
            f"FEATURE_COLS has {len(FEATURE_COLS)}. "
            "Regenerate scaler.pkl from the same training run."
        )

    if os.path.exists(COLS_PATH):
        with open(COLS_PATH) as fh:
            saved_cols = json.load(fh)
        if saved_cols != FEATURE_COLS:
            warnings.warn(
                "feature_columns.json does not match FEATURE_COLS in "
                "preprocessing.py — update one of them.",
                stacklevel=2,
            )

    return scaler


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    proc = df.copy()

    # Rename Dur → Duration
    if "Dur" in proc.columns:
        proc.rename(columns={"Dur": "Duration"}, inplace=True)

    # Numeric columns (non-port)
    for col in ["Duration", "TotBytes", "TotPkts", "SrcBytes", "sTos", "dTos"]:
        if col in proc.columns:
            proc[col] = pd.to_numeric(proc[col], errors="coerce").fillna(0)
        else:
            proc[col] = 0.0

    # Port columns — handle hex strings like 0xe11a
    proc["Sport"] = proc["Sport"].apply(_safe_port) if "Sport" in proc.columns else 0
    proc["Dport"] = proc["Dport"].apply(_safe_port) if "Dport" in proc.columns else 0

    # Guard division by zero
    proc["Duration"] = proc["Duration"].replace(0, 1e-6)
    proc["TotPkts"]  = proc["TotPkts"].replace(0, 1e-6)
    proc["TotBytes"] = proc["TotBytes"].replace(0, 1e-6)

    # Derived features
    proc["BytesPerSec"]  = proc["TotBytes"]  / proc["Duration"]
    proc["PktsPerSec"]   = proc["TotPkts"]   / proc["Duration"]
    proc["AvgPktSize"]   = proc["TotBytes"]  / proc["TotPkts"]
    proc["SrcByteRatio"] = proc["SrcBytes"]  / proc["TotBytes"]

    # Port privilege flags
    proc["Sport_is_priv"] = (proc["Sport"] <= 1024).astype(int)
    proc["Dport_is_priv"] = (proc["Dport"] <= 1024).astype(int)

    # Log1p transform
    for col in _LOG_COLS:
        proc[col] = np.log1p(proc[col].clip(lower=0))

    # Categorical encoding
    for col in ["Proto", "State", "Dir"]:
        if col in proc.columns:
            proc[col] = proc[col].astype(str).apply(_stable_hash)
        else:
            proc[col] = 0

    # sTos / dTos safety
    for col in ["sTos", "dTos"]:
        if col not in proc.columns:
            proc[col] = 0
        proc[col] = pd.to_numeric(proc[col], errors="coerce").fillna(0)

    # Assemble final DataFrame
    final = pd.DataFrame()
    for col in FEATURE_COLS:
        final[col] = pd.to_numeric(proc.get(col, 0), errors="coerce").fillna(0)

    return final


def preprocess(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    features_df = build_features(df)
    scaled = scaler.transform(features_df.values.astype(np.float32))
    return scaled.astype(np.float32)
