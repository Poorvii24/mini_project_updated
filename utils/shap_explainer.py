# =============================================================================
# utils/shap_explainer.py
# SHAP explainability for SST-Net TransformerClassifier.
# Uses GradientExplainer — fast, gradient-based, no approximation needed.
# =============================================================================

import numpy as np
import pandas as pd
import torch
import shap

from utils.preprocessing import FEATURE_COLS

# ── Human-readable feature descriptions for analyst UI ───────────────────────
FEATURE_DESCRIPTIONS = {
    'Duration':      'Flow duration (seconds)',
    'Proto':         'Protocol (TCP/UDP/ICMP encoded)',
    'Sport':         'Source port number',
    'Dir':           'Traffic direction',
    'Dport':         'Destination port number',
    'State':         'Connection state',
    'sTos':          'Source type-of-service',
    'dTos':          'Destination type-of-service',
    'TotPkts':       'Total packets (log-scaled)',
    'TotBytes':      'Total bytes transferred (log-scaled)',
    'SrcBytes':      'Bytes sent by source (log-scaled)',
    'BytesPerSec':   'Transfer rate — bytes/second (log-scaled)',
    'PktsPerSec':    'Transfer rate — packets/second (log-scaled)',
    'AvgPktSize':    'Average packet size in bytes (log-scaled)',
    'SrcByteRatio':  'Fraction of bytes from source (0–1)',
    'Sport_is_priv': 'Source port is privileged (≤1024)',
    'Dport_is_priv': 'Destination port is privileged (≤1024)',
}

# ── Known C2/botnet-associated destination ports ──────────────────────────────
_C2_PORTS = {6667, 6668, 6669, 1080, 4444, 31337, 8080, 9999, 1234}


# =============================================================================
# Core SHAP computation
# =============================================================================
def compute_shap_values(
    model: torch.nn.Module,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    device: str = "cpu",
    n_background: int = 100,
) -> np.ndarray:
    """Compute SHAP values using GradientExplainer.

    Parameters
    ----------
    model        : fitted TransformerClassifier in eval mode
    X_background : background dataset — representative sample of training data
                   shape (n, 17). Used to estimate expected feature values.
    X_explain    : flows to explain, shape (m, 17)
    device       : 'cpu' or 'cuda'
    n_background : number of background samples to use (more = slower)

    Returns
    -------
    np.ndarray shape (m, 17) — SHAP values for the BOTNET class (class 1)
    """
    model.eval()

    # Sample background to keep computation fast
    idx = np.random.choice(len(X_background),
                           size=min(n_background, len(X_background)),
                           replace=False)
    bg_tensor = torch.tensor(X_background[idx]).float().to(device)

    explainer   = shap.GradientExplainer(model, bg_tensor)
    X_tensor    = torch.tensor(X_explain).float().to(device)

    # shap_values shape: (n_samples, n_features, n_classes)
    shap_all    = explainer.shap_values(X_tensor)
    shap_array  = np.array(shap_all)            # (n_samples, 17, 2)

    # Return class-1 (botnet) SHAP values only
    return shap_array[:, :, 1]                  # (n_samples, 17)


# =============================================================================
# Per-alert analyst explanation
# =============================================================================
def explain_single_alert(
    shap_row: np.ndarray,
    raw_flow: pd.Series,
    confidence: float,
) -> dict:
    """Build analyst-friendly explanation for one alert.

    Parameters
    ----------
    shap_row   : 1-D array of length 17, SHAP values for this flow
    raw_flow   : original (unscaled) pandas Series for this row
    confidence : model probability for botnet class (0–1)

    Returns
    -------
    dict with keys:
        verdict        : 'MALICIOUS' | 'SUSPICIOUS' | 'BENIGN'
        confidence_pct : float (0–100)
        top_features   : list of dicts [{name, shap, direction, description}]
        natural_language: str — analyst-readable summary
        risk_indicators : list of str — specific red flags found
    """
    # ── Verdict ───────────────────────────────────────────────────────────────
    if confidence >= 0.7:
        verdict = "MALICIOUS"
    elif confidence >= 0.4:
        verdict = "SUSPICIOUS"
    else:
        verdict = "BENIGN"

    # ── Top contributing features (sorted by |SHAP|) ──────────────────────────
    feat_importance = sorted(
        zip(FEATURE_COLS, shap_row),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    top_features = []
    for name, val in feat_importance[:8]:
        top_features.append({
            "name":        name,
            "shap":        float(val),
            "direction":   "↑ Botnet" if val > 0 else "↓ Benign",
            "description": FEATURE_DESCRIPTIONS.get(name, name),
        })

    # ── Risk indicators (domain-specific red flags) ───────────────────────────
    indicators = []

    dport = float(raw_flow.get("Dport", 0)) if raw_flow is not None else 0
    sport = float(raw_flow.get("Sport", 0)) if raw_flow is not None else 0

    if dport in _C2_PORTS:
        indicators.append(f"Destination port {int(dport)} is a known C2/botnet port")
    if sport in _C2_PORTS:
        indicators.append(f"Source port {int(sport)} is a known C2/botnet port")

    duration = float(raw_flow.get("Duration", raw_flow.get("Dur", 1))) \
        if raw_flow is not None else 1
    tot_bytes = float(raw_flow.get("TotBytes", 0)) if raw_flow is not None else 0

    if duration > 0 and (tot_bytes / max(duration, 1e-6)) > 1_000_000:
        indicators.append("Extremely high transfer rate — possible data exfiltration")

    if duration < 0.01 and tot_bytes > 10_000:
        indicators.append("Large data volume in very short duration — scanning/flood pattern")

    src_ratio = float(raw_flow.get("SrcByteRatio", 0)) if raw_flow is not None else 0
    if src_ratio > 0.98:
        indicators.append("Nearly all bytes from source — one-directional C2 beacon pattern")

    # ── Natural language summary ──────────────────────────────────────────────
    top2 = [f["name"] for f in top_features[:2]]
    direction_words = {
        "TotBytes":      "large data volume",
        "BytesPerSec":   "high transfer rate",
        "Duration":      "flow duration",
        "PktsPerSec":    "packet rate",
        "Dport":         "destination port",
        "Sport":         "source port",
        "SrcByteRatio":  "source byte dominance",
        "AvgPktSize":    "packet size pattern",
        "Proto":         "protocol type",
        "State":         "connection state",
    }
    top2_words = " and ".join(
        direction_words.get(f, f) for f in top2
    )

    nl = (
        f"{verdict} — {confidence*100:.1f}% confidence. "
        f"Primary indicators: {top2_words}. "
    )
    if indicators:
        nl += "Red flags: " + "; ".join(indicators[:2]) + "."

    return {
        "verdict":         verdict,
        "confidence_pct":  round(confidence * 100, 1),
        "top_features":    top_features,
        "natural_language": nl,
        "risk_indicators": indicators,
    }


# =============================================================================
# Batch explanation helper (for dashboard use)
# =============================================================================
def explain_batch(
    model: torch.nn.Module,
    X_background: np.ndarray,
    X_suspicious: np.ndarray,
    probs_suspicious: np.ndarray,
    raw_df_suspicious: pd.DataFrame,
    device: str = "cpu",
    max_explain: int = 50,
) -> list[dict]:
    """Explain up to max_explain suspicious flows.

    Returns list of explanation dicts (one per flow), ready for dashboard.
    """
    # Normalize hex ports in raw df
    from utils.preprocessing import _safe_port
    if raw_df_suspicious is not None:
        for _col in ["Sport", "Dport"]:
            if _col in raw_df_suspicious.columns:
                raw_df_suspicious = raw_df_suspicious.copy()
                raw_df_suspicious[_col] = raw_df_suspicious[_col].apply(_safe_port)
    
    n = min(len(X_suspicious), max_explain)
    if n == 0:
        return []

    shap_vals = compute_shap_values(
        model, X_background, X_suspicious[:n], device=device
    )

    explanations = []
    for i in range(n):
        raw_flow = raw_df_suspicious.iloc[i] if raw_df_suspicious is not None else None
        exp = explain_single_alert(
            shap_row=shap_vals[i],
            raw_flow=raw_flow,
            confidence=float(probs_suspicious[i]),
        )
        exp["flow_index"] = i
        explanations.append(exp)

    return explanations


# =============================================================================
# Global feature importance (across all explained flows)
# =============================================================================
def global_importance(shap_matrix: np.ndarray) -> pd.DataFrame:
    """Mean absolute SHAP value per feature across all explained flows.

    Parameters
    ----------
    shap_matrix : (n_flows, 17) array of SHAP values

    Returns
    -------
    DataFrame sorted by importance descending, columns: feature, importance
    """
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": mean_abs,
        "description": [FEATURE_DESCRIPTIONS.get(f, f) for f in FEATURE_COLS],
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df
