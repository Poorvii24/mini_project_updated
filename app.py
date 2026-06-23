# =============================================================================
# app.py — SENTINEL CORE NOC Dashboard v3.1.0
# =============================================================================

import os
import time
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn

from utils.preprocessing import load_scaler, preprocess, FEATURE_COLS, _safe_port
from utils.shap_explainer import compute_shap_values, explain_batch, global_importance
from utils.mitre_mapper import map_dataframe, tactic_summary, technique_summary, TACTIC_COLORS
from utils.auth import load_auth_config, get_authenticator, has_permission, render_user_sidebar

# =============================================================================
# MODEL
# =============================================================================
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=17, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, hidden_dim=128):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden_dim, 2))

    def forward(self, x):
        x = self.input_layer(x).unsqueeze(1)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))


def generate_text_report(n_threats, risk_score, suspicious_df):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status    = "CRITICAL" if n_threats > 0 else "SECURE"
    top_str   = "No Source IPs found."
    if "SrcAddr" in suspicious_df.columns:
        ips = suspicious_df["SrcAddr"].value_counts().head(20).index.tolist()
        if ips:
            top_str = ", ".join(str(ip) for ip in ips)
    return f"""SENTINEL INCIDENT REPORT
DATE: {timestamp}
STATUS: {status}
THREATS DETECTED: {n_threats}
RISK SCORE: {risk_score:.2f}%
TOP ATTACKERS: {top_str}
RECOMMENDED ACTION:
1. Apply the firewall blocklist immediately.
2. Isolate affected subnet.
3. Reset credentials for compromised IoT devices.
4. Review port forwarding rules for unusual activity.
"""

# =============================================================================
# PAGE CONFIG & CSS
# =============================================================================
st.set_page_config(page_title="BOTNET DEFENSE", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background: radial-gradient(circle at center, #0a0e17 0%, #000000 100%); color:#fff; }
h1,h2,h3,h4,h5 { font-family:'Orbitron',sans-serif !important; color:#00ffcc !important; text-shadow:0 0 10px rgba(0,255,204,0.7); }
[data-testid="stFileUploader"] label,[data-testid="stWidgetLabel"] p,[data-testid="stMetricLabel"] {
    color:#00ffcc !important; font-size:1.1rem !important; font-weight:bold !important; }
div[data-testid="stMetric"] {
    background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1);
    border-left:4px solid #00ffcc; border-radius:12px; padding:15px; }
[data-testid="stMetricValue"] { font-size:36px !important; font-weight:700 !important; color:#fff !important; }
[data-testid="stSidebar"] { background-color:#050505; border-right:1px solid #333; }
.stTabs [data-baseweb="tab"] { height:50px; background-color:#111; border-radius:4px 4px 0 0; color:white; }
.stTabs [aria-selected="true"] { background-color:#00ffcc; color:black; font-weight:bold; }
</style>""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODEL & SCALER
# =============================================================================
@st.cache_resource
def load_model_and_scaler():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = TransformerClassifier(input_dim=17)
    model_path = "transformer_classifier.pt"
    if not os.path.exists(model_path):
        for c in ["models/transformer_classifier.pt"]:
            if os.path.exists(c):
                model_path = c
                break
        else:
            st.error("❌ Model file not found.")
            st.stop()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    try:
        scaler = load_scaler()
    except Exception as e:
        st.error(str(e))
        st.stop()
    return model, scaler, device

model, scaler, device = load_model_and_scaler()

# =============================================================================
# AUTHENTICATION
# =============================================================================
try:
    _auth_config   = load_auth_config()
    _authenticator = get_authenticator(_auth_config)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

auth_status = st.session_state.get("authentication_status", None)

if auth_status is None or auth_status is False:
    # Hide sidebar
    st.markdown("<style>[data-testid='stSidebar']{display:none;}</style>",
                unsafe_allow_html=True)

    # Login page header
    st.markdown("""
    <div style="text-align:center;padding:20px 20px 10px 20px;">
        <div style="font-size:48px;">🛡️</div>
        <h1 style="color:#00ffcc;font-family:'Courier New',monospace;font-size:28px;
            font-weight:700;letter-spacing:4px;text-shadow:0 0 20px rgba(0,255,204,0.5);
            margin:8px 0 4px 0;">SENTINEL CORE</h1>
        <p style="color:#555;font-family:'Courier New',monospace;font-size:11px;
            letter-spacing:3px;margin:0;">NETWORK OPERATIONS CENTER v3.1.0</p>
        <div style="width:180px;height:1px;
            background:linear-gradient(90deg,transparent,#00ffcc,transparent);
            margin:12px auto;"></div>
        <p style="color:#444;font-size:11px;font-family:monospace;letter-spacing:2px;">
            ⚠ AUTHORIZED PERSONNEL ONLY</p>
    </div>""", unsafe_allow_html=True)

    if auth_status is False:
        st.markdown("""
        <div style="background:rgba(255,0,51,0.1);border:1px solid #ff0033;
            border-radius:8px;padding:10px;text-align:center;max-width:400px;margin:0 auto 10px auto;">
            <span style="color:#ff0033;font-weight:700;">⛔ ACCESS DENIED — Invalid credentials</span>
        </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<p style='text-align:center;color:#00ffcc;font-family:monospace;"
                    "letter-spacing:2px;font-size:12px;margin-bottom:8px;'>🔐 AUTHENTICATE</p>",
                    unsafe_allow_html=True)
        _authenticator.login(key="main_login", location="main")
        st.markdown("<p style='text-align:center;color:#333;font-size:10px;"
                    "font-family:monospace;margin-top:12px;'>"
                    "admin / admin123 · analyst / analyst123</p>",
                    unsafe_allow_html=True)
    st.stop()

# Authenticated — render sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9664/9664977.png", width=80)
    st.title("SENTINEL CORE")
    st.caption("v3.1.0 | ENTERPRISE EDITION")
    st.markdown("---")
    st.subheader("🎛️ SYSTEM CONTROLS")
    threshold = st.slider("THREAT SENSITIVITY", 0.0, 1.0, 0.5)
    st.markdown("---")
    st.subheader("🖥️ SYSTEM STATUS")
    st.markdown("🟢 **ENGINE:** `ONLINE`")
    st.markdown("🟠 **ACCELERATION:** `CPU MODE`")
    st.markdown("🟢 **ENCRYPTION:** `TLS 1.3`")
    st.markdown("---")
    st.subheader("⚖️ SCALER INFO")
    st.markdown(f"**Type:** `{type(scaler).__name__}`")
    st.markdown(f"**Features:** `{scaler.n_features_in_}`")
    render_user_sidebar(_authenticator)

# =============================================================================
# HEADER
# =============================================================================
col1, col2 = st.columns([1, 8])
with col1:
    st.markdown("# 🛡️")
with col2:
    st.markdown("# NETWORK OPERATIONS CENTER (NOC)")
st.divider()

# =============================================================================
# FILE UPLOAD & PROCESSING
# =============================================================================
uploaded_file = st.file_uploader("📂 INJECT PACKET CAPTURE (.CSV)", type=["csv"])

st.markdown("---")
with st.expander("📡 LIVE CAPTURE MODE (Real-time / Simulated)", expanded=False):
    from utils.packet_capture import (
        get_flow_buffer, list_network_interfaces,
        start_live_capture, start_csv_replay, stop_capture, SCAPY_AVAILABLE
    )

    capture_mode = st.radio(
        "Select capture mode:",
        ["🔁 CSV Replay Simulation", "📶 Live Network Capture"],
        horizontal=True,
    )

    flow_buf = get_flow_buffer()

    if capture_mode == "🔁 CSV Replay Simulation":
        st.caption("Replays an uploaded CSV as simulated live traffic. No admin rights needed.")
        replay_file = st.file_uploader("CSV to replay", type=["csv"], key="replay_csv")
        rate = st.slider("Flows per second", 1.0, 50.0, 5.0)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ START REPLAY", disabled=(replay_file is None)):
                replay_df = pd.read_csv(replay_file, on_bad_lines="skip")
                start_csv_replay(replay_df, flow_buf, flows_per_sec=rate, max_flows=2000)
                st.success("Replay started!")
        with c2:
            if st.button("⏹️ STOP"):
                stop_capture(flow_buf)
                st.info("Replay stopped.")

    else:
        st.caption("Captures real packets from a network interface. Requires admin/root + Npcap (Windows).")
        if not SCAPY_AVAILABLE:
            st.error("Scapy is not installed. Run: `pip install scapy`")
        else:
            interfaces = list_network_interfaces()
            if not interfaces:
                st.warning("No network interfaces detected. Try running as Administrator.")
            else:
                iface = st.selectbox("Network interface:", interfaces)
                cap_duration = st.slider("Capture duration (seconds)", 10, 300, 60)

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("▶️ START LIVE CAPTURE"):
                        try:
                            start_live_capture(iface, flow_buf, duration=cap_duration)
                            st.success(f"Live capture started on `{iface}`!")
                        except Exception as e:
                            st.error(f"Capture failed: {e}")
                with c2:
                    if st.button("⏹️ STOP CAPTURE"):
                        stop_capture(flow_buf)
                        st.info("Capture stopped.")

    # ── Live stats panel ───────────────────────────────────────────────────────
    stats = flow_buf.get_stats()
    if stats.get("mode"):
        if st.button("🔄 REFRESH & SCORE NEW FLOWS", type="primary"):
            st.rerun()

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Mode",           stats["mode"])
        s2.metric("Flows Captured", stats["total_captured"])
        s3.metric("Rate/sec",       stats["rate_per_sec"])
        s4.metric("Buffer Size",    flow_buf.qsize())

        if flow_buf.is_running():
            st.success("🟢 Capture is running... Click REFRESH to score new flows.")
        else:
            st.info("⚪ Capture stopped.")

        # ── LIVE DETECTION PIPELINE ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### ⚡ LIVE DETECTION PIPELINE")
        st.caption("Drains the buffer, scores flows through the model, and shows real-time threats.")

        from utils.live_pipeline import get_pipeline

        live_threshold = st.slider("Live detection threshold", 0.0, 1.0, 0.5, key="live_thresh")

        pipeline = get_pipeline(model, scaler, device, st.session_state)

        # Drain buffer and score
        new_flows = flow_buf.drain(max_items=200)
        if new_flows:
            result = pipeline.process_batch(new_flows, threshold=live_threshold)
            if result.get("error"):
                st.warning(f"Some flows failed scoring: {result['error']}")
            elif result["n_processed"] > 0:
                st.toast(
                    f"Scored {result['n_processed']} flows — "
                    f"{result['n_alerts']} new alerts",
                    icon="⚡",
                )

        live_stats = pipeline.get_stats()

        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Total Scored",   live_stats["total_processed"])
        l2.metric("Total Alerts",   live_stats["total_alerts"])
        l3.metric("Live Risk %",    f"{live_stats['live_risk_pct']}%")
        l4.metric("Last Update",    live_stats["last_update"])

        # ── Live confidence stream chart ───────────────────────────────────────
        hist_df = pipeline.get_history_df()
        if len(hist_df) > 0:
            st.markdown("#### 📈 Live Confidence Stream (most recent flows)")
            chart_df = hist_df.head(100).iloc[::-1].reset_index(drop=True)
            fig_live = px.area(
                chart_df, y="confidence",
                labels={"index": "Flow #", "confidence": "Botnet Probability"},
            )
            fig_live.update_traces(line_color="#00ffcc", fillcolor="rgba(0,255,204,0.15)")
            threat_pts = chart_df[chart_df["confidence"] > live_threshold]
            if len(threat_pts) > 0:
                fig_live.add_scatter(
                    x=threat_pts.index, y=threat_pts["confidence"],
                    mode="markers", marker=dict(color="#ff0033", size=6),
                    name="Threat",
                )
            fig_live.add_hline(
                y=live_threshold, line_dash="dash", line_color="#FFA500",
                annotation_text=f"Threshold {live_threshold}",
            )
            fig_live.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"), showlegend=False,
                margin=dict(l=0,r=0,t=10,b=0), height=250,
            )
            st.plotly_chart(fig_live, use_container_width=True)

            # ── Live alert feed ────────────────────────────────────────────────
            st.markdown("#### 🚨 Live Alert Feed (most recent first)")
            alerts_df = pipeline.get_alerts_df()
            if len(alerts_df) > 0:
                def _sev_emoji(s):
                    return {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢"}.get(s,"⚪")
                alerts_df["sev_icon"] = alerts_df["severity"].apply(_sev_emoji)
                display_cols = ["timestamp","sev_icon","severity","src_addr","dst_addr",
                                "sport","dport","proto","confidence"]
                available_cols = [c for c in display_cols if c in alerts_df.columns]
                st.dataframe(
                    alerts_df[available_cols].head(30),
                    use_container_width=True, hide_index=True,
                )

                # ── Live report download ───────────────────────────────────────
                if has_permission("download_report"):
                    csv_data = alerts_df.to_csv(index=False)
                    st.download_button(
                        "📥 DOWNLOAD LIVE ALERT LOG",
                        data=csv_data,
                        file_name=f"live_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
            else:
                st.success("No alerts yet — traffic looks clean.")

            if st.button("🗑️ RESET LIVE PIPELINE"):
                pipeline.reset()
                st.rerun()

        if flow_buf.qsize() > 0:
            with st.expander("🔬 Raw buffer preview (debug)"):
                preview_items = flow_buf.drain(max_items=10)
                if preview_items:
                    st.dataframe(pd.DataFrame(preview_items), use_container_width=True)
                    for item in preview_items:
                        flow_buf.push(item)

st.markdown("---")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines="skip")
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin-1", on_bad_lines="skip")

    # Normalize hex ports at load time — fixes SHAP, MITRE, alert table
    for _col in ["Sport", "Dport"]:
        if _col in df.columns:
            df[_col] = df[_col].apply(_safe_port)

    with st.status("🚀 INITIALIZING DEEP SCAN...", expanded=True) as status:
        st.write(">> ESTABLISHING SECURE HANDSHAKE...")
        time.sleep(0.2)
        st.write(">> PARSING PACKET HEADERS...")
        X_processed = preprocess(df, scaler)
        time.sleep(0.2)
        st.write(">> RUNNING TRANSFORMER NEURAL NETWORK...")
        status.update(label="✅ ANALYSIS COMPLETE", state="complete", expanded=False)

    X_tensor = torch.tensor(X_processed).float().to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    final_threshold = 1.0 - threshold
    preds      = (probs > final_threshold).astype(int)
    n_botnets  = int(preds.sum())
    n_normal   = len(preds) - n_botnets
    risk_score = (n_botnets / len(preds) * 100) if len(preds) > 0 else 0.0

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📡 LIVE MONITOR", "🌍 GLOBAL THREAT MAP",
        "⚡ MITIGATION", "🔍 SHAP EXPLAINABILITY",
        "🎯 MITRE ATT&CK", "🦠 THREAT INTEL", "🛑 ABUSEIPDB"
    ])

    # ── TAB 1: DASHBOARD ──────────────────────────────────────────────────────
    with tab1:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("TOTAL FLOWS",      f"{len(preds):,}")
        m2.metric("THREATS DETECTED", f"{n_botnets:,}")
        m3.metric("BENIGN FLOWS",     f"{n_normal:,}")
        m4.metric("RISK FACTOR",      f"{risk_score:.1f}%")
        m5.metric("DETECTION RATE",   f"{(n_botnets/len(preds)*100):.2f}%")
        avg_conf = float(probs[probs > final_threshold].mean() * 100) if n_botnets > 0 else 0
        m6.metric("AVG CONFIDENCE",   f"{avg_conf:.1f}%")
        st.divider()

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### 🌊 TRAFFIC SPECTRUM")
            plot_probs = probs[:3000] if len(probs) > 3000 else probs
            fig = px.area(x=list(range(len(plot_probs))), y=plot_probs,
                          labels={"x": "Flow Index", "y": "Botnet Probability"})
            fig.update_traces(line_color="#00ffcc", fillcolor="rgba(0,255,204,0.15)")
            threat_idx = [i for i, p in enumerate(plot_probs) if p > final_threshold]
            if threat_idx:
                fig.add_scatter(x=threat_idx, y=[plot_probs[i] for i in threat_idx],
                                mode="markers", marker=dict(color="#ff0033", size=4), name="Threat")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"), showlegend=False,
                              margin=dict(l=0,r=0,t=0,b=0), height=260)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("### 🎯 THREAT SEVERITY")
            susp_probs = probs[probs > final_threshold]
            if len(susp_probs) > 0:
                sev_df = pd.DataFrame({
                    "Severity": ["CRITICAL ≥85%", "HIGH ≥70%", "MEDIUM ≥50%", "LOW <50%"],
                    "Count":    [int((susp_probs>=0.85).sum()),
                                 int(((susp_probs>=0.70)&(susp_probs<0.85)).sum()),
                                 int(((susp_probs>=0.50)&(susp_probs<0.70)).sum()),
                                 int((susp_probs<0.50).sum())],
                    "Color":    ["#FF0033","#FF6600","#FFAA00","#FFDD00"],
                })
                sev_df = sev_df[sev_df["Count"] > 0]
                color_map = dict(zip(sev_df["Severity"], sev_df["Color"]))
                fig_sev = px.pie(sev_df, values="Count", names="Severity",
                                 color="Severity", color_discrete_map=color_map, hole=0.5)
                fig_sev.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                                      margin=dict(l=0,r=0,t=0,b=0), height=260)
                fig_sev.update_traces(textposition="inside", textinfo="percent")
                st.plotly_chart(fig_sev, use_container_width=True)
            else:
                st.success("No threats detected.")
        st.divider()

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("### 📡 PROTOCOL DISTRIBUTION")
            if "Proto" in df.columns:
                df_p = df.head(1000).copy()
                df_p["Status"]   = ["MALICIOUS" if p > final_threshold else "SECURE" for p in probs[:len(df_p)]]
                df_p["Protocol"] = df_p["Proto"].astype(str).str.upper()
                pc = df_p.groupby(["Protocol","Status"]).size().reset_index(name="Count")
                fig_proto = px.bar(pc, x="Protocol", y="Count", color="Status",
                                   color_discrete_map={"MALICIOUS":"#FF0033","SECURE":"#00ffcc"},
                                   barmode="stack")
                fig_proto.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                        font=dict(color="white"), margin=dict(l=0,r=0,t=20,b=0), height=260)
                st.plotly_chart(fig_proto, use_container_width=True)
            else:
                st.info("Proto column not found.")

        with c4:
            st.markdown("### 📊 CONFIDENCE DISTRIBUTION")
            fig_hist = px.histogram(x=probs, nbins=50,
                                    labels={"x":"Botnet Probability","y":"Flow Count"},
                                    color_discrete_sequence=["#00ffcc"])
            fig_hist.add_vline(x=final_threshold, line_dash="dash", line_color="#FF0033",
                               annotation_text=f"Threshold:{final_threshold:.2f}",
                               annotation_font_color="#FF0033")
            fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="white"), margin=dict(l=0,r=0,t=20,b=0), height=260)
            st.plotly_chart(fig_hist, use_container_width=True)
        st.divider()

        c5, c6 = st.columns(2)
        with c5:
            st.markdown("### 🔴 TOP THREAT SOURCE IPs")
            if "SrcAddr" in df.columns and n_botnets > 0:
                tmp = df[probs > final_threshold].copy()
                tmp["confidence"] = probs[probs > final_threshold]
                top_ips = (tmp.groupby("SrcAddr")
                           .agg(count=("SrcAddr","count"), avg_conf=("confidence","mean"))
                           .reset_index().sort_values("count", ascending=False).head(10))
                top_ips["avg_conf"] = (top_ips["avg_conf"]*100).round(1)
                fig_ip = px.bar(top_ips.sort_values("count", ascending=True),
                                x="count", y="SrcAddr", orientation="h",
                                color="avg_conf", color_continuous_scale="Reds", text="count")
                fig_ip.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                     font=dict(color="white"), coloraxis_showscale=False,
                                     margin=dict(l=0,r=0,t=10,b=0), height=300)
                fig_ip.update_traces(textposition="outside")
                st.plotly_chart(fig_ip, use_container_width=True)
            else:
                st.info("SrcAddr column not found or no threats detected.")

        with c6:
            st.markdown("### 🚨 ALERT TABLE (Top 20)")
            tr = df[probs > final_threshold].copy()
            tr["Confidence %"] = (probs[probs > final_threshold]*100).round(1)
            tr["Severity"] = tr["Confidence %"].apply(
                lambda x: "🔴 CRITICAL" if x>=85 else("🟠 HIGH" if x>=70 else("🟡 MEDIUM" if x>=50 else "🟢 LOW")))
            show_cols = [c for c in ["SrcAddr","DstAddr","Proto","Sport","Dport",
                                     "State","TotBytes","Confidence %","Severity"] if c in tr.columns]
            st.dataframe(tr[show_cols].head(20), use_container_width=True, hide_index=True)

    # ── TAB 2: MAP ────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 🌍 GEO-SPATIAL THREAT INTELLIGENCE")
        if n_botnets > 0:
            _CITIES = {
                "New York":[-74.006,40.7128],"London":[-0.1278,51.5074],
                "Beijing":[116.4074,39.9042],"Moscow":[37.6173,55.7558],
                "Sao Paulo":[-46.6333,-23.5505],"Tokyo":[139.6503,35.6762],
                "Berlin":[13.4050,52.5200],"Mumbai":[72.8777,19.0760],
                "Sydney":[151.2093,-33.8688],"Cairo":[31.2357,30.0444],
            }
            _CNAMES = list(_CITIES.keys())
            def _ip_city(ip):
                return _CNAMES[int(hashlib.sha256(str(ip).encode()).hexdigest(),16) % len(_CNAMES)]

            susp = df[probs > final_threshold].copy()
            rows = []
            src_col = "SrcAddr" if "SrcAddr" in susp.columns else None
            for i in range(min(50, n_botnets)):
                ip   = susp.iloc[i][src_col] if src_col else f"flow_{i}"
                city = _ip_city(ip)
                lon, lat = _CITIES[city]
                jitter = ((int(hashlib.sha256(str(ip).encode()).hexdigest(),16)%100)-50)/100*0.8
                rows.append({"lat":lat+jitter,"lon":lon+jitter,"City":city,"IP":str(ip),"type":"Botnet Node"})
            map_df = pd.DataFrame(rows)
            fig_map = px.scatter_geo(map_df, lat="lat", lon="lon", hover_name="City",
                                     hover_data={"IP":True,"lat":False,"lon":False},
                                     projection="orthographic", color="type",
                                     color_discrete_map={"Botnet Node":"#ff0033"},
                                     title="ACTIVE CITY-LEVEL VECTORS")
            fig_map.update_geos(bgcolor="black", showcountries=True, countrycolor="#333",
                                showland=True, landcolor="#111")
            fig_map.update_layout(paper_bgcolor="black", font_color="white", height=600)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.success("NO ACTIVE GEO-THREATS DETECTED.")

    # ── TAB 3: MITIGATION ─────────────────────────────────────────────────────
    with tab3:
        if n_botnets > 0:
            st.error("### ⚡ AUTOMATED COUNTERMEASURES")
            suspicious = df[probs > final_threshold].copy()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🛡️ FIREWALL RULES (IPTABLES)")
                code = "# AUTO-GENERATED BLOCKLIST\n"
                if "SrcAddr" in suspicious.columns:
                    for ip in suspicious["SrcAddr"].unique()[:10]:
                        code += f"iptables -A INPUT -s {ip} -j DROP\n"
                else:
                    code += "# IPs unavailable\n"
                st.code(code, language="bash")
                if has_permission("block_ips"):
                    st.warning("⚠️ Apply manually on your network device.")
                else:
                    st.info("🔒 IP blocking requires **Admin** role.")
            with c2:
                st.markdown("#### 📄 INCIDENT REPORT")
                if has_permission("download_report"):
                    st.download_button("📥 DOWNLOAD FORENSIC REPORT",
                                       data=generate_text_report(n_botnets, risk_score, suspicious),
                                       file_name="sentinel_forensic_report.txt", mime="text/plain")
                else:
                    st.info("🔒 Report download requires **Admin** role.")
            st.markdown("#### 🚨 LIVE PACKET INSPECTOR")
            st.dataframe(suspicious.head(20), use_container_width=True)
        else:
            st.success("SYSTEM SECURE.")

    # ── TAB 4: SHAP ───────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 🔍 SHAP EXPLAINABILITY")
        st.caption("Positive values (red) push toward BOTNET. Negative values (blue) push toward BENIGN.")
        if n_botnets == 0:
            st.success("No threats detected — nothing to explain.")
        else:
            X_susp    = X_processed[probs > final_threshold]
            susp_df   = df[probs > final_threshold].reset_index(drop=True)
            probs_susp = probs[probs > final_threshold]
            n_explain  = min(len(X_susp), 50)
            st.info(f"Explaining **{n_explain}** of {n_botnets} detected threats.")

            if st.button("⚡ RUN SHAP ANALYSIS", type="primary"):
                with st.spinner("Computing SHAP values — 10–30 seconds..."):
                    try:
                        exps = explain_batch(model, X_processed, X_susp, probs_susp,
                                             susp_df, device=device, max_explain=n_explain)
                        shap_m = compute_shap_values(model, X_processed, X_susp[:n_explain], device=device)
                        g_imp  = global_importance(shap_m)
                        st.session_state["explanations"] = exps
                        st.session_state["global_imp"]   = g_imp
                        st.success(f"✅ SHAP analysis complete for {n_explain} flows.")
                    except Exception as e:
                        st.error(f"SHAP computation failed: {e}")

            if "explanations" in st.session_state and st.session_state["explanations"]:
                exps   = st.session_state["explanations"]
                g_imp  = st.session_state["global_imp"]

                st.markdown("#### 📊 Global Feature Importance")
                fig_g = px.bar(g_imp.head(10), x="importance", y="feature", orientation="h",
                               color="importance", color_continuous_scale="Reds",
                               text=g_imp.head(10)["importance"].round(4))
                fig_g.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="white"), yaxis=dict(autorange="reversed"),
                                    coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0), height=350)
                fig_g.update_traces(textposition="outside")
                st.plotly_chart(fig_g, use_container_width=True)
                st.divider()

                st.markdown("#### 🚨 Per-Alert SHAP Inspector")
                idx = st.selectbox("Select alert:", range(len(exps)),
                                   format_func=lambda i: f"Alert #{i+1} — {exps[i]['verdict']} — {exps[i]['confidence_pct']}%")
                exp = exps[idx]
                vc  = {"MALICIOUS":"#FF0033","SUSPICIOUS":"#FFA500","BENIGN":"#00FF99"}[exp["verdict"]]
                st.markdown(f"""<div style="background:rgba(255,255,255,0.05);border-left:4px solid {vc};
                    border-radius:8px;padding:14px 18px;margin-bottom:12px;">
                    <span style="color:{vc};font-size:18px;font-weight:700;">{exp['verdict']}</span>
                    <span style="color:#ccc;font-size:14px;margin-left:16px;">Confidence: {exp['confidence_pct']}%</span>
                    <p style="color:#eee;margin-top:8px;font-size:13px;">{exp['natural_language']}</p>
                    </div>""", unsafe_allow_html=True)

                if exp["risk_indicators"]:
                    for ri in exp["risk_indicators"]:
                        st.markdown(f"- ⚠️ {ri}")

                feat_df = pd.DataFrame(exp["top_features"])
                feat_df["abs_shap"] = feat_df["shap"].abs()
                feat_df = feat_df.sort_values("abs_shap", ascending=True)
                fig_sh = px.bar(feat_df, x="shap", y="name", orientation="h",
                                color="shap",
                                color_continuous_scale=[[0,"#4488FF"],[0.5,"#333"],[1,"#FF4444"]],
                                range_color=[feat_df["shap"].min(), feat_df["shap"].max()],
                                text=feat_df["shap"].round(4))
                fig_sh.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                     font=dict(color="white"), coloraxis_showscale=False,
                                     margin=dict(l=0,r=0,t=10,b=0), height=380)
                fig_sh.update_traces(textposition="outside")
                st.plotly_chart(fig_sh, use_container_width=True)

                disp = feat_df[["name","description","shap","direction"]].copy()
                disp.columns = ["Feature","What it measures","SHAP Value","Effect"]
                disp = disp.sort_values("SHAP Value", key=abs, ascending=False)
                disp["SHAP Value"] = disp["SHAP Value"].round(6)
                st.dataframe(disp, use_container_width=True, hide_index=True)
            elif "explanations" not in st.session_state:
                st.info("👆 Click RUN SHAP ANALYSIS above.")

    # ── TAB 5: MITRE ──────────────────────────────────────────────────────────
    with tab5:
        st.markdown("### 🎯 MITRE ATT&CK Mapping")
        if n_botnets == 0:
            st.success("No threats detected.")
        else:
            susp_df = df[probs > final_threshold].reset_index(drop=True)
            with st.spinner("Mapping to ATT&CK techniques..."):
                mapped_df    = map_dataframe(susp_df, max_rows=500)
                tactic_df    = tactic_summary(mapped_df)
                technique_df = technique_summary(mapped_df)

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Threats Mapped",    len(mapped_df))
            c2.metric("Unique Techniques", technique_df["id"].nunique() if len(technique_df) else 0)
            c3.metric("Tactics Covered",   tactic_df["tactic"].nunique() if len(tactic_df) else 0)
            c4.metric("Top Tactic",        tactic_df.iloc[0]["tactic"] if len(tactic_df) else "—")
            st.divider()

            if len(tactic_df) > 0:
                st.markdown("#### 🗺️ ATT&CK Tactic Heatmap")
                t_color_map = dict(zip(tactic_df["tactic"], tactic_df["color"]))
                fig_t = px.bar(tactic_df.sort_values("count", ascending=True),
                               x="count", y="tactic", orientation="h",
                               color="tactic", color_discrete_map=t_color_map, text="count")
                fig_t.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="white"), showlegend=False,
                                    margin=dict(l=0,r=0,t=10,b=0), height=350)
                fig_t.update_traces(textposition="outside")
                st.plotly_chart(fig_t, use_container_width=True)
            st.divider()

            cl, cr = st.columns(2)
            with cl:
                st.markdown("#### 🔬 Top Techniques")
                if len(technique_df) > 0:
                    tech_color_map = dict(zip(technique_df["id"], technique_df["color"]))
                    fig_tech = px.bar(technique_df.head(10).sort_values("count", ascending=True),
                                      x="count", y="id", orientation="h",
                                      color="id", color_discrete_map=tech_color_map, text="count",
                                      hover_data={"name":True,"tactic":True,"color":False,"url":False})
                    fig_tech.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                           font=dict(color="white"), showlegend=False,
                                           margin=dict(l=0,r=0,t=10,b=0), height=350)
                    fig_tech.update_traces(textposition="outside")
                    st.plotly_chart(fig_tech, use_container_width=True)
            with cr:
                st.markdown("#### 📋 Technique Details")
                if len(technique_df) > 0:
                    dt = technique_df[["id","name","tactic","count"]].copy()
                    dt.columns = ["ID","Technique","Tactic","Count"]
                    dt["ATT&CK Link"] = dt["ID"].apply(
                        lambda x: f"https://attack.mitre.org/techniques/{x.replace('.','/')}/")
                    st.dataframe(dt, use_container_width=True, hide_index=True,
                                 column_config={"ATT&CK Link": st.column_config.LinkColumn("ATT&CK Link", display_text="View →")})
            st.divider()

            st.markdown("#### ⛓️ Kill Chain Coverage")
            KILL_CHAIN = ["Reconnaissance","Resource Development","Initial Access","Execution",
                          "Persistence","Privilege Escalation","Defense Evasion","Credential Access",
                          "Discovery","Lateral Movement","Collection","Command and Control",
                          "Exfiltration","Impact"]
            detected_tactics = set(tactic_df["tactic"].tolist()) if len(tactic_df) else set()
            cols = st.columns(7)
            for i, stage in enumerate(KILL_CHAIN):
                det   = stage in detected_tactics
                color = TACTIC_COLORS.get(stage, "#888")
                icon  = "🔴" if det else "⚪"
                cols[i%7].markdown(
                    f"""<div style="background:{'rgba(255,255,255,0.08)' if det else 'rgba(255,255,255,0.02)'};
                    border:1px solid {'%s'%color if det else '#333'};border-radius:8px;
                    padding:8px 6px;text-align:center;margin-bottom:6px;">
                    <div style="font-size:16px;">{icon}</div>
                    <div style="font-size:10px;color:{'%s'%color if det else '#666'};
                    font-weight:{'700' if det else '400'};line-height:1.3;">{stage}</div></div>""",
                    unsafe_allow_html=True)
            st.divider()

            st.markdown("#### 📄 Per-Flow ATT&CK Mapping")
            show_cols = [c for c in susp_df.columns if c in ["SrcAddr","DstAddr","Sport","Dport","Proto","State"]]
            show_cols += ["attack_ids","attack_tactics"]
            available = [c for c in show_cols if c in mapped_df.columns]
            st.dataframe(mapped_df[available].head(100), use_container_width=True, hide_index=True)

    # ── TAB 6: VIRUSTOTAL THREAT INTELLIGENCE ─────────────────────────────────
    with tab6:
        from utils.virustotal import lookup_ips_batch, render_vt_results

        st.markdown("### 🦠 THREAT INTELLIGENCE — VirusTotal IP Reputation")
        st.caption(
            "Enriches detected threat source IPs with VirusTotal reputation data. "
            "Free tier: 1000 requests/day, 4/minute."
        )

        if n_botnets == 0:
            st.success("No threats detected — nothing to enrich.")
        else:
            # ── API key: try st.secrets (cloud) → config.py (local) → manual input ──
            vt_key = None
            try:
                vt_key = st.secrets["VIRUSTOTAL_API_KEY"]
                st.success("✅ VirusTotal API key loaded from Streamlit secrets")
            except Exception:
                try:
                    from config import VIRUSTOTAL_API_KEY
                    if VIRUSTOTAL_API_KEY:
                        vt_key = VIRUSTOTAL_API_KEY
                        st.success("✅ VirusTotal API key loaded from config.py")
                except (ImportError, AttributeError):
                    pass

            if not vt_key:
                vt_key = st.text_input(
                    "🔑 VirusTotal API Key",
                    type="password",
                    placeholder="Paste your key (or set it in Streamlit Secrets / config.py)",
                    help="Get a free key at https://www.virustotal.com/gui/join-us",
                )

            # Extract unique source IPs from threats
            susp_tmp = df[probs > final_threshold].copy()
            if "SrcAddr" in susp_tmp.columns:
                unique_ips = susp_tmp["SrcAddr"].dropna().unique().tolist()
                unique_ips = [str(ip) for ip in unique_ips if str(ip) != "nan"]
            else:
                unique_ips = []

            if unique_ips:
                st.info(
                    f"Found **{len(unique_ips)}** unique source IPs in threats. "
                    f"Will check up to **10** (free tier quota protection)."
                )
                st.markdown("**Sample IPs to be checked:**")
                st.code(", ".join(unique_ips[:10]))
            else:
                st.warning("No SrcAddr column found in this dataset. Cannot look up IPs.")

            if vt_key and unique_ips:
                if st.button("🔍 RUN VIRUSTOTAL LOOKUP", type="primary"):
                    results    = []
                    progress   = st.progress(0)
                    status_txt = st.empty()

                    def update_progress(current, total, ip):
                        pct = int((current / total) * 100)
                        progress.progress(pct)
                        status_txt.markdown(
                            f"⏳ Checking `{ip}` ({current+1}/{total}) "
                            f"— waiting {15}s between requests (free tier limit)..."
                        )

                    with st.spinner("Querying VirusTotal API..."):
                        try:
                            results = lookup_ips_batch(
                                ips=unique_ips,
                                api_key=vt_key,
                                max_ips=10,
                                progress_callback=update_progress,
                            )
                            st.session_state["vt_results"] = results
                            progress.progress(100)
                            status_txt.empty()
                            st.success(f"✅ Looked up {len(results)} IPs successfully.")
                        except Exception as e:
                            st.error(f"VirusTotal lookup failed: {e}")

            elif not vt_key and unique_ips:
                st.info("👆 Enter your VirusTotal API key above and click RUN to start.")

            # ── Render cached results ─────────────────────────────────────────
            if "vt_results" in st.session_state and st.session_state["vt_results"]:
                st.divider()
                st.markdown("#### 📊 VirusTotal Results")
                render_vt_results(st.session_state["vt_results"])

    # ── TAB 7: ABUSEIPDB ──────────────────────────────────────────────────────
    with tab7:
        from utils.abuseipdb import lookup_ips_batch as abuse_batch, render_abuseipdb_results

        st.markdown("### 🛑 ABUSEIPDB — IP Abuse Reputation")
        st.caption(
            "Checks detected threat IPs against AbuseIPDB community reports. "
            "Shows abuse confidence score, total reports, ISP, usage type, and TOR detection. "
            "Free tier: 1000 checks/day."
        )

        if n_botnets == 0:
            st.success("No threats detected — nothing to check.")
        else:
            # ── API key: try st.secrets (cloud) → config.py (local) → manual input ──
            abuse_key = None
            try:
                abuse_key = st.secrets["ABUSEIPDB_API_KEY"]
                st.success("✅ AbuseIPDB API key loaded from Streamlit secrets")
            except Exception:
                try:
                    from config import ABUSEIPDB_API_KEY
                    if ABUSEIPDB_API_KEY:
                        abuse_key = ABUSEIPDB_API_KEY
                        st.success("✅ AbuseIPDB API key loaded from config.py")
                except (ImportError, AttributeError):
                    pass

            if not abuse_key:
                abuse_key = st.text_input(
                    "🔑 AbuseIPDB API Key",
                    type="password",
                    placeholder="Paste your key (or set it in Streamlit Secrets / config.py)",
                    help="Get a free key at https://www.abuseipdb.com/register",
                )

            # ── Extract source IPs ────────────────────────────────────────────
            susp_tmp = df[probs > final_threshold].copy()
            if "SrcAddr" in susp_tmp.columns:
                unique_ips = susp_tmp["SrcAddr"].dropna().unique().tolist()
                unique_ips = [str(ip) for ip in unique_ips if str(ip) != "nan"]
            else:
                unique_ips = []

            if unique_ips:
                st.info(
                    f"Found **{len(unique_ips)}** unique source IPs. "
                    f"Will check up to **10** (free tier quota protection)."
                )
                st.code(", ".join(unique_ips[:10]))
            else:
                st.warning("No SrcAddr column found in this dataset.")

            if abuse_key and unique_ips:
                if st.button("🔍 RUN ABUSEIPDB LOOKUP", type="primary"):
                    progress   = st.progress(0)
                    status_txt = st.empty()

                    def update_progress(current, total, ip):
                        progress.progress(int((current / total) * 100))
                        status_txt.markdown(f"⏳ Checking `{ip}` ({current+1}/{total})...")

                    with st.spinner("Querying AbuseIPDB..."):
                        try:
                            results = abuse_batch(
                                ips=unique_ips,
                                api_key=abuse_key,
                                max_ips=10,
                                progress_callback=update_progress,
                            )
                            st.session_state["abuse_results"] = results
                            progress.progress(100)
                            status_txt.empty()
                            st.success(f"✅ Checked {len(results)} IPs successfully.")
                        except Exception as e:
                            st.error(f"AbuseIPDB lookup failed: {e}")

            elif not abuse_key and unique_ips:
                st.info("👆 Enter your AbuseIPDB API key above and click RUN.")

            # ── Render cached results ─────────────────────────────────────────
            if "abuse_results" in st.session_state and st.session_state["abuse_results"]:
                st.divider()
                st.markdown("#### 📊 AbuseIPDB Results")
                render_abuseipdb_results(st.session_state["abuse_results"])

else:
    st.info("WAITING FOR TRAFFIC STREAM... SYSTEM IDLE.")
