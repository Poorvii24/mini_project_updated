# =============================================================================
# utils/mitre_mapper.py
# MITRE ATT&CK mapping for SST-Net detections.
# =============================================================================

import pandas as pd
import numpy as np
from dataclasses import dataclass

# =============================================================================
# ATT&CK Technique definitions
# =============================================================================
@dataclass
class ATTACKTechnique:
    id:          str
    name:        str
    tactic:      str
    description: str
    url:         str
    color:       str


TACTIC_COLORS = {
    "Reconnaissance":       "#9b59b6",
    "Resource Development": "#8e44ad",
    "Initial Access":       "#e74c3c",
    "Execution":            "#c0392b",
    "Persistence":          "#e67e22",
    "Privilege Escalation": "#d35400",
    "Defense Evasion":      "#f39c12",
    "Credential Access":    "#f1c40f",
    "Discovery":            "#2ecc71",
    "Lateral Movement":     "#27ae60",
    "Collection":           "#1abc9c",
    "Command and Control":  "#3498db",
    "Exfiltration":         "#2980b9",
    "Impact":               "#e74c3c",
}

TECHNIQUES: dict[str, ATTACKTechnique] = {
    "T1595": ATTACKTechnique(
        id="T1595", name="Active Scanning", tactic="Reconnaissance",
        description="Adversary scans victim infrastructure.",
        url="https://attack.mitre.org/techniques/T1595/",
        color=TACTIC_COLORS["Reconnaissance"],
    ),
    "T1595.001": ATTACKTechnique(
        id="T1595.001", name="Scanning IP Blocks", tactic="Reconnaissance",
        description="Scanning sequential IP ranges.",
        url="https://attack.mitre.org/techniques/T1595/001/",
        color=TACTIC_COLORS["Reconnaissance"],
    ),
    "T1595.002": ATTACKTechnique(
        id="T1595.002", name="Vulnerability Scanning", tactic="Reconnaissance",
        description="Scanning for open ports and vulnerable services.",
        url="https://attack.mitre.org/techniques/T1595/002/",
        color=TACTIC_COLORS["Reconnaissance"],
    ),
    "T1071": ATTACKTechnique(
        id="T1071", name="Application Layer Protocol", tactic="Command and Control",
        description="Adversary uses application-layer protocols for C2.",
        url="https://attack.mitre.org/techniques/T1071/",
        color=TACTIC_COLORS["Command and Control"],
    ),
    "T1071.001": ATTACKTechnique(
        id="T1071.001", name="Web Protocols (HTTP/HTTPS C2)", tactic="Command and Control",
        description="C2 over HTTP/HTTPS (ports 80, 443, 8080, 8443).",
        url="https://attack.mitre.org/techniques/T1071/001/",
        color=TACTIC_COLORS["Command and Control"],
    ),
    "T1071.003": ATTACKTechnique(
        id="T1071.003", name="Mail Protocols", tactic="Command and Control",
        description="C2 using mail protocols SMTP/IMAP/POP3.",
        url="https://attack.mitre.org/techniques/T1071/003/",
        color=TACTIC_COLORS["Command and Control"],
    ),
    "T1071.004": ATTACKTechnique(
        id="T1071.004", name="DNS C2", tactic="Command and Control",
        description="C2 over DNS (port 53).",
        url="https://attack.mitre.org/techniques/T1071/004/",
        color=TACTIC_COLORS["Command and Control"],
    ),
    "T1095": ATTACKTechnique(
        id="T1095", name="Non-Application Layer Protocol", tactic="Command and Control",
        description="C2 using ICMP, UDP, or raw TCP.",
        url="https://attack.mitre.org/techniques/T1095/",
        color=TACTIC_COLORS["Command and Control"],
    ),
    "T1048": ATTACKTechnique(
        id="T1048", name="Exfiltration Over Alternative Protocol", tactic="Exfiltration",
        description="Data exfiltration using alternative protocols.",
        url="https://attack.mitre.org/techniques/T1048/",
        color=TACTIC_COLORS["Exfiltration"],
    ),
    "T1041": ATTACKTechnique(
        id="T1041", name="Exfiltration Over C2 Channel", tactic="Exfiltration",
        description="Exfiltration over existing C2 channel.",
        url="https://attack.mitre.org/techniques/T1041/",
        color=TACTIC_COLORS["Exfiltration"],
    ),
    "T1498": ATTACKTechnique(
        id="T1498", name="Network Denial of Service", tactic="Impact",
        description="Flooding attack to degrade availability.",
        url="https://attack.mitre.org/techniques/T1498/",
        color=TACTIC_COLORS["Impact"],
    ),
    "T1571": ATTACKTechnique(
        id="T1571", name="Non-Standard Port", tactic="Command and Control",
        description="C2 communication over non-standard ports.",
        url="https://attack.mitre.org/techniques/T1571/",
        color=TACTIC_COLORS["Command and Control"],
    ),
    "T1572": ATTACKTechnique(
        id="T1572", name="Protocol Tunneling", tactic="Command and Control",
        description="Encapsulating traffic inside allowed protocols.",
        url="https://attack.mitre.org/techniques/T1572/",
        color=TACTIC_COLORS["Command and Control"],
    ),
    "T1018": ATTACKTechnique(
        id="T1018", name="Remote System Discovery", tactic="Discovery",
        description="Enumerating hosts on the network.",
        url="https://attack.mitre.org/techniques/T1018/",
        color=TACTIC_COLORS["Discovery"],
    ),
    "T1046": ATTACKTechnique(
        id="T1046", name="Network Service Discovery", tactic="Discovery",
        description="Port scanning to identify running services.",
        url="https://attack.mitre.org/techniques/T1046/",
        color=TACTIC_COLORS["Discovery"],
    ),
    "T1219": ATTACKTechnique(
        id="T1219", name="Remote Access Software", tactic="Command and Control",
        description="Legitimate remote access tools used for C2.",
        url="https://attack.mitre.org/techniques/T1219/",
        color=TACTIC_COLORS["Command and Control"],
    ),
    "T1090": ATTACKTechnique(
        id="T1090", name="Proxy", tactic="Command and Control",
        description="Traffic routed through proxy to obscure origin.",
        url="https://attack.mitre.org/techniques/T1090/",
        color=TACTIC_COLORS["Command and Control"],
    ),
    "T1132": ATTACKTechnique(
        id="T1132", name="Data Encoding", tactic="Command and Control",
        description="Encoding C2 data to obscure content.",
        url="https://attack.mitre.org/techniques/T1132/",
        color=TACTIC_COLORS["Command and Control"],
    ),
}

# =============================================================================
# Port to technique mapping
# =============================================================================
_PORT_MAP: dict[int, list[str]] = {
    6667:  ["T1071", "T1571"],
    6668:  ["T1071", "T1571"],
    6669:  ["T1071", "T1571"],
    1080:  ["T1090", "T1571"],
    4444:  ["T1071", "T1571"],
    31337: ["T1071", "T1571"],
    9999:  ["T1071", "T1571"],
    1234:  ["T1071", "T1571"],
    80:    ["T1071.001"],
    8080:  ["T1071.001", "T1571"],
    8443:  ["T1071.001"],
    443:   ["T1071.001"],
    53:    ["T1071.004", "T1572"],
    25:    ["T1071.003"],
    143:   ["T1071.003"],
    110:   ["T1071.003"],
    3389:  ["T1219"],
    5900:  ["T1219"],
    22:    ["T1219"],
    23:    ["T1595.002"],
    445:   ["T1046"],
    21:    ["T1595.002"],
    3306:  ["T1595.002"],
    5432:  ["T1595.002"],
}


# =============================================================================
# Helper: safe port parsing (handles hex strings like 0xe11a)
# =============================================================================
def _safe_port(val) -> int:
    try:
        s = str(val).strip()
        if s.startswith('0x') or s.startswith('0X'):
            return int(s, 16)
        return int(float(s))
    except (ValueError, TypeError):
        return 0


# =============================================================================
# Feature-signature to technique mapping
# =============================================================================
def _map_by_features(flow_features: dict) -> list[str]:
    techniques = []

    duration  = float(flow_features.get("Duration", 1) or 1e-6)
    tot_bytes = float(flow_features.get("TotBytes",  0) or 0)
    tot_pkts  = float(flow_features.get("TotPkts",   0) or 0)
    src_bytes = float(flow_features.get("SrcBytes",  0) or 0)
    bps       = float(flow_features.get("BytesPerSec", 0) or 0)
    pps       = float(flow_features.get("PktsPerSec",  0) or 0)
    avg_pkt   = float(flow_features.get("AvgPktSize",  0) or 0)
    src_ratio = float(flow_features.get("SrcByteRatio", 0) or 0)
    dport     = _safe_port(flow_features.get("Dport", 0))
    sport     = _safe_port(flow_features.get("Sport", 0))

    # Port-based mapping
    for port in [dport, sport]:
        if port in _PORT_MAP:
            techniques.extend(_PORT_MAP[port])

    # Scanning pattern
    if duration < 0.5 and tot_pkts <= 3 and tot_bytes < 500:
        techniques.extend(["T1595", "T1595.001", "T1046"])

    # Port scan
    if dport <= 1024 and duration < 0.1 and tot_pkts <= 2:
        techniques.extend(["T1595.002", "T1046"])

    # Exfiltration
    if tot_bytes > 100_000 and src_ratio > 0.85:
        techniques.extend(["T1048", "T1041"])

    # DDoS
    if pps > 10_000 and avg_pkt < 100:
        techniques.append("T1498")

    # Beaconing
    if 0.01 < duration < 2.0 and tot_bytes < 1000 and src_ratio > 0.6:
        techniques.extend(["T1071", "T1132"])

    # DNS tunneling
    if dport == 53 and avg_pkt > 200:
        techniques.extend(["T1572", "T1071.004"])

    # Non-standard port C2
    if dport > 1024 and dport not in _PORT_MAP and tot_bytes > 1000:
        techniques.append("T1571")

    # Deduplicate
    seen = set()
    result = []
    for t in techniques:
        if t not in seen and t in TECHNIQUES:
            seen.add(t)
            result.append(t)

    return result if result else ["T1071"]


# =============================================================================
# Public API
# =============================================================================
def map_flow_to_techniques(raw_flow: pd.Series) -> list[ATTACKTechnique]:
    technique_ids = _map_by_features(raw_flow.to_dict())
    return [TECHNIQUES[tid] for tid in technique_ids if tid in TECHNIQUES]


def map_dataframe(raw_df: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame:
    """Map all flows in a DataFrame to ATT&CK techniques."""
    df = raw_df.head(max_rows).copy()

    # Normalize hex port values before mapping
    for col in ["Sport", "Dport"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_port)

    ids_list     = []
    names_list   = []
    tactics_list = []
    primary_list = []

    for _, row in df.iterrows():
        techs = map_flow_to_techniques(row)
        if techs:
            ids_list.append(", ".join(t.id   for t in techs))
            names_list.append(", ".join(t.name for t in techs))
            tactics_list.append(", ".join(dict.fromkeys(t.tactic for t in techs)))
            primary_list.append(techs[0].id)
        else:
            ids_list.append("T1071")
            names_list.append("Application Layer Protocol")
            tactics_list.append("Command and Control")
            primary_list.append("T1071")

    df["attack_ids"]     = ids_list
    df["attack_names"]   = names_list
    df["attack_tactics"] = tactics_list
    df["attack_primary"] = primary_list

    return df


def tactic_summary(mapped_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tactics_str in mapped_df["attack_tactics"].dropna():
        for tactic in tactics_str.split(", "):
            tactic = tactic.strip()
            if tactic:
                rows.append(tactic)

    if not rows:
        return pd.DataFrame(columns=["tactic", "count", "color"])

    series = pd.Series(rows).value_counts().reset_index()
    series.columns = ["tactic", "count"]
    series["color"] = series["tactic"].map(lambda t: TACTIC_COLORS.get(t, "#888888"))
    return series


def technique_summary(mapped_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ids_str in mapped_df["attack_ids"].dropna():
        for tid in ids_str.split(", "):
            tid = tid.strip()
            if tid and tid in TECHNIQUES:
                rows.append({
                    "id":     tid,
                    "name":   TECHNIQUES[tid].name,
                    "tactic": TECHNIQUES[tid].tactic,
                    "url":    TECHNIQUES[tid].url,
                    "color":  TECHNIQUES[tid].color,
                })

    if not rows:
        return pd.DataFrame(columns=["id", "name", "tactic", "count", "url", "color"])

    df = pd.DataFrame(rows)
    counts = df.groupby(["id", "name", "tactic", "url", "color"]).size().reset_index(name="count")
    return counts.sort_values("count", ascending=False).reset_index(drop=True)
