# =============================================================================
# utils/abuseipdb.py — AbuseIPDB API v2 integration for SST-Net
# Free tier: 1000 checks/day
# Docs: https://docs.abuseipdb.com
# =============================================================================

import time
import requests
import streamlit as st

ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"
REQUEST_DELAY = 1.5   # seconds between requests (free tier is generous)
MAX_IPS       = 10


def lookup_ip(ip: str, api_key: str) -> dict:
    """Look up a single IP on AbuseIPDB.

    Returns
    -------
    dict with keys:
        ip, abuse_confidence, total_reports, last_reported,
        country, isp, usage_type, domain, is_tor,
        abuseipdb_link, error
    """
    result = {
        "ip":               ip,
        "abuse_confidence": 0,
        "total_reports":    0,
        "last_reported":    "Never",
        "country":          "Unknown",
        "isp":              "Unknown",
        "usage_type":       "Unknown",
        "domain":           "Unknown",
        "is_tor":           False,
        "abuseipdb_link":   f"https://www.abuseipdb.com/check/{ip}",
        "error":            None,
    }

    try:
        response = requests.get(
            ABUSEIPDB_URL,
            headers={
                "Key":    api_key,
                "Accept": "application/json",
            },
            params={
                "ipAddress":    ip,
                "maxAgeInDays": 90,
                "verbose":      True,
            },
            timeout=10,
        )

        if response.status_code == 401:
            result["error"] = "Invalid API key"
            return result
        if response.status_code == 429:
            result["error"] = "Rate limit exceeded"
            return result
        if response.status_code == 422:
            result["error"] = f"Invalid IP address: {ip}"
            return result
        if response.status_code != 200:
            result["error"] = f"API error {response.status_code}"
            return result

        data = response.json().get("data", {})
        result["abuse_confidence"] = data.get("abuseConfidenceScore", 0)
        result["total_reports"]    = data.get("totalReports",          0)
        result["country"]          = data.get("countryCode",           "Unknown")
        result["isp"]              = data.get("isp",                   "Unknown")
        result["usage_type"]       = data.get("usageType",             "Unknown")
        result["domain"]           = data.get("domain",                "Unknown")
        result["is_tor"]           = data.get("isTor",                 False)

        lr = data.get("lastReportedAt")
        if lr:
            result["last_reported"] = lr[:10]   # YYYY-MM-DD

    except requests.exceptions.ConnectionError:
        result["error"] = "No internet connection"
    except requests.exceptions.Timeout:
        result["error"] = "Request timed out"
    except Exception as e:
        result["error"] = str(e)

    return result


def lookup_ips_batch(
    ips: list,
    api_key: str,
    max_ips: int = MAX_IPS,
    progress_callback=None,
) -> list:
    """Look up multiple IPs with rate limiting."""
    unique_ips = list(dict.fromkeys(str(ip) for ip in ips if ip and str(ip) != "nan"))
    unique_ips = unique_ips[:max_ips]
    results    = []

    for i, ip in enumerate(unique_ips):
        if progress_callback:
            progress_callback(i, len(unique_ips), ip)
        results.append(lookup_ip(ip, api_key))
        if i < len(unique_ips) - 1:
            time.sleep(REQUEST_DELAY)

    return results


def get_verdict(result: dict) -> tuple:
    """Return (verdict, color) based on abuse confidence score."""
    if result.get("error"):
        return "UNKNOWN", "#888888"

    score = result["abuse_confidence"]
    if score >= 75:
        return "MALICIOUS",  "#FF0033"
    elif score >= 25:
        return "SUSPICIOUS", "#FFA500"
    elif score >= 1:
        return "LOW RISK",   "#FFDD00"
    else:
        return "CLEAN",      "#00FF99"


def render_abuseipdb_results(results: list):
    """Render AbuseIPDB results as table + expandable cards."""
    if not results:
        st.info("No AbuseIPDB results to display.")
        return

    import pandas as pd

    # ── Summary metrics ───────────────────────────────────────────────────────
    total      = len(results)
    malicious  = sum(1 for r in results if get_verdict(r)[0] == "MALICIOUS")
    suspicious = sum(1 for r in results if get_verdict(r)[0] == "SUSPICIOUS")
    clean      = sum(1 for r in results if get_verdict(r)[0] == "CLEAN")
    avg_score  = sum(r["abuse_confidence"] for r in results) / total if total else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("IPs Checked",      total)
    c2.metric("🔴 Malicious",     malicious)
    c3.metric("🟠 Suspicious",    suspicious)
    c4.metric("🟢 Clean",         clean)
    c5.metric("Avg Abuse Score",  f"{avg_score:.1f}%")

    st.divider()

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    for r in results:
        verdict, _ = get_verdict(r)
        rows.append({
            "IP":             r["ip"],
            "Verdict":        verdict,
            "Abuse Score %":  r["abuse_confidence"],
            "Total Reports":  r["total_reports"],
            "Last Reported":  r["last_reported"],
            "Country":        r["country"],
            "ISP":            r["isp"],
            "Usage Type":     r["usage_type"],
            "Domain":         r["domain"],
            "TOR Node":       "Yes" if r["is_tor"] else "No",
            "AbuseIPDB Link": r["abuseipdb_link"],
            "Error":          r.get("error") or "—",
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Abuse Score %": st.column_config.ProgressColumn(
                "Abuse Score %", min_value=0, max_value=100, format="%d%%"
            ),
            "AbuseIPDB Link": st.column_config.LinkColumn(
                "AbuseIPDB Link", display_text="View →"
            ),
        },
    )

    st.divider()

    # ── Per-IP expandable cards ───────────────────────────────────────────────
    for r in results:
        verdict, color = get_verdict(r)
        tor_flag = " 🧅 TOR" if r["is_tor"] else ""

        with st.expander(
            f"**{r['ip']}** — {verdict}{tor_flag} "
            f"| Abuse: {r['abuse_confidence']}% "
            f"| Reports: {r['total_reports']} "
            f"| {r['country']}",
            expanded=(verdict in ["MALICIOUS", "SUSPICIOUS"]),
        ):
            if r.get("error"):
                st.warning(f"⚠️ Lookup error: {r['error']}")
                continue

            # Abuse score bar
            score = r["abuse_confidence"]
            bar_color = color
            st.markdown(
                f"""<div style="background:rgba(255,255,255,0.04);
                    border-left:4px solid {bar_color};border-radius:8px;
                    padding:12px 16px;margin-bottom:12px;">
                    <span style="color:{bar_color};font-size:18px;font-weight:700;">
                        {verdict}
                    </span>
                    <span style="color:#aaa;font-size:13px;margin-left:12px;">
                        Abuse Confidence Score: {score}%
                    </span>
                    <div style="margin-top:8px;background:#222;border-radius:4px;height:8px;">
                        <div style="width:{score}%;background:{bar_color};
                            border-radius:4px;height:8px;"></div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Country:** {r['country']}")
                st.markdown(f"**ISP:** {r['isp']}")
                st.markdown(f"**Domain:** {r['domain']}")
                st.markdown(f"**Usage Type:** {r['usage_type']}")
            with col2:
                st.markdown(f"**Total Reports:** {r['total_reports']}")
                st.markdown(f"**Last Reported:** {r['last_reported']}")
                st.markdown(f"**TOR Exit Node:** {'🧅 Yes' if r['is_tor'] else 'No'}")

            st.markdown(f"[🔗 View full report on AbuseIPDB]({r['abuseipdb_link']})")
