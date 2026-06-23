# =============================================================================
# utils/virustotal.py — VirusTotal API v3 integration for SST-Net
# =============================================================================

import time
import requests
import streamlit as st


VT_BASE_URL   = "https://www.virustotal.com/api/v3"
REQUEST_DELAY = 15.5
MAX_IPS       = 10


def lookup_ip(ip: str, api_key: str) -> dict:
    result = {
        "ip": ip, "malicious": 0, "suspicious": 0, "harmless": 0,
        "undetected": 0, "total_engines": 0, "reputation": 0,
        "tags": [], "country": "Unknown", "asn": "Unknown", "isp": "Unknown",
        "last_analysis_date": "Unknown",
        "vt_link": f"https://www.virustotal.com/gui/ip-address/{ip}",
        "error": None,
    }
    try:
        response = requests.get(
            f"{VT_BASE_URL}/ip_addresses/{ip}",
            headers={"x-apikey": api_key}, timeout=10)
        if response.status_code == 401:
            result["error"] = "Invalid API key"; return result
        if response.status_code == 429:
            result["error"] = "Rate limit exceeded"; return result
        if response.status_code == 404:
            result["error"] = "IP not found in VT database"; return result
        if response.status_code != 200:
            result["error"] = f"API error {response.status_code}"; return result

        attrs = response.json().get("data", {}).get("attributes", {})
        stats = attrs.get("last_analysis_stats", {})
        result["malicious"]     = stats.get("malicious",  0)
        result["suspicious"]    = stats.get("suspicious", 0)
        result["harmless"]      = stats.get("harmless",   0)
        result["undetected"]    = stats.get("undetected", 0)
        result["total_engines"] = sum(stats.values())
        result["reputation"]    = attrs.get("reputation", 0)
        result["tags"]          = attrs.get("tags", [])
        result["country"]       = attrs.get("country", "Unknown")
        result["asn"]           = str(attrs.get("asn", "Unknown"))
        result["isp"]           = attrs.get("as_owner", "Unknown")
        ts = attrs.get("last_analysis_date")
        if ts:
            from datetime import datetime
            result["last_analysis_date"] = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    except requests.exceptions.ConnectionError:
        result["error"] = "No internet connection"
    except requests.exceptions.Timeout:
        result["error"] = "Request timed out"
    except Exception as e:
        result["error"] = str(e)
    return result


def lookup_ips_batch(ips, api_key, max_ips=MAX_IPS, progress_callback=None):
    unique_ips = list(dict.fromkeys(str(ip) for ip in ips if ip and str(ip) != "nan"))
    unique_ips = unique_ips[:max_ips]
    results = []
    for i, ip in enumerate(unique_ips):
        if progress_callback:
            progress_callback(i, len(unique_ips), ip)
        results.append(lookup_ip(ip, api_key))
        if i < len(unique_ips) - 1:
            time.sleep(REQUEST_DELAY)
    return results


def get_verdict(result: dict) -> tuple:
    if result.get("error"):
        return "UNKNOWN", "#888888"
    mal = result["malicious"]
    sus = result["suspicious"]
    rep = result["reputation"]
    if mal >= 5 or rep <= -50:
        return "MALICIOUS",  "#FF0033"
    elif mal >= 1 or sus >= 3 or rep <= -20:
        return "SUSPICIOUS", "#FFA500"
    elif mal == 0 and sus == 0 and rep >= 0:
        return "CLEAN",      "#00FF99"
    else:
        return "LOW RISK",   "#FFDD00"


def render_vt_results(results: list):
    """Render VirusTotal results — table + expandable cards, no plotly."""
    if not results:
        st.info("No VirusTotal results to display.")
        return

    import pandas as pd

    # Summary metrics
    total      = len(results)
    malicious  = sum(1 for r in results if get_verdict(r)[0] == "MALICIOUS")
    suspicious = sum(1 for r in results if get_verdict(r)[0] == "SUSPICIOUS")
    clean      = sum(1 for r in results if get_verdict(r)[0] == "CLEAN")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("IPs Checked",   total)
    c2.metric("🔴 Malicious",  malicious)
    c3.metric("🟠 Suspicious", suspicious)
    c4.metric("🟢 Clean",      clean)

    st.divider()

    # Summary table
    rows = []
    for r in results:
        verdict, _ = get_verdict(r)
        rows.append({
            "IP":            r["ip"],
            "Verdict":       verdict,
            "Malicious":     r["malicious"],
            "Suspicious":    r["suspicious"],
            "Total Engines": r["total_engines"],
            "Reputation":    r["reputation"],
            "Country":       r["country"],
            "ISP":           r["isp"],
            "Tags":          ", ".join(r["tags"]) if r["tags"] else "—",
            "Last Analysis": r["last_analysis_date"],
            "VT Link":       r["vt_link"],
            "Error":         r.get("error") or "—",
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "VT Link": st.column_config.LinkColumn("VT Link", display_text="View →")
        },
    )

    st.divider()

    # Per-IP expandable cards
    for r in results:
        verdict, color = get_verdict(r)
        with st.expander(
            f"**{r['ip']}** — {verdict} "
            f"| 🔴 {r['malicious']} malicious "
            f"| 🟠 {r['suspicious']} suspicious "
            f"| {r['country']}",
            expanded=(verdict in ["MALICIOUS", "SUSPICIOUS"]),
        ):
            if r.get("error"):
                st.warning(f"⚠️ Lookup error: {r['error']}")
                continue

            st.markdown(
                f"""<div style="background:rgba(255,255,255,0.04);
                    border-left:4px solid {color};border-radius:8px;
                    padding:12px 16px;margin-bottom:8px;">
                    <span style="color:{color};font-size:18px;font-weight:700;">
                        {verdict}
                    </span>
                    <span style="color:#aaa;font-size:13px;margin-left:12px;">
                        Reputation: {r['reputation']}
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Country:** {r['country']}")
                st.markdown(f"**ASN:** {r['asn']}")
                st.markdown(f"**ISP:** {r['isp']}")
                st.markdown(f"**Last Analysis:** {r['last_analysis_date']}")
                if r["tags"]:
                    st.markdown(f"**Tags:** `{'`, `'.join(r['tags'])}`")
            with col2:
                st.markdown(f"**Malicious:** {r['malicious']} / {r['total_engines']}")
                st.markdown(f"**Suspicious:** {r['suspicious']} / {r['total_engines']}")
                st.markdown(f"**Harmless:** {r['harmless']} / {r['total_engines']}")
                st.markdown(f"**Undetected:** {r['undetected']} / {r['total_engines']}")
            st.markdown(f"[🔗 View full report on VirusTotal]({r['vt_link']})")
