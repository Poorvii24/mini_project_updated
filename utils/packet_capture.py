# =============================================================================
# utils/packet_capture.py
# Real-time packet capture (Scapy) + CSV replay simulation for SST-Net.
#
# Two modes:
#   1. LIVE   — captures real packets from a network interface (needs admin
#               privileges + Npcap on Windows)
#   2. REPLAY — replays a CTU-13 CSV file as simulated live traffic
#               (no admin rights needed, works everywhere)
#
# Both modes feed flows into the same queue consumed by the live dashboard.
# =============================================================================

import time
import threading
import queue
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

# ── Scapy import is optional — only required for LIVE mode ───────────────────
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, get_if_list
    SCAPY_AVAILABLE = True
except (ImportError, OSError):
    SCAPY_AVAILABLE = False


# =============================================================================
# Shared flow queue — both capture modes push flows here
# =============================================================================
class FlowBuffer:
    """Thread-safe buffer holding captured/simulated flows for the dashboard."""

    def __init__(self, maxsize: int = 5000):
        self._queue   = queue.Queue(maxsize=maxsize)
        self._running = False
        self._thread  = None
        self._stats   = {"total_captured": 0, "start_time": None, "mode": None}

    def push(self, flow: dict):
        try:
            self._queue.put_nowait(flow)
            self._stats["total_captured"] += 1
        except queue.Full:
            try:
                self._queue.get_nowait()   # drop oldest
                self._queue.put_nowait(flow)
            except queue.Empty:
                pass

    def drain(self, max_items: int = 1000) -> list:
        """Pull all currently buffered flows without blocking."""
        items = []
        for _ in range(max_items):
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items

    def qsize(self) -> int:
        return self._queue.qsize()

    def is_running(self) -> bool:
        return self._running

    def get_stats(self) -> dict:
        stats = dict(self._stats)
        if stats["start_time"]:
            stats["elapsed_sec"] = round(time.time() - stats["start_time"], 1)
            stats["rate_per_sec"] = round(
                stats["total_captured"] / max(stats["elapsed_sec"], 1), 2
            )
        else:
            stats["elapsed_sec"]  = 0
            stats["rate_per_sec"] = 0
        return stats

    def stop(self):
        self._running = False

    def reset_stats(self):
        self._stats = {"total_captured": 0, "start_time": time.time(), "mode": None}


# Global singleton buffer used across the app
_flow_buffer = FlowBuffer()


def get_flow_buffer() -> FlowBuffer:
    return _flow_buffer


# =============================================================================
# MODE 1 — LIVE CAPTURE (Scapy)
# =============================================================================
def list_network_interfaces() -> list:
    """Return available network interfaces for live capture."""
    if not SCAPY_AVAILABLE:
        return []
    try:
        return get_if_list()
    except Exception:
        return []


class LiveFlowTracker:
    """Aggregates raw packets into flow records (mirrors CTU-13 schema)."""

    def __init__(self, flow_timeout: int = 30):
        self.flows        = defaultdict(lambda: {
            "start_time": None, "last_time": None,
            "pkt_count": 0, "byte_count": 0, "src_bytes": 0,
        })
        self.flow_timeout = flow_timeout

    def _flow_key(self, pkt) -> tuple:
        if IP not in pkt:
            return None
        proto = "tcp" if TCP in pkt else ("udp" if UDP in pkt else "icmp")
        sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
        dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
        return (pkt[IP].src, pkt[IP].dst, sport, dport, proto)

    def process_packet(self, pkt) -> dict | None:
        """Process one packet, return a completed flow dict if flow should emit."""
        if IP not in pkt:
            return None

        key  = self._flow_key(pkt)
        if key is None:
            return None

        now  = time.time()
        flow = self.flows[key]

        if flow["start_time"] is None:
            flow["start_time"] = now
        flow["last_time"]   = now
        flow["pkt_count"]  += 1
        flow["byte_count"] += len(pkt)
        if pkt[IP].src == key[0]:
            flow["src_bytes"] += len(pkt)

        # Emit a flow record on every packet (simplified — real NetFlow
        # aggregates over time windows, but per-packet emission works
        # fine for dashboard demo purposes)
        src, dst, sport, dport, proto = key
        duration = max(flow["last_time"] - flow["start_time"], 1e-6)

        return {
            "StartTime": datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f"),
            "Dur":       round(duration, 6),
            "Proto":     proto,
            "SrcAddr":   src,
            "Sport":     sport,
            "Dir":       "->",
            "DstAddr":   dst,
            "Dport":     dport,
            "State":     "CON",
            "sTos":      0,
            "dTos":      0,
            "TotPkts":   flow["pkt_count"],
            "TotBytes":  flow["byte_count"],
            "SrcBytes":  flow["src_bytes"],
            "Label":     "flow=Background",   # unknown — model will classify
        }


def start_live_capture(interface: str, buffer: FlowBuffer, duration: int = 60):
    """Start live packet capture in a background thread.

    Parameters
    ----------
    interface : network interface name (from list_network_interfaces())
    buffer    : FlowBuffer to push captured flows into
    duration  : max capture duration in seconds (safety limit)
    """
    if not SCAPY_AVAILABLE:
        raise RuntimeError(
            "Scapy is not available. Install with: pip install scapy\n"
            "On Windows, also install Npcap from https://npcap.com/#download"
        )

    tracker = LiveFlowTracker()
    buffer._running = True
    buffer.reset_stats()
    buffer._stats["mode"] = "LIVE"

    def _packet_callback(pkt):
        if not buffer.is_running():
            return True   # stop sniffing
        flow = tracker.process_packet(pkt)
        if flow:
            buffer.push(flow)
        return False

    def _capture_thread():
        try:
            sniff(
                iface=interface,
                prn=_packet_callback,
                store=False,
                timeout=duration,
                stop_filter=lambda p: not buffer.is_running(),
            )
        except Exception as e:
            buffer._stats["error"] = str(e)
        finally:
            buffer._running = False

    thread = threading.Thread(target=_capture_thread, daemon=True)
    thread.start()
    buffer._thread = thread
    return thread


# =============================================================================
# MODE 2 — CSV REPLAY SIMULATION
# =============================================================================
def start_csv_replay(
    csv_path_or_df,
    buffer: FlowBuffer,
    flows_per_sec: float = 5.0,
    max_flows: int = 2000,
):
    """Replay a CTU-13 CSV file as simulated live traffic.

    Parameters
    ----------
    csv_path_or_df : path to CSV file, or an already-loaded DataFrame
    buffer         : FlowBuffer to push flows into
    flows_per_sec  : simulated traffic rate
    max_flows      : max rows to replay (avoid infinite loop on large files)
    """
    if isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df
    else:
        df = pd.read_csv(csv_path_or_df, on_bad_lines="skip")

    df = df.head(max_flows)

    buffer._running = True
    buffer.reset_stats()
    buffer._stats["mode"] = "REPLAY"

    delay = 1.0 / max(flows_per_sec, 0.1)

    def _replay_thread():
        for _, row in df.iterrows():
            if not buffer.is_running():
                break
            flow = row.to_dict()
            buffer.push(flow)
            time.sleep(delay)
        buffer._running = False

    thread = threading.Thread(target=_replay_thread, daemon=True)
    thread.start()
    buffer._thread = thread
    return thread


def stop_capture(buffer: FlowBuffer):
    """Stop any running capture/replay session."""
    buffer.stop()
