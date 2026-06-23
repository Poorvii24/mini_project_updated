# =============================================================================
# Dockerfile — SST-Net Botnet Detection Dashboard
# =============================================================================

# Python 3.11 slim keeps the base image small
FROM python:3.11-slim

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="Poorvi Prahlad Purohit"
LABEL project="SST-Net"
LABEL version="1.0.0"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker caches this layer
# (only re-runs if requirements.txt changes)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY app.py           .
COPY utils/           ./utils/
COPY models/          ./models/
COPY transformer_classifier.pt .

# ── Streamlit config ──────────────────────────────────────────────────────────
# Disable telemetry and set server options
RUN mkdir -p /app/.streamlit
COPY .streamlit/config.toml .streamlit/config.toml

# ── Create non-root user for security ────────────────────────────────────────
RUN useradd -m -u 1000 sst_net && \
    chown -R sst_net:sst_net /app
USER sst_net

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Start command ─────────────────────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
