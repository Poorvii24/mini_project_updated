#!/bin/bash
# =============================================================================
# start.sh — Start both FastAPI backend and Streamlit dashboard
# =============================================================================

echo "🚀 Starting SST-Net API on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "🚀 Starting SST-Net Dashboard on port 8501..."
streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true &
DASH_PID=$!

echo "✅ Both services running."
echo "   Dashboard → http://localhost:8501"
echo "   API       → http://localhost:8000"
echo "   API Docs  → http://localhost:8000/docs"

# Wait for either process to exit
wait $API_PID $DASH_PID
