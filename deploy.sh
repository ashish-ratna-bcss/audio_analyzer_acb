#!/bin/bash
set -e

# Run on the server after git pull

echo "=== Speech Intelligence API Deployment ==="

# 1. Python environment
echo "[1/3] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Check HF token
echo "[2/3] Checking HuggingFace token..."
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Export it before running:"
    echo "  export HF_TOKEN=your_token_here"
    echo "  Get token at: https://huggingface.co/settings/tokens"
    echo "  Accept pyannote/speaker-diarization-3.1 license at huggingface.co/pyannote/speaker-diarization-3.1"
    exit 1
fi

# 3. Start API
echo "[3/3] Starting API on port 8002..."
uvicorn app:app --host 0.0.0.0 --port 8002 --workers 1

# Smoke test (run separately after server starts)
# curl http://localhost:8002/health
# curl -X POST http://localhost:8002/stt/transcribe \
#   -F "audio=@sample.wav" -F "language=te" -F "diarize=true"
