#!/bin/bash
set -e

# Run on the server after git pull

echo "=== Speech Intelligence API Deployment ==="

# 1. Nginx setup
echo "[1/5] Configuring Nginx..."
sudo cp nginx/speech-api.conf /etc/nginx/sites-available/speech-api
sudo ln -sf /etc/nginx/sites-available/speech-api /etc/nginx/sites-enabled/speech-api
sudo nginx -t
sudo systemctl reload nginx

# 2. Python environment
echo "[2/5] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Check HF token
echo "[3/5] Checking HuggingFace token..."
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Export it before running:"
    echo "  export HF_TOKEN=your_token_here"
    echo "  Get token at: https://huggingface.co/settings/tokens"
    echo "  Accept pyannote/speaker-diarization-3.1 license at huggingface.co/pyannote/speaker-diarization-3.1"
    exit 1
fi

# 4. Start API
echo "[4/5] Starting API..."
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

# 5. Smoke test (run separately after server starts)
# curl http://localhost:8000/health
# curl -X POST http://localhost:8000/stt/transcribe \
#   -F "audio=@sample.wav" -F "language=te" -F "diarize=true"
