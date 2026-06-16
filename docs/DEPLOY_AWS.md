# Deploy on AWS (Docker, GPU, auto-start)

Runs the API as a Docker service on an AWS GPU instance. The container:
- exposes the endpoint on port 8009,
- requires an `X-API-Key` header (shared with your app),
- runs in the background and **survives SSH logout**,
- **auto-starts when the instance reboots / is stopped and started again**
  (Docker daemon starts on boot + `restart: unless-stopped`).

The same image also runs on a CPU instance (slower) — see notes at the end.

---

## A. Endpoint (reference for app integration)

```bash
curl -s -X POST http://<INSTANCE_PUBLIC_IP>:8009/stt/transcribe \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -F "audio=@/path/to/call.mp4"
```

Response:
```json
{
  "language": "te",
  "duration": 58.0,
  "raw":     { "dialogue": [ {"start":1.46,"end":9.67,"speaker":"Speaker_1","text":"<original language>"} ] },
  "english": { "dialogue": [ {"start":1.46,"end":9.67,"speaker":"Speaker_1","text":"<english>"} ] }
}
```
- `raw` = original-language transcript (code-switch preserved: te+en, hi+en, …)
- `english` = full English translation
- omit `language` to auto-detect; add `-F debug=true` for confidence/metrics;
  `-F diarize=false` for a single speaker.
- Health check (no key): `curl http://<IP>:8009/health`

In your app, send the same key from `.env`:
```ts
fetch("http://<IP>:8009/stt/transcribe", {
  method: "POST",
  headers: { "X-API-Key": process.env.STT_API_KEY! },
  body: formData,   // field name: audio
});
```

---

## B. AWS Console (GUI) — step by step

### 1. Launch the instance
1. EC2 → **Launch instance**.
2. Name: `acb-audio-analyzer`.
3. AMI: **Deep Learning Base GPU AMI (Ubuntu 22.04)** — search "Deep Learning Base GPU" in AMIs. (It ships NVIDIA drivers + Docker + nvidia-container-toolkit, saving setup. A plain Ubuntu 22.04 AMI also works but you install those yourself — see step 4b.)
4. Instance type: **g4dn.xlarge** (or g5.xlarge). Any G-series works.
5. Key pair: select/create one (for SSH).
6. Storage: **100+ GB** gp3 (model weights + CUDA image are large).
7. Network / **Security group** — create one:
   - SSH: TCP **22**, source = **My IP**.
   - Custom TCP: port **8009**, source = **0.0.0.0/0** (public access from
     anywhere, as required for the app).
     WARNING: this is plain HTTP — the API key and uploaded audio travel
     UNENCRYPTED. The API key is the only thing stopping abuse. Use a long
     random key, rotate it, and move to HTTPS (Caddy/nginx + domain) before
     handling real evidence at scale. Keeping SSH (22) restricted to My IP.
8. **Launch instance**.

### 2. Allocate a stable IP (so it survives stop/start)
- EC2 → **Elastic IPs** → **Allocate** → **Associate** with the instance.
- Without this, the public IP changes every stop/start (you'd edit your app each time).

### 3. Connect
- EC2 → instance → **Connect** → SSH, e.g.
  `ssh -i key.pem ubuntu@<ELASTIC_IP>`

### 4. One-time setup on the instance
```bash
# 4a. Verify Docker + GPU (Deep Learning AMI already has these)
docker --version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi   # should list the GPU

# 4b. ONLY if using a plain Ubuntu AMI (skip on Deep Learning AMI):
#   sudo apt-get update && sudo apt-get install -y docker.io
#   sudo systemctl enable --now docker
#   # install nvidia-container-toolkit per NVIDIA docs, then:
#   sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
#   sudo usermod -aG docker $USER && newgrp docker

# 4c. Make sure Docker starts on boot (required for auto-start)
sudo systemctl enable docker

# 4d. Get the code + config
git clone https://github.com/ashish-ratna-bcss/audio_analyzer_acb.git
cd audio_analyzer_acb
cp .env.example .env
nano .env        # set API_KEY (openssl rand -hex 32) and HF_TOKEN
```

### 5. Build + run (background, auto-restart)
```bash
docker compose up -d --build
```
- `-d` = detached (survives logout). `restart: unless-stopped` in the compose
  file = auto-start on reboot.
- First boot downloads the models (large-v3 + pyannote) into named volumes;
  this takes a few minutes once, then they persist.

### 6. Verify
```bash
docker compose logs -f          # watch startup; Ctrl-C to stop watching
curl http://localhost:8009/health
# from your machine:
curl -s -X POST http://<ELASTIC_IP>:8009/stt/transcribe \
  -H "X-API-Key: <YOUR_API_KEY>" -F "audio=@sample.mp4"
```

---

## C. Stop / resume for cost saving

- **Stop** (EC2 → Instance state → Stop): compute billing pauses; the EBS disk,
  Docker image, and model volumes persist.
- **Start** again: Docker starts on boot and `restart: unless-stopped` brings
  the container back up **automatically** — no manual `docker` command. Models
  are already cached in the volumes, so it's a warm start.
- Keep the **Elastic IP** associated so the endpoint URL is unchanged.

Confirm after a stop/start:
```bash
docker ps          # container 'acb-audio-analyzer' running
```

Useful ops:
```bash
docker compose restart        # restart service
docker compose down           # stop + remove container (volumes kept)
docker compose up -d --build  # rebuild after a git pull
git pull && docker compose up -d --build
```

---

## D. CPU-only instance (optional, slower)

The image runs without a GPU too:
1. Edit `docker-compose.yml` and delete the entire `deploy:` block.
2. `docker compose up -d --build`.
3. config auto-selects `device=cpu`, `compute_type=int8`. Expect minutes per
   file with large-v3 — fine for testing, not for throughput.

---

## E. Security notes

This deployment is **public HTTP + API key** (chosen for the demo). Implications:
- The API key and the uploaded audio/video travel **unencrypted**. Anyone on
  the network path can read them. Acceptable for a demo, NOT for sustained
  evidence handling.
- The API key is the only access control. Make it long and random
  (`openssl rand -hex 32`) and rotate it: edit `.env` then `docker compose up -d`.
- Uploads accept audio and common video (mp4/mov/mkv/avi/webm/…); ffmpeg
  extracts the audio. Max size `MAX_FILE_SIZE_MB` (default 500).
- `/health` is unauthenticated for health checks.

**Upgrade path to HTTPS (do this before real public use):** point a domain at
the Elastic IP and run Caddy as a reverse proxy in front of port 8009 — Caddy
auto-provisions a Let's Encrypt cert. The app then calls `https://api.your
domain.com/stt/transcribe`. Ask and this can be added as a compose service.
