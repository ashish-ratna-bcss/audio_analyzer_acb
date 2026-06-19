# CUDA + cuDNN runtime so the image works on NVIDIA G-series instances.
# On a CPU-only instance the same image still runs (torch falls back to CPU,
# config auto-selects device=cpu / compute_type=int8) -- just slower.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev build-essential ffmpeg bash git \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# NeMo is intentionally NOT installed here. It conflicts with this image's
# torch/torchaudio (proven: breaks libtorchaudio). Sortformer runs in the
# separate `sortformer` sidecar (Dockerfile.sortformer) with its own torch.

COPY . .

RUN chmod +x /app/docker-entrypoint-api.sh

EXPOSE 8009

CMD ["/bin/bash", "/app/docker-entrypoint-api.sh"]
