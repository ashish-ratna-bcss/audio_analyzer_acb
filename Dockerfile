# CUDA + cuDNN runtime so the image works on NVIDIA G-series instances.
# On a CPU-only instance the same image still runs (torch falls back to CPU,
# config auto-selects device=cpu / compute_type=int8) -- just slower.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8009

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8009", "--workers", "1"]
