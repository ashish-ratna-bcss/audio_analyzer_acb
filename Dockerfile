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

# NeMo / Sortformer is an OPT-IN pilot layer only. nemo_toolkit pins older
# transformers/numpy that conflict with the base ASR stack, so it is NOT built
# into the production image by default. Enable explicitly to evaluate Sortformer:
#   docker build --build-arg INSTALL_NEMO=true ...
ARG INSTALL_NEMO=false
COPY requirements_nemo.txt .
RUN if [ "$INSTALL_NEMO" = "true" ]; then \
        pip3 install --no-cache-dir "Cython>=3.0" "setuptools<76" wheel && \
        pip3 install --no-cache-dir -r requirements_nemo.txt ; \
    else \
        echo "Skipping NeMo (Sortformer pilot) — set INSTALL_NEMO=true to include." ; \
    fi

COPY . .

RUN chmod +x /app/docker-entrypoint-api.sh

EXPOSE 8009

CMD ["/bin/bash", "/app/docker-entrypoint-api.sh"]
