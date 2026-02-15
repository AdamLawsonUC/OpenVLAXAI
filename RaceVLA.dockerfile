# ===== base image with CUDA + build tools for flash-attn =====
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# 1) system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip \
      git git-lfs \
      build-essential \
      ffmpeg libsm6 libxext6 \
      ca-certificates \
      curl && \
    rm -rf /var/lib/apt/lists/*

# make "python" available
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN git lfs install

# 2) python setup
RUN python3 -m pip install --upgrade pip

# 3) install a CUDA build of torch
RUN pip install \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 4) clone openvla (we'll still use their deploy.py server)
RUN git clone https://github.com/openvla/openvla.git
WORKDIR /workspace/openvla

# 5) minimal deps from the repo
RUN pip install -r requirements-min.txt

# 6) install the repo itself (editable) so scripts work
RUN pip install -e .

# Robust: always load processor from base OpenVLA repo
RUN sed -i \
  "s|AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)|AutoProcessor.from_pretrained('openvla/openvla-7b', trust_remote_code=True)|" \
  /workspace/openvla/vla-scripts/deploy.py


# Upgrade HF stack (OpenVLA / remote processors need newer than 4.22)
RUN pip install -U \
    "transformers>=4.40.0" \
    "huggingface_hub>=0.23.0" \
    "accelerate>=0.28.0" \
    "safetensors>=0.4.2" \
    "tokenizers>=0.15.2"


# 7) flash-attn (OpenVLA recommends 2.5.5)
RUN pip install packaging ninja && \
    pip install "flash-attn==2.5.5" --no-build-isolation

# 7b) server deps that deploy.py needs
RUN pip install fastapi uvicorn json-numpy draccus huggingface_hub

# 8) download RaceVLA checkpoint from HF into the image
# NOTE: This is a *big* download (~15 GB repo; ~15 GB folder). The folder we need contains the sharded safetensors + config.
# Model folder name (as published):
# drone3_cycle/finetuned_model/openvla-7b+drone_set3+b16+lr-0.0005+lora-r32+dropout-0.0 :contentReference[oaicite:1]{index=1}
ARG RACEVLA_REPO=SerValera/RaceVLA_models
ARG RACEVLA_SUBDIR=drone3_cycle/finetuned_model/openvla-7b+drone_set3+b16+lr-0.0005+lora-r32+dropout-0.0
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download( \
    repo_id='${RACEVLA_REPO}', \
    allow_patterns=['${RACEVLA_SUBDIR}/*'], \
    local_dir='/workspace/racevla', \
    local_dir_use_symlinks=False \
)"

# 9) expose the REST server
EXPOSE 8000

# 10) serve RaceVLA weights through OpenVLA deploy.py
# Point --openvla_path to the *local folder* we downloaded above.
CMD ["python", "vla-scripts/deploy.py", \
     "--openvla_path", "/workspace/racevla/drone3_cycle/finetuned_model/openvla-7b+drone_set3+b16+lr-0.0005+lora-r32+dropout-0.0", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
