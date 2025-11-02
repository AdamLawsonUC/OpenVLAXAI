# ===== base image with CUDA + build tools for flash-attn =====
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

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

RUN git lfs install

# 2) python setup
RUN python3 -m pip install --upgrade pip

# 3) install a CUDA build of torch that matches what OpenVLA expects
#    OpenVLA README says: Python 3.10, PyTorch 2.2.*. :contentReference[oaicite:5]{index=5}
#    cu121 wheels work fine on A100.
RUN pip install \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 4) clone openvla
RUN git clone https://github.com/openvla/openvla.git
WORKDIR /workspace/openvla

# 5) minimal deps from the repo (transformers, timm, tokenizers, ...)
#    README tells us to "pip install -r requirements-min.txt" first. :contentReference[oaicite:6]{index=6}
RUN pip install -r requirements-min.txt

# 6) install the repo itself (editable) so scripts work
RUN pip install -e .

# 7) flash-attn (OpenVLA explicitly recommends 2.5.5) :contentReference[oaicite:7]{index=7}
RUN pip install packaging ninja && \
    pip install "flash-attn==2.5.5" --no-build-isolation

# 8) expose the REST server
#    Their README says they provide a "lightweight script for serving OpenVLA over a REST API"
#    via `vla-scripts/deploy.py`. :contentReference[oaicite:8]{index=8}
EXPOSE 8000

# default: serve the 7B model on 0.0.0.0:8000
CMD ["python", "vla-scripts/deploy.py", \
     "--model", "openvla/openvla-7b", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
