FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

WORKDIR /home/marvin/alan/openvla_finetuner

# Install system dependencies for flash-attn
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install openvla, lerobot, and flash-attn; download openvla-7b
COPY . /workspaces/openvla_finetuner
WORKDIR /workspaces/openvla_finetuner
RUN pip install -e . && \
    cd lerobot && \
    pip install -e . && \
    cd .. && \
    pip check | awk '$1 ~ /openvla/ {gsub(/,/,"",$5); print $5}' | xargs pip install && \
    pip install packaging ninja && \
    pip install "flash-attn==2.5.5" --no-build-isolation && \
    pip install huggingface-hub && \
    huggingface-cli download openvla/openvla-7b
