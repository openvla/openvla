FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Install system dependencies for flash-attn.
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone -b merge-finetuner-changes https://github.com/nomagiclab/openvla.git && \
    cd openvla && \
    git submodule init third_party/lerobot && \
    git submodule update --recursive --init third_party/lerobot 
    
WORKDIR /workspace/openvla

RUN ls -la third_party/lerobot 

# Editable install of openvla, then lerobot submodule, then reinstall newly
# missing openvla dependencies to negotiate dependency incompatibility.
# Then install flash-attn separately (per OpenVLA instructions)
# and download the openvla-7b model.
RUN pip install -e . && \
    cd third_party/lerobot/ && \
    pip install -e . && \
    cd ../../ && \
    pip check | awk '$1 ~ /openvla/ {gsub(/,/,"",$5); print $5}' | xargs pip install && \
    pip install packaging ninja && \
    pip install "flash-attn==2.5.5" --no-build-isolation && \
    pip install huggingface-hub && \
    huggingface-cli download openvla/openvla-7b
