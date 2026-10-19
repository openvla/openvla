FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime@sha256:e5c32997194edfc6fb0235223d1cd0433056045a468e7dfeaac390a3fcf8ae6d

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    HF_HOME=/tmp/huggingface XDG_CACHE_HOME=/tmp/cache
WORKDIR /opt/openvla-quic
ARG EXTRAS=serve
COPY requirements ./requirements
RUN set -eu; set -- -r requirements/core.txt; IFS=,; \
    for extra in ${EXTRAS}; do \
      case "$extra" in serve|rlds|train) set -- "$@" -r "requirements/$extra.txt" ;; \
      *) echo "Supported image extras: serve,rlds,train" >&2; exit 2 ;; esac; \
    done; unset IFS; \
    python -m pip install --no-cache-dir 'setuptools>=77' 'wheel>=0.45' "$@"
COPY . .
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation . && python -m pip check \
    && groupadd --gid 10001 openvla \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent openvla
USER 10001:10001
ENTRYPOINT ["openvla-quic"]
CMD ["--help"]
