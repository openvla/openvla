# Standalone container

Build from the OpenVLA-QuIC repository root:

```bash
docker build -f deploy/Dockerfile -t openvla-quic:local .
```

This image supports dense-host inference, prepared-NPZ adapter training, and
HTTP serving. It has no OVLAB, LIBERO, or simulator dependency. To include
optional TFDS/OXE preparation and the upstream LoRA tuner, build with
`--build-arg EXTRAS=serve,train`. FlashAttention is optional; the examples use
Torch SDPA. Model weights are mounted at runtime, never copied into the image.

```bash
docker run --rm --gpus all --read-only --tmpfs /tmp:rw,size=2g \
  -p 127.0.0.1:8000:8000 \
  --mount type=bind,source=/absolute/checkpoint,target=/models/openvla-7b,readonly \
  --mount type=bind,source=/absolute/adapter,target=/adapter,readonly \
  --mount type=bind,source=/absolute/config.json,target=/config.json,readonly \
  openvla-quic:local serve --config /config.json --adapter /adapter --host 0.0.0.0
```

The JSON host paths must refer to the mounted container paths. The process runs
as UID/GID 10001; mounted data must be readable and training outputs writable
by that user. Use a new child directory of a writable output mount for training:

```bash
docker run --rm --gpus all --read-only --tmpfs /tmp:rw,size=2g \
  --mount type=bind,source=/absolute/checkpoint,target=/models/openvla-7b,readonly \
  --mount type=bind,source=/absolute/prepared,target=/datasets/prepared,readonly \
  --mount type=bind,source=/absolute/config.json,target=/config.json,readonly \
  --mount type=bind,source=/absolute/writable-output,target=/outputs \
  openvla-quic:local train --config /config.json --output /outputs/run-1 \
  --fit-action-normalization
```

`GET /health` reports readiness after the model has loaded. `POST /act` accepts
`{"image": [[[0, 0, 0]]], "instruction": "pick up the object"}` with a real RGB
array and returns `{"action": [...]}`. The image resolution is handled by the
host processor. `scripts/quic_client.py` sends an image file using this contract.
