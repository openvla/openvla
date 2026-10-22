# OpenVLA-QuIC

Standalone dense token OpenVLA with parameter-efficient QuIC adapters. This fork
can prepare data, tune adapters, run teacher-forced validation, predict actions,
and serve HTTP inference without OVLAB or a benchmark simulator.

The portable method is a separate dependency,
[quic-peft-vla](https://github.com/kon0327/quic-peft-vla), pinned to an immutable
Git commit in `pyproject.toml`. This project supplies the OpenVLA model integration
and standalone tools. The upstream OpenVLA documentation is retained in
[README.upstream.md](README.upstream.md).

## Installation

Use Python 3.10 or 3.11 and a suitable Torch/CUDA environment:

```bash
git clone https://github.com/kon0327/openvla-quic.git
cd openvla-quic
python -m pip install .
openvla-quic --help
```

The default installation supports inference and adapter training on prepared NPZ
episodes. It does not install TensorFlow, LIBERO, Robosuite, or MuJoCo. Install
extras for the workflows you use:

```bash
python -m pip install '.[serve]'  # FastAPI / Uvicorn
python -m pip install '.[rlds]'   # TFDS / OXE episode preparation and TF crop
python -m pip install '.[train]'  # RLDS plus the preserved upstream LoRA tuner
```

`requirements/core.txt`, `serve.txt`, `rlds.txt`, and `train.txt` describe the same
boundaries. The supported host uses Torch 2.2.0, Transformers 4.40.1, TIMM 0.9.10,
and NumPy 1.26. FlashAttention 2.5.5 is optional and must be built for the matching
Torch/CUDA toolchain; the standalone examples use SDPA. The dependency installer
fetches the separate method from its public Git repository. OVLAB containers may
instead preinstall their pinned local method checkout and install this fork with
`--no-deps`, as they do for other reviewed external packages.

## Model API

The adapter API is built into `OpenVLAForActionPrediction`:

```python
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from quic_petf_vla import AdapterConfig

model = OpenVLAForActionPrediction.from_pretrained(
    '/models/openvla-7b', local_files_only=True,
    attn_implementation='sdpa', torch_dtype='auto',
)
model.enable_quic({
    'projector.fc2': AdapterConfig(orders=(1, 2), block_count=8, base_size=31),
})
parameters = model.quic_parameters()  # pass to your own optimizer
# model.forward(...) and model.predict_action(...) use these adapters directly.
# Supply a verified immutable checkpoint identity when serializing:
# model.save_quic_adapter(directory, host_identity, metadata=metadata)
# model.load_quic_adapter(directory, host_identity)
```

### Optional continuous action chunks

The token model remains the default. Before `enable_quic`, enable a continuous
readout with:

```python
model.configure_action_head({"type": "mlp_chunk", "horizon": 8, "hidden_dim": 512})
# Inject your explicit active backbone targets, excluding language_model.lm_head.
model.enable_quic(targets)
optimizer = torch.optim.AdamW(model.adaptation_parameters(), lr=1e-4)
# forward(..., actions=[B,H,7], action_mask=[B,H]) -> masked L1 loss and .actions
# predict_action(...) -> decoded [H,7], including H=1
```

The small head reads the last valid prompt state using the decoder directly,
without vocabulary logits or autoregressive action generation. It is trained in
full in FP32 alongside QuIC. `quic_parameters()` continues to select adapters
only; `adaptation_parameters()` adds the head. The QuIC compound operators and
portable adapter format are unchanged.

For standalone commands put the same object in `host.action_head` of the JSON
configuration. `prepare` retains complete action trajectories even when images
are sampled with a frame stride. Training masks future steps beyond episode or
instruction boundaries. Validation reports normalized L1. Use the model's
`save_quic_adapter`/`load_quic_adapter` methods for the complete bundle: optional
head weights are stored in `action_head.safetensors`, bound by configuration,
host identity and SHA256 in the adapter manifest. `load_policy` reconstructs the
variant from that manifest. `/act` returns a nested `action` array for chunks.
Omit the option or set it to `null` to construct the original token model.

For a complete local checkpoint + processor + artifact runtime:

```python
from quic_petf import load_policy

policy = load_policy({
    'checkpoint': '/models/openvla-7b',
    'revision': 'YOUR_IMMUTABLE_40_CHARACTER_SOURCE_REVISION',
    'adaptation_state': 'none',
    'unnorm_key': 'my_robot',
    'device': 'cuda:0',
    'dtype': 'bfloat16',
    'attention': 'sdpa',
}, adapter_path='/adapters/my-run')
action = policy.predict(rgb_uint8_array, 'pick up the object')
```

An artifact does not include the dense base weights. Loading verifies checkpoint
bytes, model/method source, target dimensions, tensor content, preprocessing, and
normalization. Fitted action statistics are restored from artifact metadata.
An active adapter is exported with `save_quic_adapter`; `save_pretrained` rejects
it rather than writing an HF checkpoint that cannot reconstruct its wrappers.
Changing the model implementation changes its source identity; artifacts bound
to an older integration require that older source/image or a newly produced
artifact. Identity checks must not be bypassed by relabelling old metadata.

## Tuning, evaluation, and deployment

Start with [experiments/quic/projector.json](experiments/quic/projector.json).
Replace its checkpoint revision, paths, normalization key and explicit target
parameters. Full data formats and optional robot benchmark commands are in
[experiments/quic/README.md](experiments/quic/README.md).

```bash
openvla-quic inventory --config experiments/quic/projector.json
openvla-quic train --config experiments/quic/projector.json \
  --output outputs/my-run --fit-action-normalization
openvla-quic evaluate --config experiments/quic/projector.json \
  --adapter outputs/my-run/adapter --output outputs/my-eval
openvla-quic predict --config experiments/quic/projector.json \
  --adapter outputs/my-run/adapter --image observation.png \
  --instruction 'pick up the object'
openvla-quic serve --config experiments/quic/projector.json \
  --adapter outputs/my-run/adapter
```

Without `--fit-action-normalization`, training freezes host statistics. The flag
fits only training episodes; held-out episodes never contribute. Whole-episode
splits, frame hashes, masked action-token labels, identity initialization, frozen
base weights, finite adapter gradients, and exact save/reload logits are checked.
The current standalone tuner is single-process and adapter-only. It supports
neither quantized hosts nor active LoRA wrappers. Validation is teacher-forced;
rollout success belongs to the optional environment experiment.

The CLI draws progress bars and prints compact summaries. Full JSON remains in
the output directory; `--json` requests it on stdout. Historical script paths
now dispatch to these commands: `vla-scripts/finetune.py`, `vla-scripts/deploy.py`,
and `scripts/quic.py`. The original LoRA trainer and vanilla HTTP server remain
available as `finetune_lora.py` and `deploy_legacy.py`.

[deploy/README.md](deploy/README.md) describes the standalone Docker image and
mounts. `POST /act` accepts an RGB integer array plus instruction and returns a
JSON action vector. The model loads once, before `/health` reports readiness.

## Ownership and OVLAB compatibility

- `quic-peft-vla` (distribution `quic-petf-vla`): model-independent compound mathematics and adapter artifacts.
- `prismatic/extern/hf/modeling_prismatic.py`: dense OpenVLA and its adapter API.
- `quic_petf/host.py` and `runtime.py`: local checkpoint identity, loading,
  processor/preprocessing and saved normalization composition.
- `quic_petf/configuration.py`: explicit standalone run settings; no QP/A rules.
- OVLAB: experiment hierarchy, named profiles/placements, ablations, run registry,
  reports, simulator orchestration and its provider protocol.

`quic_petf` contains only the PETF QuIC runtime. It does not contain a WC
backend, hybrid-method dispatch, the legacy compound-PEFT bridge, or an OVLAB
provider. OVLAB's protocol adapter lives in `ovlab_openvla_quic.runtime` and
imports this package lazily. Named research selectors live in OVLAB's
`quic_experiments.py`.

The Python namespace is `quic_petf` (`python -m quic_petf`); the distribution and
console command remain `openvla-quic`.

| Module | Used by / responsibility |
| --- | --- |
| `__init__`, `__main__`, `cli` | Public `load_policy`, module entry point and commands |
| `configuration` | CLI JSON settings; explicit target resolution for tuning |
| `host` | Checkpoint identity, processor/model loading, normalization and prediction |
| `runtime` | Composition of host + saved PETF adapter for Python, validation and serving |
| `preprocessing` | Optional center crop and legacy prompt used by the host policy |
| `artifacts` | Shared file hashes and atomic JSON writes for data and tuning |
| `data` | Verified NPZ episodes, train-only statistics and tokenized datasets |
| `prepare` | Optional TFDS/OXE episode export through the CLI |
| `tuning` | Adapter-only training and held-out teacher-forced evaluation |
| `server` | Optional HTTP inference service through the CLI |

## Development and attribution

```bash
python -m pip install '.[dev,serve]'
python -m pytest tests/quic
python -m build
python -m twine check dist/*
```

OpenVLA is by Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti and the OpenVLA
contributors; see the upstream README for the full paper citation. QuIC is by
[Snehal Raj and Brian Coyle](https://arxiv.org/abs/2502.06916). The standalone
OpenVLA integration and extracted method packaging are by Jiří Konečný. The
portable method records its exact `quic-petf-mpv` source provenance in its own
repository. This fork retains OpenVLA's [MIT license](LICENSE).
