# Standalone QuIC experiments

These are small explicit recipes, independent of OVLAB and any simulator.
Copy `projector.json` and replace the checkpoint directory, immutable source
revision, normalization key and prepared-data manifest. Paths are relative to
the JSON file unless absolute; command-line path overrides are relative to the
working directory. No named QP/A profile or ablation hierarchy is interpreted.

```bash
openvla-quic inventory --config experiments/quic/projector.json
openvla-quic train --config experiments/quic/projector.json \
  --output outputs/projector-run --fit-action-normalization
openvla-quic evaluate --config experiments/quic/projector.json \
  --adapter outputs/projector-run/adapter --output outputs/projector-eval
openvla-quic predict --config experiments/quic/projector.json \
  --adapter outputs/projector-run/adapter --image observation.png \
  --instruction 'pick up the object'
```

`inventory` instantiates the architecture on a meta device without loading 7B
weights. Targets must name actual policy-active `torch.nn.Linear` modules and
must fit the configured compound dimensions. `projector.fc2` is a 4096-output
example; discover the actual names/dimensions for your checkpoint first.
Training is single-process, dense-host, adapter-only. Validation metrics use
teacher forcing and do not measure environment task success. Output directories
must be new. Results and full metrics are persisted; `--json` explicitly opts
into a complete console result document.

## Data preparation

Install `.[rlds]`, copy `prepare.json`, and select a TFDS builder directory and
its OpenVLA OXE mapping name. The source dataset must already be downloaded.

```bash
openvla-quic prepare --config experiments/quic/prepare.json --output outputs/prepared
```

The OXE standardizer sees whole trajectories before frame sampling. Selection
and train/validation splitting use whole TFDS episode IDs within instruction
groups; groups are exact instruction strings. At least two episodes per selected
group are required. Preparation records source/standardizer digests, resizes RGB
with TF Lanczos3, and exports read-only-source data to verified NPZ episodes.
No benchmark package is imported. Optional `data.groups` selects exact
instruction groups.

Other data providers can write the same `openvla-quic.episodes/v1` manifest:

```json
{
  "schema_version": "openvla-quic.episodes/v1",
  "source": {"name": "my_robot"},
  "action_mask": [true, true, true, true, true, true, false],
  "episodes": [
    {"id": "episode-1", "split": "train", "file": "episode-1.npz", "sha256": "FILE_SHA256", "frames": 4},
    {"id": "episode-2", "split": "validation", "file": "episode-2.npz", "sha256": "FILE_SHA256", "frames": 4}
  ]
}
```

Each NPZ has `images: uint8[F,H,W,3]`, finite raw standardized
`actions: float[F,D]`, and `instructions: unicode[F]`. A row-level `instruction`
can replace the array when constant throughout an episode. Paths stay inside
the manifest directory; IDs must be unique across splits. `action_mask` declares
which coordinates to scale into [-1,1]; omitted masks scale every coordinate.
The seven-coordinate mask above is an example, not a benchmark dependency.
Previously prepared source-MVP episode manifests are accepted as an input
format; add an explicit action mask when fitting their normalization.

Without `--fit-action-normalization`, training uses frozen host statistics.
The flag fits q01/q99 on training episodes only, saves the contract, and binds
it to the adapter. Inference restores the same statistics from artifact metadata.

## Optional robot benchmarks

The existing LIBERO and Bridge scripts accept `--quic_adapter_path`,
`--quic_base_revision`, `--quic_adaptation_state`, `--quic_attention`, and
`--quic_norm_stats_path`. Install the simulator/controller separately, alongside
`.[train]` and the benchmark's own requirements. For a non-cropped QuIC artifact:

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla --pretrained_checkpoint /models/openvla-7b \
  --quic_adapter_path outputs/projector-run/adapter \
  --quic_base_revision YOUR_IMMUTABLE_REVISION \
  --quic_adaptation_state none --quic_attention sdpa \
  --task_suite_name libero_10 --center_crop False --use_wandb False
```

The selected benchmark normalization key must match the artifact. Benchmark
action transforms and simulator conventions remain in the experiment scripts.
