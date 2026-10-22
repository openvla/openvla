"""Optional readout lifecycle. The portable QuIC artifact format stays unchanged."""

import hashlib
import json
from pathlib import Path

from .artifacts import sha256_file


def _binding(identity, specification):
    return hashlib.sha256(json.dumps({"host_identity": identity, "action_head": specification},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save_policy_adapter(model, directory, host_identity, metadata=None):
    from quic_petf_vla import save_adapter
    from safetensors.torch import save_file
    import torch

    metadata = dict(metadata or {})
    if "action_head" in metadata:
        raise ValueError("action_head artifact metadata is owned by the model serializer")
    head = model.action_head
    if head is None:
        if host_identity.get("action_head") is not None:
            raise ValueError("Host identity requires a missing action head")
        return save_adapter(model, directory, host_identity, metadata=metadata)
    spec = dict(model.config.action_head)
    if host_identity.get("action_head") != spec:
        raise ValueError("Action head configuration must be bound to host identity")
    directory = Path(directory)
    if any((directory / name).exists() for name in ("adapter.json", "adapter.safetensors", "action_head.safetensors")):
        raise FileExistsError("Policy artifact already exists")
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in head.state_dict().items()}
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("Nonfinite action head weights")
    allowed = {id(p) for p in model.adaptation_parameters()}
    if any(p.requires_grad and id(p) not in allowed for p in model.parameters()):
        raise ValueError("Only QuIC adapters and the action head may be trained")
    directory.mkdir(parents=True, exist_ok=True)
    save_file(state, str(directory / "action_head.safetensors"),
              metadata={"action_head_binding_sha256": _binding(host_identity, spec)})
    metadata["action_head"] = {"config": spec, "tensor_sha256": sha256_file(directory / "action_head.safetensors")}
    flags = [p.requires_grad for p in head.parameters()]
    try:
        head.requires_grad_(False)
        return save_adapter(model, directory, host_identity, metadata=metadata)
    finally:
        for parameter, flag in zip(head.parameters(), flags):
            parameter.requires_grad_(flag)


def load_policy_adapter(model, directory, host_identity):
    from quic_petf_vla import load_adapter
    from safetensors import safe_open
    from safetensors.torch import load_file
    import torch

    directory = Path(directory)
    payload = json.loads((directory / "adapter.json").read_text())
    if payload.get("host_identity") != host_identity:
        raise ValueError("Policy host identity mismatch")
    record = payload.get("metadata", {}).get("action_head")
    spec = getattr(model.config, "action_head", None)
    if spec != host_identity.get("action_head"):
        raise ValueError("Model and host action head configurations differ")
    state = None
    if spec is None:
        if record is not None or (directory / "action_head.safetensors").exists():
            raise ValueError("Token policy cannot load action head weights")
    else:
        if not isinstance(record, dict) or set(record) != {"config", "tensor_sha256"} or record["config"] != spec:
            raise ValueError("Action head artifact configuration mismatch")
        target = directory / "action_head.safetensors"
        if sha256_file(target) != record["tensor_sha256"]:
            raise ValueError("Action head tensor SHA256 mismatch")
        with safe_open(str(target), framework="pt", device="cpu") as saved:
            if (saved.metadata() or {}).get("action_head_binding_sha256") != _binding(host_identity, spec):
                raise ValueError("Action head tensor binding mismatch")
        state = load_file(str(target), device="cpu")
        expected = model.action_head.state_dict()
        if set(state) != set(expected) or any(
                state[key].shape != expected[key].shape or state[key].dtype not in {torch.float32, torch.float16, torch.bfloat16}
                or not torch.isfinite(state[key]).all() for key in state):
            raise ValueError("Action head tensor inventory, dtype, shape or values mismatch")
        if any(entry["name"].startswith("action_head.") or entry["name"] == "language_model.lm_head"
               for entry in payload["manifest"]):
            raise ValueError("Chunk artifacts cannot target the action head or inactive lm_head")
    manifest = load_adapter(model, directory, host_identity)
    if state is not None:
        model.action_head.load_state_dict(state, strict=True)
        model.action_head.requires_grad_(True)
    return manifest


def chunk_prompt(instruction):
    if isinstance(instruction, bytes):
        instruction = instruction.decode("utf-8")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Chunk examples need a nonempty instruction")
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def action_window(actions, start, horizon, instructions=None):
    """Window original adjacent control steps; never cross an episode or instruction boundary."""
    import numpy as np
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 7 or not 0 <= start < len(values):
        raise ValueError("Action trajectories must have shape [T,7]")
    length = min(horizon, len(values) - start)
    if instructions is not None:
        for offset in range(1, length):
            if instructions[start + offset] != instructions[start]:
                length = offset
                break
    result = np.zeros((horizon, 7), dtype=np.float32)
    mask = np.arange(horizon) < length
    result[:length] = values[start:start + length]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite future action targets")
    return result, mask


class ChunkBatchTransform:
    """RLDS transform containing prompt-only inputs and separate regression targets."""

    def __init__(self, processor, horizon):
        self.processor, self.horizon = processor, horizon

    def __call__(self, frame):
        import numpy as np
        import torch
        from PIL import Image
        prompt = chunk_prompt(frame["task"]["language_instruction"])
        ids = self.processor.tokenizer(prompt, add_special_tokens=True).input_ids
        if not ids or len(ids) > self.processor.tokenizer.model_max_length:
            raise ValueError("Action prompt exceeds host context")
        actions = np.asarray(frame["action"], dtype=np.float32)
        mask = np.asarray(frame["action_mask"], dtype=bool)
        if actions.shape != (self.horizon, 7) or mask.shape != (self.horizon,) or not mask.any():
            raise ValueError("Malformed action chunk example")
        pixels = self.processor.image_processor.apply_transform(Image.fromarray(frame["observation"]["image_primary"][0]))
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "pixel_values": pixels,
                "actions": torch.from_numpy(actions.copy()), "action_mask": torch.from_numpy(mask.copy())}


def chunk_collator(processor):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    if processor.tokenizer.padding_side != "right" or processor.tokenizer.pad_token_id is None:
        raise ValueError("Chunk training requires right padding and a pad token")

    def collate(rows):
        lengths = torch.tensor([len(row["input_ids"]) for row in rows])
        ids = pad_sequence([row["input_ids"] for row in rows], batch_first=True,
                           padding_value=processor.tokenizer.pad_token_id)
        return {"input_ids": ids, "attention_mask": torch.arange(ids.shape[1])[None] < lengths[:, None],
                "pixel_values": torch.stack([row["pixel_values"] for row in rows]),
                "actions": torch.stack([row["actions"] for row in rows]),
                "action_mask": torch.stack([row["action_mask"] for row in rows])}
    return collate


def validate_chunks(model, loader, *, device, dtype=None, max_batches=4, description="Chunk validation"):
    import itertools
    import torch
    from tqdm.auto import tqdm
    model.eval()
    absolute_error = 0.0
    count = batches = steps = 0
    with torch.inference_mode():
        for raw in tqdm(itertools.islice(loader, max_batches), total=min(len(loader), max_batches),
                        desc=description, unit="batch"):
            batch = {key: value.to(device=device, dtype=(dtype or model.dtype) if key == "pixel_values" else None)
                     for key, value in raw.items()}
            output = model(**batch, use_cache=False)
            errors = (output.actions.float() - batch["actions"].float()).abs()
            selected = errors.masked_select(batch["action_mask"].unsqueeze(-1).expand_as(errors))
            if not torch.isfinite(selected).all() or not torch.isfinite(output.loss):
                raise ValueError("Nonfinite action chunk validation")
            absolute_error += float(selected.sum())
            count += selected.numel()
            steps += int(batch["action_mask"].sum())
            batches += 1
    if not count:
        raise ValueError("Validation contains no valid action chunk targets")
    return {"role": "action_chunk_regression_validation", "batches": batches, "valid_action_steps": steps,
            "action_coordinate_count": count, "normalized_l1": absolute_error / count, "loss": absolute_error / count}
