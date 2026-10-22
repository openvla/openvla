"""Local token OpenVLA loading with immutable, content-verified host identity."""
import hashlib
import json
import re
import time
from types import MethodType
from pathlib import Path

import torch
from PIL import Image

from .artifacts import sha256_file


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _inline_action_normalization(host):
    """Validate the fitted, content-bound normalization overlay if present."""
    inline = host.get("action_normalization")
    digest = host.get("normalization_contract_sha256")
    if inline is None and digest is None:
        return None
    key = host.get("unnorm_key")
    if not isinstance(inline, dict) or set(inline) != {key}:
        raise ValueError("fitted action_normalization must contain exactly the selected unnorm_key")
    record = inline[key]
    if not isinstance(record, dict) or set(record) != {"action"} or not isinstance(record["action"], dict):
        raise ValueError("fitted action_normalization requires an action statistics object")
    stats = record["action"]
    low, high = stats.get("q01"), stats.get("q99")
    if (not isinstance(low, list) or not isinstance(high, list) or not low or len(low) != len(high)
            or any(type(v) not in (int, float) for v in low + high)
            or any(a > b for a, b in zip(low, high))):
        raise ValueError("fitted action statistics require ordered equal-length numeric q01/q99 vectors")
    mask = stats.get("mask")
    if mask is not None and (not isinstance(mask, list) or len(mask) != len(low)
                             or any(type(value) is not bool for value in mask)):
        raise ValueError("fitted action statistics mask must be a boolean vector matching q01/q99")
    contract = {"unnorm_key": key, "action_statistics": stats, "mode": "train-q99"}
    calculated = hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if not isinstance(digest, str) or digest != calculated:
        raise ValueError("fitted normalization contract SHA256 mismatch")
    return json.loads(json.dumps(inline, allow_nan=False))


def apply_action_normalization(model, host):
    """Overlay fitted train-split statistics in memory, leaving host files intact."""
    inline = _inline_action_normalization(host)
    if inline is not None:
        model.norm_stats = dict(model.norm_stats or {})
        for key, record in inline.items():
            model.norm_stats[key] = {**model.norm_stats.get(key, {}), **record}


def checkpoint_path(host):
    path = Path(host["checkpoint"]).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Local checkpoint directory not found: {path}")
    if host.get("family", "openvla_token") != "openvla_token":
        raise ValueError("This workflow implements token OpenVLA; OpenVLA-OFT is a separate host")
    if not re.fullmatch(r"[0-9a-f]{40}", host.get("revision", "")):
        raise ValueError("Declare the immutable 40-character checkpoint revision")
    if host.get("adaptation_state") not in {"merged", "none"}:
        raise ValueError("Host must be a verified dense merged checkpoint or unadapted checkpoint")
    if (path / "adapter_config.json").exists() or (path / "lora_adapter").exists():
        raise ValueError("Active/unmerged LoRA is not supported in this host loader")
    checkpoint_config = json.loads((path / "config.json").read_text())
    if checkpoint_config.get("model_type") != "openvla":
        raise ValueError("Checkpoint is not token OpenVLA")
    if checkpoint_config.get("quantization_config"):
        raise ValueError("QuIC requires dense host weights; quantized checkpoints are unsupported")
    return path


def checkpoint_identity(host):
    from quic_petf_vla import adapter as method
    path = checkpoint_path(host)
    files = sorted(p for p in path.iterdir() if p.is_file() and
                   (p.suffix in {".json", ".model", ".safetensors"}))
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Checkpoint must contain safetensors weights")
    # Verify bytes on every load; file timestamps alone are not a content proof.
    hashes = {p.name: sha256_file(p) for p in files}
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if any("lora_" in k for k in index["weight_map"]):
            raise ValueError("Checkpoint contains active LoRA parameters")
        if not set(index["weight_map"].values()).issubset(hashes):
            raise ValueError("Missing checkpoint shards")
    from safetensors import safe_open
    for shard in (p for p in files if p.suffix == ".safetensors"):
        with safe_open(str(shard), framework="pt", device="cpu") as saved:
            if any("lora_" in key for key in saved.keys()):
                raise ValueError("Checkpoint contains active LoRA parameters")
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    normalization_sha256 = None
    if host.get("norm_stats_path"):
        norm_stats_path = Path(host["norm_stats_path"]).expanduser().resolve()
        if not norm_stats_path.is_file():
            raise ValueError(f"Normalization statistics file not found: {norm_stats_path}")
        normalization_sha256 = sha256_file(norm_stats_path)
    # Bind the model implementation actually loaded by this fork. The adapter
    # implementation has its own identity in the artifact metadata.
    source = {str(p.relative_to(SOURCE_ROOT)): sha256_file(p)
              for p in sorted((SOURCE_ROOT / "prismatic").rglob("*.py"))}
    identity = {"family": "openvla_token", "checkpoint_id": host.get("checkpoint_id", "content-addressed-local"), "revision": host["revision"],
                "adaptation_state": host["adaptation_state"], "checkpoint_sha256": digest,
                "source_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
                "quic_method_sha256": sha256_file(method.__file__),
                "integration_source_sha256": hashlib.sha256(json.dumps({
                    name: sha256_file(Path(__file__).parent / name)
                    for name in ("host.py", "preprocessing.py", "runtime.py", "artifacts.py", "action_heads.py")
                }, sort_keys=True).encode()).hexdigest()}
    if host.get("action_head") is not None:
        from prismatic.extern.hf.action_chunks import validate_action_head
        identity["action_head"] = validate_action_head(host["action_head"])
    if normalization_sha256:
        identity["normalization_stats_sha256"] = normalization_sha256
    if _inline_action_normalization(host) is not None:
        identity["normalization_contract_sha256"] = host["normalization_contract_sha256"]
    return identity


def disable_tensorflow_gpu():
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    # Keep TFDS from spawning one thread per host CPU during small demos.
    try:
        if not tf.config.threading.get_inter_op_parallelism_threads():
            tf.config.threading.set_inter_op_parallelism_threads(2)
        if not tf.config.threading.get_intra_op_parallelism_threads():
            tf.config.threading.set_intra_op_parallelism_threads(2)
    except RuntimeError:
        # Thread settings are an optimization and cannot change after a caller
        # has initialized TF. GPU exclusion above remains mandatory.
        pass


def _verify_model_source():
    import prismatic
    actual = Path(prismatic.__file__).resolve().parent
    expected = (SOURCE_ROOT / "prismatic").resolve()
    if actual != expected:
        raise RuntimeError(f"Loaded prismatic source {actual} differs from QuIC fork {expected}")


def load_processor(host):
    _verify_model_source()
    from transformers import AutoTokenizer
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    path = checkpoint_path(host)
    image_processor = PrismaticImageProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)


def module_inventory(model):
    """Inventory weighted targets, retaining unsupported types for explicit errors.

    Output dimensions consistently name the dimension transformed by QuIC.
    Conv2d rotates output channels, independently at every spatial position.
    """
    entries = []
    for name, module in model.named_modules():
        if "weight" not in module._parameters or module._parameters["weight"] is None:
            continue
        if name.startswith("action_head."):
            continue  # The optional readout is trained in full, outside QuIC targets.
        entry = {"name": name, "type": type(module).__name__,
                 "supported": type(module) in (torch.nn.Linear, torch.nn.Conv2d),
                 "weight_shape": list(module.weight.shape), "used_in_policy": True}
        if type(module) is torch.nn.Linear:
            entry.update(in_features=module.in_features, out_features=module.out_features,
                         bias=module.bias is not None)
        elif type(module) is torch.nn.Conv2d:
            entry.update(in_features=module.in_channels, out_features=module.out_channels,
                         bias=module.bias is not None, module_type="Conv2d",
                         in_channels=module.in_channels, out_channels=module.out_channels,
                         kernel_size=list(module.kernel_size), stride=list(module.stride),
                         padding=module.padding if isinstance(module.padding, str) else list(module.padding),
                         dilation=list(module.dilation), groups=module.groups,
                         padding_mode=module.padding_mode)
        match = re.match(r"vision_backbone\.(featurizer|fused_featurizer)\.blocks\.(\d+)\.", name)
        if match:
            encoder = getattr(model.vision_backbone, match[1])
            if int(match[2]) > len(encoder.blocks) - 2:
                entry["used_in_policy"] = False
                entry["inactive_reason"] = "OpenVLA returns second-to-last vision block features"
        # get_intermediate_layers(norm=False) also bypasses forward_head and
        # its SigLIP attention pool, plus the encoder's final normalization.
        if re.match(r"vision_backbone\.(?:featurizer|fused_featurizer)\.(?:attn_pool|fc_norm|head|norm)(?:\.|$)", name):
            entry["used_in_policy"] = False
            entry["inactive_reason"] = "OpenVLA intermediate vision features bypass final normalization, pooling and classifier"
        if name == "language_model.lm_head" and getattr(model, "action_head", None) is not None:
            entry["used_in_policy"] = False
            entry["inactive_reason"] = "Continuous action chunks bypass the vocabulary head"
        entries.append(entry)
    return entries


def _checkpointed_vision_block(module, *args, **kwargs):
    from torch.utils.checkpoint import checkpoint
    original = module._quic_original_forward
    # Vision intentionally stays in eval mode during adapter training, so the
    # autograd context, rather than module.training, decides whether to checkpoint.
    if torch.is_grad_enabled():
        return checkpoint(original, *args, use_reentrant=False, **kwargs)
    return original(*args, **kwargs)


def enable_gradient_checkpointing(model):
    """Checkpoint Llama and the timm blocks used by intermediate vision features.

    timm's set_grad_checkpointing only affects forward_features, while OpenVLA
    calls get_intermediate_layers. Preserve module names and artifact targets
    by checkpointing each original block forward without adding wrapper modules.
    """
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    count = 0
    for name in ("featurizer", "fused_featurizer"):
        encoder = getattr(model.vision_backbone, name, None)
        if encoder is None:
            continue
        for block in encoder.blocks:
            if not hasattr(block, "_quic_original_forward"):
                block._quic_original_forward = block.forward
                block.forward = MethodType(_checkpointed_vision_block, block)
            count += 1
    return {"schema_version": "openvla.gradient-checkpointing/v1", "language": True,
            "vision_blocks": count, "use_reentrant": False}


def text_hidden_size(inventory):
    """Get decoder width from the meta-model inventory, not sparse HF JSON defaults."""
    for entry in inventory:
        if entry["name"] == "language_model.model.layers.0.self_attn.q_proj":
            return entry["in_features"]
    raise ValueError("OpenVLA inventory is missing the first Llama q_proj dimensions")


def layer_inventory(host):
    """Instantiate architecture on meta device; no 7B weights are allocated."""
    _verify_model_source()
    from accelerate import init_empty_weights
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    path = checkpoint_path(host)
    config = OpenVLAConfig.from_pretrained(path, local_files_only=True)
    if "action_head" in host:
        from prismatic.extern.hf.action_chunks import validate_action_head
        config.action_head = validate_action_head(host["action_head"])
    if getattr(config.text_config, "pretraining_tp", 1) != 1:
        raise ValueError("pretraining_tp must equal 1: TP sliced F.linear branches bypass adapter wrappers")
    with init_empty_weights():
        model = OpenVLAForActionPrediction(config)
    return module_inventory(model), config.text_config.num_hidden_layers


def load_host(host, training=False):
    _verify_model_source()
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    path = checkpoint_path(host)
    device = host.get("device", "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: execute outside GPU-restricted sandbox")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[host.get("dtype", "bfloat16")]
    if training and dtype == torch.float16:
        raise ValueError("Reference trainer supports BF16 or FP32; FP16 requires a scaler")
    config = OpenVLAConfig.from_pretrained(path, local_files_only=True)
    if "action_head" in host:
        from prismatic.extern.hf.action_chunks import validate_action_head
        config.action_head = validate_action_head(host["action_head"])
    if getattr(config.text_config, "pretraining_tp", 1) != 1:
        raise ValueError("pretraining_tp must equal 1: TP sliced F.linear branches bypass adapter wrappers")
    new_head = config.action_head if host.get("action_head") is not None else None
    if new_head is not None:
        config.action_head = None
    start = time.perf_counter()
    model = OpenVLAForActionPrediction.from_pretrained(
        path, config=config, local_files_only=True, torch_dtype=dtype,
        attn_implementation=host.get("attention", "flash_attention_2"), low_cpu_mem_usage=True)
    if new_head is not None:
        model.configure_action_head(new_head)
    model.to(device).eval()
    stats = path / "dataset_statistics.json"
    if stats.exists():
        model.norm_stats = json.loads(stats.read_text())
    if host.get("norm_stats_path"):
        stats_path = Path(host["norm_stats_path"]).expanduser().resolve()
        if not stats_path.is_file():
            raise ValueError(f"Normalization statistics file not found: {stats_path}")
        payload = json.loads(stats_path.read_text())
        model.norm_stats = payload.get("norm_stats", payload)
    apply_action_normalization(model, host)
    key = host.get("unnorm_key")
    if key not in (model.norm_stats or {}):
        raise ValueError(f"Host normalization key {key!r} missing; available: {list(model.norm_stats or {})}")
    processor = load_processor(host)
    runtime = {"cold_load_seconds": time.perf_counter() - start,
               "device": device, "dtype": str(dtype),
               "torch_version": torch.__version__, "torch_cuda_runtime": torch.version.cuda}
    if device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(device)
        runtime.update(gpu_name=properties.name, gpu_total_memory_bytes=properties.total_memory)
    return model, processor, runtime


class OpenVLAQuICPolicy:
    def __init__(self, model, processor, host, baseline=False):
        self.model, self.processor, self.host = model, processor, host
        self.baseline = baseline

    def predict(self, image, instruction):
        from contextlib import nullcontext
        from quic_petf_vla import disabled_adapters
        pil_image = Image.fromarray(image).convert("RGB")
        if self.host.get("center_crop", False):
            disable_tensorflow_gpu()
            import numpy as np
            import tensorflow as tf
            from .preprocessing import crop_and_resize
            values = tf.image.convert_image_dtype(tf.convert_to_tensor(np.array(pil_image)), tf.float32)
            values = crop_and_resize(values, 0.9, 1)
            pil_image = Image.fromarray(tf.image.convert_image_dtype(tf.clip_by_value(values, 0, 1), tf.uint8, saturate=True).numpy())
        style = self.host.get("prompt_style", "modern")
        if style == "modern":
            prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        elif style == "v01":
            from .preprocessing import OPENVLA_V01_SYSTEM_PROMPT
            prompt = f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
        else:
            raise ValueError("prompt_style must be modern or v01")
        if getattr(self.model, "action_head", None) is not None:
            if style != "modern":
                raise ValueError("Action chunk heads require the modern prompt")
            from .action_heads import chunk_prompt
            prompt = chunk_prompt(instruction)
        inputs = self.processor(prompt, pil_image)
        inputs = inputs.to(self.host.get("device", "cuda:0"), dtype=self.model.dtype)
        self.model.eval()
        context = disabled_adapters(self.model) if self.baseline else nullcontext()
        with context, torch.inference_mode():
            return self.model.predict_action(**inputs, unnorm_key=self.host["unnorm_key"], do_sample=False)
