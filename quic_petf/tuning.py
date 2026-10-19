"""Standalone adapter-only tuning and finite teacher-forced validation."""

import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random

from .artifacts import sha256_file, write_json


def move_batch(batch, device, dtype):
    return {key: value.to(device=device, dtype=dtype if key == "pixel_values" else None)
            for key, value in batch.items()}


def make_loader(dataset, processor, batch_size, *, shuffle=False, seed=7):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader
    if processor.tokenizer.padding_side != "right" or processor.tokenizer.pad_token_id is None:
        raise ValueError("Training requires right padding and a host pad token")
    def collate(rows):
        ids = pad_sequence([row["input_ids"] for row in rows], batch_first=True,
                           padding_value=processor.tokenizer.pad_token_id)
        return {"input_ids": ids, "attention_mask": ids.ne(processor.tokenizer.pad_token_id),
                "labels": pad_sequence([row["labels"] for row in rows], batch_first=True, padding_value=-100),
                "pixel_values": torch.stack([row["pixel_values"] for row in rows])}
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle,
                      collate_fn=collate, generator=torch.Generator().manual_seed(seed))


def validate(model, loader, *, device, max_batches=4, description="Validation"):
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm
    model.eval()
    count = correct = batches = 0
    loss = 0.0
    with torch.inference_mode():
        for raw in tqdm(itertools.islice(loader, max_batches), total=min(len(loader), max_batches),
                        desc=description, unit="batch"):
            batch = move_batch(raw, device, model.dtype)
            output = model(**batch, use_cache=False)
            labels = batch["labels"][:, 1:]
            logits = output.logits[:, -batch["labels"].shape[1]:-1]
            mask = labels != -100
            target, predictions = labels[mask], logits[mask].float()
            if target.numel():
                cross_entropy = F.cross_entropy(predictions, target, reduction="sum")
                if not torch.isfinite(cross_entropy):
                    raise ValueError("Validation produced nonfinite action-token loss")
                loss += float(cross_entropy)
                count += target.numel()
                correct += int((predictions.argmax(-1) == target).sum())
            batches += 1
    if not count:
        raise ValueError("Validation contains no supervised action tokens")
    return {"role": "teacher_forced_validation", "batches": batches, "action_tokens": count,
            "action_token_cross_entropy": loss / count, "action_token_accuracy": correct / count}


def training_settings(config):
    defaults = {"max_steps": 3, "batch_size": 1, "gradient_accumulation_steps": 1,
                "validation_batches": 4, "learning_rate": 1e-4, "seed": 7,
                "gradient_checkpointing": True, "max_trainable_parameters": 1000000}
    values = config.get("training", {})
    if set(values) - defaults.keys():
        raise ValueError(f"Unknown training fields: {sorted(set(values) - defaults.keys())}")
    defaults.update(values)
    for key in ("max_steps", "batch_size", "gradient_accumulation_steps", "validation_batches", "max_trainable_parameters"):
        if type(defaults[key]) is not int or defaults[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if (type(defaults["seed"]) is not int or defaults["seed"] < 0
            or type(defaults["gradient_checkpointing"]) is not bool
            or not isinstance(defaults["learning_rate"], (int, float))
            or not math.isfinite(defaults["learning_rate"]) or defaults["learning_rate"] <= 0):
        raise ValueError("Invalid seed, gradient-checkpointing flag, or learning rate")
    return defaults


def train(config, output, *, fit_action_normalization=False):
    import numpy as np
    import torch
    from tqdm.auto import tqdm
    from quic_petf_vla import QuICLinear, disabled_adapters
    from .configuration import explicit_targets
    from .data import EpisodeDataset, fit_statistics, read_manifest
    from .host import checkpoint_identity, layer_inventory, load_host
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The standalone adapter trainer currently supports a single process/device")
    config = copy.deepcopy(config)
    settings = training_settings(config)
    host = config["host"]
    if host.get("prompt_style", "modern") != "modern" or host.get("center_crop", False):
        raise ValueError("This trainer uses the modern prompt without crop augmentation")
    random.seed(settings["seed"])
    np.random.seed(settings["seed"] % 2**32)
    torch.manual_seed(settings["seed"])
    path, manifest = read_manifest(config["data"]["manifest"])
    inventory, _ = layer_inventory(host)
    targets = explicit_targets(config["targets"], inventory)
    count = sum((1 if cfg.share_across_blocks else cfg.block_count) *
                (cfg.base_size * (cfg.base_size - 1) // 2 if cfg.orthogonal else cfg.base_size**2)
                for name, target in targets.items() for cfg in [target.resolve(next(
                    row["out_features"] for row in inventory if row["name"] == name))])
    if count > settings["max_trainable_parameters"]:
        raise ValueError("Adapter parameter budget exceeded before loading model weights")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    normalization = None
    if fit_action_normalization:
        stats = fit_statistics(path, manifest)
        normalization = {"unnorm_key": host["unnorm_key"], "action_statistics": stats, "mode": "train-q99"}
        normalization["sha256"] = hashlib.sha256(json.dumps(normalization, sort_keys=True, allow_nan=False).encode()).hexdigest()
        host["action_normalization"] = {host["unnorm_key"]: {"action": stats}}
        host["normalization_contract_sha256"] = normalization["sha256"]
        write_json(output / "fitted-action-normalization.json", normalization)
    identity = checkpoint_identity(host)
    model, processor, runtime = load_host(host, training=True)
    device = host.get("device", "cuda:0")
    stats = model.norm_stats[host["unnorm_key"]]["action"]
    if normalization is None:
        normalization = {"unnorm_key": host["unnorm_key"], "action_statistics": stats, "mode": "host"}
        normalization["sha256"] = hashlib.sha256(json.dumps(normalization, sort_keys=True, allow_nan=False).encode()).hexdigest()
    train_ds = EpisodeDataset(path, manifest, "train", processor, stats)
    val_ds = EpisodeDataset(path, manifest, "validation", processor, stats)
    loader = make_loader(train_ds, processor, settings["batch_size"], shuffle=True, seed=settings["seed"])
    val_loader = make_loader(val_ds, processor, settings["batch_size"])
    write_json(output / "config.json", config)
    write_json(output / "data-manifest.json", manifest)
    write_json(output / "normalization.json", normalization)
    write_json(output / "host-identity.json", identity)
    sample = move_batch(next(iter(val_loader)), device, model.dtype)
    def reference():
        model.eval()
        with torch.inference_mode():
            logits = model(**sample, use_cache=False).logits[:, -sample["labels"].shape[1]:-1]
            return logits[sample["labels"][:, 1:] != -100].float().cpu()
    baseline_reference = reference()
    selection = model.enable_quic(targets)
    if not torch.equal(reference(), baseline_reference):
        raise ValueError("QuIC initialization changed host action-token logits")
    del baseline_reference
    parameters = model.quic_parameters()
    initial = [parameter.detach().cpu().clone() for parameter in parameters]
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in parameters}:
        raise ValueError("Optimizer would include parameters outside QuIC adapters")
    if settings["gradient_checkpointing"]:
        model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    with disabled_adapters(model):
        baseline = validate(model, val_loader, device=device, max_batches=settings["validation_batches"], description="Host validation")
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=0.0)
    iterator = iter(loader)
    with (output / "metrics.jsonl").open("x", buffering=1) as metrics:
        with tqdm(total=settings["max_steps"], desc="QuIC training", unit="step") as progress:
            for step in range(1, settings["max_steps"] + 1):
                model.train()
                model.vision_backbone.eval()
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for _ in range(settings["gradient_accumulation_steps"]):
                    try:
                        raw = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        raw = next(iterator)
                    batch = move_batch(raw, device, model.dtype)
                    loss = model(**batch, use_cache=False).loss
                    if loss is None or not torch.isfinite(loss):
                        raise ValueError("Training produced a missing or nonfinite loss")
                    losses.append(float(loss.detach()))
                    (loss / settings["gradient_accumulation_steps"]).backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise ValueError("A targeted adapter has a missing or nonfinite gradient")
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                value = sum(losses) / len(losses)
                metrics.write(json.dumps({"optimizer_step": step, "training_loss": value}) + "\n")
                progress.set_postfix(loss=f"{value:.4f}")
                progress.update()
    if any(torch.equal(before, after.detach().cpu()) for before, after in zip(initial, parameters)):
        raise ValueError("A targeted QuIC tensor did not update")
    final = validate(model, val_loader, device=device, max_batches=settings["validation_batches"], description="Adapter validation")
    learned = reference()
    split_id = sha256_file(path)
    metadata = {"normalization_contract": normalization, "split_manifest_id": split_id,
                "preprocessing": {"prompt_style": "modern", "center_crop": False},
                "peft": {"method": "quic", "adapter": {"profile": "custom"}},
                "training": settings, "source": manifest.get("source", {})}
    model.save_quic_adapter(output / "adapter", identity, metadata=metadata)
    processor.save_pretrained(output / "processor")
    del optimizer, parameters
    for name, module in list(model.named_modules()):
        if isinstance(module, QuICLinear):
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, child, module.base_layer)
    model.load_quic_adapter(output / "adapter", identity)
    if not torch.equal(reference(), learned):
        raise ValueError("Serialized adapter reload changed action-token logits")
    result = {"status": "completed", "optimizer_steps": settings["max_steps"], "trainable_parameters": count,
              "baseline_validation": baseline, "adapter_validation": final, "host_identity": identity,
              "normalization_contract": normalization, "target_manifest": selection, "runtime": runtime,
              "identity_logit_max_absolute_error": 0.0, "reload_logit_max_absolute_error": 0.0,
              "adapter_path": str(output / "adapter")}
    write_json(output / "result.json", result)
    return result


def evaluate(config, adapter, output):
    from .runtime import load_policy
    from .data import EpisodeDataset, read_manifest
    policy = load_policy(config["host"], adapter)
    path, manifest = read_manifest(config["data"]["manifest"])
    stats = policy.model.norm_stats[policy.host["unnorm_key"]]["action"]
    dataset = EpisodeDataset(path, manifest, "validation", policy.processor, stats)
    settings = training_settings(config)
    loader = make_loader(dataset, policy.processor, settings["batch_size"])
    result = validate(policy.model, loader, device=policy.host.get("device", "cuda:0"), max_batches=settings["validation_batches"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "result.json", result)
    return result
