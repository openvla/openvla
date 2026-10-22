"""Checkpoint and adapter composition for direct Python use, CLI, and serving."""

import copy
import json
from pathlib import Path


def adapter_host(host, adapter_path):
    """Restore the saved preprocessing/normalization contract before host loading."""
    host = copy.deepcopy(host)
    payload = json.loads((Path(adapter_path) / "adapter.json").read_text())
    saved = payload["host_identity"]
    host.setdefault("checkpoint_id", saved["checkpoint_id"])
    if "action_head" in host and host["action_head"] != saved.get("action_head"):
        raise ValueError("Requested model action head differs from artifact")
    if saved.get("action_head") is not None:
        host["action_head"] = saved["action_head"]
    metadata = payload.get("metadata", {})
    preprocessing = metadata.get("preprocessing", {})
    for key in ("prompt_style", "center_crop"):
        if key in preprocessing:
            if key in host and host[key] != preprocessing[key]:
                raise ValueError(f"Adapter preprocessing mismatch: {key}")
            host[key] = preprocessing[key]
    normalization = metadata.get("normalization_contract", {})
    key = normalization.get("unnorm_key")
    if key:
        if host.get("unnorm_key", key) != key:
            raise ValueError("Adapter and host unnorm_key differ")
        host["unnorm_key"] = key
    if normalization.get("mode") == "train-q99":
        host["action_normalization"] = {key: {"action": normalization["action_statistics"]}}
        host["normalization_contract_sha256"] = normalization["sha256"]
    return host, payload


def load_policy(host, adapter_path=None, *, baseline=False):
    """Load a local dense checkpoint and optional QuIC artifact without OVLAB.

    Returns a policy with ``predict(rgb_uint8, instruction)``. The underlying
    model remains available as ``policy.model`` with normal HF APIs.
    """
    from .host import OpenVLAQuICPolicy, checkpoint_identity, load_host
    host = copy.deepcopy(host)
    payload = None
    if adapter_path is not None:
        host, payload = adapter_host(host, adapter_path)
    identity = checkpoint_identity(host)
    if payload is not None and identity != payload["host_identity"]:
        raise ValueError("Adapter host identity does not match checkpoint, method, source, or normalization")
    model, processor, runtime = load_host(host)
    if adapter_path is not None:
        model.load_quic_adapter(adapter_path, identity)
        contract = payload.get("metadata", {}).get("normalization_contract", {})
        if contract.get("action_statistics") != model.norm_stats[host["unnorm_key"]]["action"]:
            raise ValueError("Loaded normalization differs from the adapter contract")
    policy = OpenVLAQuICPolicy(model, processor, host, baseline=baseline)
    policy.host_identity, policy.runtime = identity, runtime
    return policy
