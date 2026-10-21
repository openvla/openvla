"""Standalone run settings. Named research profiles/ablations belong to callers."""

import copy
import json
from pathlib import Path


def load_config(path):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected a JSON object with schema_version: 1")
    unknown = set(config) - {"schema_version", "host", "targets", "data", "training"}
    if unknown:
        raise ValueError(f"Unknown standalone config fields: {sorted(unknown)}")
    config = copy.deepcopy(config)
    for section in ("host", "data", "training", "targets"):
        if not isinstance(config.get(section, {}), dict):
            raise ValueError(f"{section} must be an object")
    for section, field in (("host", "checkpoint"), ("host", "norm_stats_path"),
                           ("data", "manifest"), ("data", "tfds_directory")):
        value = config.get(section, {}).get(field)
        if value is not None:
            value = Path(value).expanduser()
            config[section][field] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    return config


def explicit_targets(specification, inventory):
    """Resolve only exact model-module names and plain AdapterConfig values."""
    from quic_petf_vla import AdapterConfig
    if not isinstance(specification, dict) or not specification:
        raise ValueError("targets must map exact Linear/Conv2d module names to AdapterConfig objects")
    lookup = {row["name"]: row for row in inventory}
    selected = {}
    for name, values in specification.items():
        if name not in lookup or not lookup[name].get("used_in_policy", True):
            raise ValueError(f"Target is absent or does not contribute to policy outputs: {name}")
        if not lookup[name].get("supported", True):
            raise ValueError(f"Unsupported QuIC target type {lookup[name]['type']}: {name}")
        if not isinstance(values, dict):
            raise ValueError(f"Target {name} requires an explicit AdapterConfig object")
        config = AdapterConfig(**values)
        if lookup[name].get("module_type") == "Conv2d" and config.block_count % lookup[name]["groups"]:
            raise ValueError(f"Conv2d block_count must be divisible by groups: {name}")
        config.resolve(lookup[name]["out_features"])
        selected[name] = config
    return selected
