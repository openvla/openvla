"""Verified episode NPZ datasets; no benchmark imports or model orchestration."""

import bisect
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .artifacts import sha256_file

SCHEMA = "openvla-quic.episodes/v1"


def read_manifest(path):
    path = Path(path).expanduser().resolve()
    document = json.loads(path.read_text())
    # The source MVP's already prepared arrays can also be consumed directly.
    legacy = document.get("format") == "quic-openvla-libero-episodes" and document.get("version") == 1
    if document.get("schema_version") != SCHEMA and not legacy:
        raise ValueError("Unsupported episode manifest; run openvla-quic prepare or supply the documented NPZ schema")
    seen, files, counts = set(), set(), {"train": 0, "validation": 0}
    for row in document.get("episodes", []):
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            raise ValueError("Episode IDs must be unique across both partitions")
        seen.add(row["id"])
        if row.get("split") not in counts or type(row.get("frames")) is not int or row["frames"] < 1:
            raise ValueError("Each episode requires a valid split and positive frame count")
        target = (path.parent / row["file"]).resolve()
        if path.parent not in target.parents or not target.is_file():
            raise ValueError("Episode paths must name existing files within the prepared dataset")
        if target in files:
            raise ValueError("A physical episode file cannot appear under multiple IDs or partitions")
        files.add(target)
        if sha256_file(target) != row["sha256"]:
            raise ValueError(f"Episode content changed: {row['id']}")
        counts[row["split"]] += row["frames"]
    if not all(counts.values()):
        raise ValueError("Separate nonempty training and validation episodes are required")
    return path, document


def fit_statistics(path, manifest):
    """Fit only selected training frames. Validation arrays are never opened."""
    from tqdm.auto import tqdm
    rows = [row for row in manifest["episodes"] if row["split"] == "train"]
    actions = []
    for row in tqdm(rows, desc="Fit action normalization", unit="episode"):
        with np.load(path.parent / row["file"], allow_pickle=False) as archive:
            values = np.asarray(archive["actions"], dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != row["frames"] or not np.isfinite(values).all():
            raise ValueError("Training action arrays require finite [frames, action_dim] values")
        actions.append(values)
    values = np.concatenate(actions)
    low, high = np.quantile(values, [.01, .99], axis=0)
    mask = manifest.get("action_mask", [True] * values.shape[1])
    if len(mask) != values.shape[1] or any(type(item) is not bool for item in mask):
        raise ValueError("Manifest action_mask must be an explicit boolean vector matching action_dim")
    return {"q01": low.tolist(), "q99": high.tolist(), "min": values.min(0).tolist(),
            "max": values.max(0).tolist(), "mask": mask}


def normalize_actions(actions, statistics):
    actions = np.asarray(actions, dtype=np.float32)
    low, high = [np.asarray(statistics[name], dtype=np.float32) for name in ("q01", "q99")]
    mask = np.asarray(statistics.get("mask", [True] * len(low)), dtype=bool)
    if (low.ndim != 1 or low.shape != high.shape or mask.shape != low.shape or actions.shape != low.shape
            or not np.isfinite(actions).all() or not np.isfinite(low).all() or not np.isfinite(high).all()
            or np.any(high < low)):
        raise ValueError("Action dimensions or normalization bounds are invalid")
    result = np.where(mask, np.clip(2 * (actions - low) / (high - low + 1e-8) - 1, -1, 1), actions)
    if "min" in statistics and "max" in statistics:
        result = np.where(np.asarray(statistics["min"]) == np.asarray(statistics["max"]), 0, result)
    return result.astype(np.float32)


class EpisodeDataset(Dataset):
    def __init__(self, path, manifest, split, processor, statistics):
        from prismatic.vla.action_tokenizer import ActionTokenizer
        self.path, self.processor, self.statistics = path, processor, statistics
        self.rows = [row for row in manifest["episodes"] if row["split"] == split]
        self.ends = np.cumsum([row["frames"] for row in self.rows]).tolist()
        self.action_tokenizer = ActionTokenizer(processor.tokenizer)
        self.cached = None

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        position = bisect.bisect_right(self.ends, index)
        row = self.rows[position]
        frame = index - (self.ends[position - 1] if position else 0)
        if self.cached != position:
            with np.load(self.path.parent / row["file"], allow_pickle=False) as archive:
                self.arrays = {key: archive[key] for key in archive.files}
            if (len(self.arrays["images"]) != row["frames"] or len(self.arrays["actions"]) != row["frames"]
                    or self.arrays["images"].dtype != np.uint8 or self.arrays["images"].ndim != 4
                    or self.arrays["images"].shape[-1] != 3):
                raise ValueError("Episode arrays require matching frame counts and uint8 [F,H,W,3] images")
            if "instructions" in self.arrays and self.arrays["instructions"].shape != (row["frames"],):
                raise ValueError("instructions must contain one string per frame")
            self.cached = position
        instruction = (self.arrays["instructions"][frame] if "instructions" in self.arrays
                       else row.get("language_instruction", row.get("instruction", "")))
        instruction = instruction.decode("utf-8") if isinstance(instruction, bytes) else instruction
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Each frame requires a nonempty language instruction")
        action = normalize_actions(self.arrays["actions"][frame], self.statistics)
        tokenizer = self.processor.tokenizer
        prompt = (f"In: What action should the robot take to {instruction.lower()}?\nOut: "
                  f"{self.action_tokenizer(action).strip()}{tokenizer.eos_token}")
        ids = torch.tensor(tokenizer(prompt, add_special_tokens=True).input_ids, dtype=torch.long)
        dimension = len(action)
        action_ids = ids[-dimension - 1:-1]
        if (len(ids) > tokenizer.model_max_length or len(action_ids) != dimension
                or int(ids[-1]) != tokenizer.eos_token_id
                or not ((action_ids > self.action_tokenizer.action_token_begin_idx)
                        & (action_ids < tokenizer.vocab_size)).all()):
            raise ValueError("Prompt must fit the host context and supervise exactly its action tokens before EOS")
        labels = ids.clone()
        labels[:-dimension - 1] = -100
        labels[-1] = -100
        pixels = self.processor.image_processor.apply_transform(Image.fromarray(self.arrays["images"][frame]))
        return {"input_ids": ids, "labels": labels, "pixel_values": pixels}
