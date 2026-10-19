"""Prepare generic OXE/RLDS episodes with an explicit whole-episode split."""

from collections import defaultdict
import hashlib
import inspect
import math
from pathlib import Path

from .artifacts import sha256_file, write_json


def prepare(config, output):
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tqdm.auto import tqdm
    from .host import disable_tensorflow_gpu
    disable_tensorflow_gpu()
    from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
    from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
    from .data import SCHEMA
    settings = config["data"]
    name = settings["dataset_name"]
    if name not in OXE_DATASET_CONFIGS or name not in OXE_STANDARDIZATION_TRANSFORMS:
        raise ValueError("Source needs an OXE dataset mapping and action standardizer; no benchmark plugin is required")
    mapping = OXE_DATASET_CONFIGS[name]
    standardize = OXE_STANDARDIZATION_TRANSFORMS[name]
    image_key = mapping["image_obs_keys"]["primary"]
    builder = tfds.builder_from_directory(settings["tfds_directory"])
    split = settings.get("source_split", "train")
    seed = settings.get("seed", 7)
    fraction = settings.get("validation_fraction", .2)
    stride = settings.get("frame_stride", 1)
    limit = settings.get("max_frames_per_episode")
    episode_limit = settings.get("max_episodes_per_group")
    size = settings.get("image_size", 224)
    if (type(seed) is not int or not isinstance(fraction, (int, float)) or not 0 < fraction < 1
            or type(stride) is not int or stride < 1 or type(size) is not int or size < 1
            or any(value is not None and (type(value) is not int or value < 1) for value in (limit, episode_limit))):
        raise ValueError("Invalid episode selection, resize, or partition settings")
    decoders = {"steps": {"observation": {
        key: tfds.decode.SkipDecoding() for key, feature in builder.info.features["steps"]["observation"].items()
        if isinstance(feature, tfds.features.Image)}}}

    def text(value):
        value = value.numpy() if hasattr(value, "numpy") else value
        return value.decode() if isinstance(value, bytes) else str(value)

    def episodes():
        dataset = builder.as_dataset(split=split, shuffle_files=False,
            read_config=tfds.ReadConfig(add_tfds_id=True), decoders=decoders)
        options = tf.data.Options()
        options.deterministic = True
        options.threading.private_threadpool_size = 2
        yield from dataset.with_options(options)

    def trajectory(episode):
        steps = list(episode["steps"])
        if not steps:
            return None
        # Adjacent-action/gripper transforms must see the complete trajectory.
        return standardize(tf.nest.map_structure(lambda *parts: tf.stack(parts), *steps))

    grouped = defaultdict(list)
    for episode in tqdm(episodes(), total=int(builder.info.splits[split].num_examples), desc="Catalog data", unit="episode"):
        whole = trajectory(episode)
        if whole is None or not len(whole["action"]):
            continue
        group = text(whole["language_instruction"][0]).strip()
        if not group:
            raise ValueError("Each source episode requires a language instruction")
        grouped[group].append({"id": text(episode["tfds_id"]), "group": group})
    groups = settings.get("groups", sorted(grouped))
    if not groups or set(groups) - grouped.keys():
        raise ValueError("Requested source groups are absent or empty")
    selected = {}
    for group in sorted(groups):
        rows = sorted(grouped[group], key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
        rows = rows if episode_limit is None else rows[:episode_limit]
        if len(rows) < 2:
            raise ValueError("Each selected instruction group needs at least two whole episodes")
        count = min(len(rows) - 1, max(1, math.ceil(len(rows) * fraction)))
        for index, row in enumerate(rows):
            selected[row["id"]] = {**row, "split": "validation" if index < count else "train"}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records, dimension = [], None
    for episode in tqdm(episodes(), total=int(builder.info.splits[split].num_examples), desc="Prepare episodes", unit="episode"):
        identifier = text(episode["tfds_id"])
        if identifier not in selected:
            continue
        whole = trajectory(episode)
        indices = list(range(0, len(whole["action"]), stride))
        indices = indices if limit is None else indices[:limit]
        images, actions, instructions = [], [], []
        for index in indices:
            value = whole["observation"][image_key][index]
            image = tf.io.decode_image(value, channels=3, expand_animations=False) if value.dtype == tf.string else value
            image = tf.image.resize(image, (size, size), method="lanczos3", antialias=True)
            images.append(tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy())
            action = np.asarray(whole["action"][index].numpy(), dtype=np.float32)
            if action.ndim != 1 or not len(action) or not np.isfinite(action).all():
                raise ValueError("Standardized actions must be finite vectors")
            dimension = len(action) if dimension is None else dimension
            if len(action) != dimension:
                raise ValueError("Source action dimensions differ")
            actions.append(action)
            instructions.append(text(whole["language_instruction"][index]))
        file = hashlib.sha256(identifier.encode()).hexdigest() + ".npz"
        np.savez_compressed(output / file, images=np.stack(images), actions=np.stack(actions), instructions=np.asarray(instructions))
        records.append({**selected[identifier], "file": file, "sha256": sha256_file(output / file), "frames": len(indices)})
    if len(records) != len(selected):
        raise ValueError("Selected source episodes changed after cataloguing")
    mask = [True] * dimension
    encoding = getattr(mapping.get("action_encoding"), "name", "")
    if encoding in {"EEF_POS", "EEF_R6", "JOINT_POS"}:
        mask[-1] = False
    elif encoding == "JOINT_POS_BIMANUAL":
        if dimension % 2:
            raise ValueError("Bimanual action mapping requires two equal coordinate groups")
        mask[dimension // 2 - 1] = mask[-1] = False
    manifest = {"schema_version": SCHEMA, "episodes": sorted(records, key=lambda row: row["id"]),
        "action_mask": mask, "selection": settings,
        "preprocessing": {"resize": "tf-lanczos3-antialias-round-clip-uint8", "image_size": size,
                          "standardization": "complete trajectory before frame sampling"},
        "source": {"name": name, "split": split,
            "dataset_info_sha256": sha256_file(Path(builder.data_dir) / "dataset_info.json"),
            "standardizer_sha256": hashlib.sha256(inspect.getsource(standardize).encode()).hexdigest()}}
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"
