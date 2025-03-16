"""
Preprocesses ALOHA dataset(s) and splits them into train/val sets.

Preprocessing includes downsizing images from 480x640 to 256x256.
Splits happen at the episode level (not step level), which means that
an episode is treated as an atomic unit that entirely goes to either
the train set or val set.

Original ALOHA data layout:
    /PATH/TO/DATASET/dataset_name/
        - episode_0.hdf5
        - episode_1.hdf5
        - ...
        - episode_N.hdf5

Preprocessed data layout (after running this script):
    /PATH/TO/PREPROCESSED_DATASETS/dataset_name/
        - train/
            - episode_0.hdf5
            - episode_1.hdf5
            - ...
            - episode_M.hdf5
        - val/
            - episode_0.hdf5
            - episode_1.hdf5
            - ...
            - episode_K.hdf5

    where N > M > K

Example usage:
    # "put X into pot" task
    python experiments/robot/aloha/preprocess_split_aloha_data.py \
        --dataset_path /scr/moojink/data/aloha1_raw/put_green_pepper_into_pot/ \
        --out_base_dir /scr/moojink/data/aloha1_preprocessed/ \
        --percent_val 0.05 && \
    python experiments/robot/aloha/preprocess_split_aloha_data.py \
        --dataset_path /scr/moojink/data/aloha1_raw/put_red_pepper_into_pot/ \
        --out_base_dir /scr/moojink/data/aloha1_preprocessed/ \
        --percent_val 0.05 && \
    python experiments/robot/aloha/preprocess_split_aloha_data.py \
        --dataset_path /scr/moojink/data/aloha1_raw/put_yellow_corn_into_pot/ \
        --out_base_dir /scr/moojink/data/aloha1_preprocessed/ \
        --percent_val 0.05
"""

import argparse
import glob
import os
import random

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_hdf5(demo_path):
    """Loads single episode."""
    if not os.path.isfile(demo_path):
        print(f"Dataset does not exist at \n{demo_path}\n")
        exit()

    print(f"Loading {demo_path}...")
    with h5py.File(demo_path, "r") as root:
        is_sim = root.attrs["sim"]
        qpos = root["/observations/qpos"][()]
        qvel = root["/observations/qvel"][()]
        effort = root["/observations/effort"][()]
        action = root["/action"][()]
        image_dict = dict()
        for cam_name in root["/observations/images/"].keys():
            image_dict[cam_name] = root[f"/observations/images/{cam_name}"][()]
    print(f"Loading episode complete: {demo_path}")

    return qpos, qvel, effort, action, image_dict, is_sim


def load_and_preprocess_all_episodes(demo_paths, out_dataset_dir):
    """
    Loads and preprocesses all episodes.
    Resizes all images in one episode before loading the next, to reduce memory usage.
    """
    cam_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    idx = 0
    for demo in tqdm(demo_paths):
        qpos, qvel, effort, action, image_dict, is_sim = load_hdf5(demo)
        # Save non-image info
        episode_len = image_dict["cam_high"].shape[0]
        # Resize all images
        print("Resizing images in episode...")
        for k in cam_names:
            resized_images = []
            for i in range(episode_len):
                resized_images.append(
                    np.array(
                        Image.fromarray(image_dict[k][i]).resize(
                            (args.img_resize_size, args.img_resize_size), resample=Image.BICUBIC
                        )
                    )  # BICUBIC is default; specify explicitly to make it clear
                )
            image_dict[k] = np.stack(resized_images)
        print("Resizing images in episode complete!")
        # Save preprocessed episode
        data_dict = dict(
            qpos=qpos,
            qvel=qvel,
            effort=effort,
            action=action,
            image_dict=image_dict,
            is_sim=is_sim,
        )
        save_new_hdf5(out_dataset_dir, data_dict, idx)
        idx += 1


def randomly_split(full_qpos, full_qvel, full_effort, full_action, full_image_dict, percent_val):
    """Randomly splits dataset into train and validation sets."""
    # Create a list of episode indices
    num_episodes_total = len(full_qpos)
    indices = list(range(num_episodes_total))
    # Shuffle the episode indices
    random.shuffle(indices)
    # Create new lists using the shuffled indices
    shuffled_qpos = [full_qpos[idx] for idx in indices]
    shuffled_qvel = [full_qvel[idx] for idx in indices]
    shuffled_effort = [full_effort[idx] for idx in indices]
    shuffled_action = [full_action[idx] for idx in indices]
    shuffled_image_dict = {
        "cam_high": [],
        "cam_left_wrist": [],
        "cam_right_wrist": [],
    }
    for k in full_image_dict.keys():
        shuffled_image_dict[k] = [full_image_dict[k][idx] for idx in indices]
    # Split into train and val sets
    num_episodes_val = int(num_episodes_total * percent_val)
    print(f"Total # steps: {num_episodes_total}; using {num_episodes_val} ({percent_val:.2f}%) for val set")
    num_episodes_train = num_episodes_total - num_episodes_val
    train_dict = dict(
        qpos=shuffled_qpos[:num_episodes_train],
        qvel=shuffled_qvel[:num_episodes_train],
        effort=shuffled_effort[:num_episodes_train],
        action=shuffled_action[:num_episodes_train],
        image_dict=dict(
            cam_high=shuffled_image_dict["cam_high"][:num_episodes_train],
            cam_left_wrist=shuffled_image_dict["cam_left_wrist"][:num_episodes_train],
            cam_right_wrist=shuffled_image_dict["cam_right_wrist"][:num_episodes_train],
        ),
    )
    val_dict = dict(
        qpos=shuffled_qpos[num_episodes_train:],
        qvel=shuffled_qvel[num_episodes_train:],
        effort=shuffled_effort[num_episodes_train:],
        action=shuffled_action[num_episodes_train:],
        image_dict=dict(
            cam_high=shuffled_image_dict["cam_high"][num_episodes_train:],
            cam_left_wrist=shuffled_image_dict["cam_left_wrist"][num_episodes_train:],
            cam_right_wrist=shuffled_image_dict["cam_right_wrist"][num_episodes_train:],
        ),
    )
    return train_dict, val_dict


def save_new_hdf5(out_dataset_dir, data_dict, episode_idx):
    """Saves an HDF5 file for a new episode."""
    camera_names = data_dict["image_dict"].keys()
    H, W, C = data_dict["image_dict"]["cam_high"][0].shape
    out_path = os.path.join(out_dataset_dir, f"episode_{episode_idx}.hdf5")
    # Save HDF5 with same structure as original demos (except that now we combine all episodes into one HDF5 file)
    with h5py.File(
        out_path, "w", rdcc_nbytes=1024**2 * 2
    ) as root:  # Magic constant for rdcc_nbytes comes from ALOHA codebase
        episode_len = data_dict["qpos"].shape[0]
        root.attrs["sim"] = data_dict["is_sim"]
        obs = root.create_group("observations")
        _ = obs.create_dataset("qpos", (episode_len, 14))
        _ = obs.create_dataset("qvel", (episode_len, 14))
        _ = obs.create_dataset("effort", (episode_len, 14))
        root["/observations/qpos"][...] = data_dict["qpos"]
        root["/observations/qvel"][...] = data_dict["qvel"]
        root["/observations/effort"][...] = data_dict["effort"]
        image = obs.create_group("images")
        for cam_name in camera_names:
            _ = image.create_dataset(
                cam_name,
                (episode_len, H, W, C),
                dtype="uint8",
                chunks=(1, H, W, C),
            )
            root[f"/observations/images/{cam_name}"][...] = data_dict["image_dict"][cam_name]
        _ = root.create_dataset("action", (episode_len, 14))
        root["/action"][...] = data_dict["action"]
        # Compute and save *relative* actions as well
        actions = data_dict["action"]
        relative_actions = np.zeros_like(actions)
        relative_actions[:-1] = actions[1:] - actions[:-1]  # Relative actions are the changes in joint pos
        relative_actions[-1] = relative_actions[-2]  # Just copy the second-to-last action for the last action
        _ = root.create_dataset("relative_action", (episode_len, 14))
        root["/relative_action"][...] = relative_actions
    print(f"Saved dataset: {out_path}")


def main(args):
    # Create directory to save preprocessed dataset (if it doesn't exist already)
    os.makedirs(args.out_base_dir, exist_ok=True)
    out_dataset_dir = os.path.join(args.out_base_dir, os.path.basename(args.dataset_path.rstrip("/")))
    os.makedirs(out_dataset_dir, exist_ok=True)
    # Get list of filepaths of all episodes
    all_demo_paths = glob.glob(os.path.join(args.dataset_path, "*.hdf5"))  # List of HDF5 filepaths
    all_demo_paths.sort()
    # Create a list of episode indices
    num_episodes_total = len(all_demo_paths)
    indices = list(range(num_episodes_total))
    # Shuffle the episode indices
    random.shuffle(indices)
    # Split into train and val sets
    num_episodes_val = int(num_episodes_total * args.percent_val)
    print(f"Total # episodes: {num_episodes_total}; using {num_episodes_val} ({args.percent_val:.2f}%) for val set")
    num_episodes_train = num_episodes_total - num_episodes_val
    train_indices = indices[:num_episodes_train]
    val_indices = indices[num_episodes_train:]
    train_demo_paths = [all_demo_paths[i] for i in train_indices]
    val_demo_paths = [all_demo_paths[i] for i in val_indices]
    # Preprocess all episodes and save the result
    out_dataset_dir_train = os.path.join(out_dataset_dir, "train")
    out_dataset_dir_val = os.path.join(out_dataset_dir, "val")
    os.makedirs(out_dataset_dir_train, exist_ok=True)
    os.makedirs(out_dataset_dir_val, exist_ok=True)
    load_and_preprocess_all_episodes(train_demo_paths, out_dataset_dir_train)
    load_and_preprocess_all_episodes(val_demo_paths, out_dataset_dir_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to raw ALOHA dataset directory. Example: /PATH/TO/USER/data/aloha_raw/put_green_pepper_into_pot/",
    )
    parser.add_argument(
        "--out_base_dir",
        required=True,
        help="Path to directory in which to save preprocessed dataset. Example: /PATH/TO/USER/data/aloha_preprocessed/",
    )
    parser.add_argument(
        "--percent_val",
        type=float,
        help="Percent of dataset to use as validation set (measured in episodes, not steps).",
        default=0.05,
    )
    parser.add_argument(
        "--img_resize_size",
        type=int,
        help="Size to resize images to. Final images will be square (img_resize_size x img_resize_size pixels).",
        default=256,
    )
    args = parser.parse_args()

    main(args)
