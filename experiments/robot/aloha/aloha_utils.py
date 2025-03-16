"""Utils for evaluating policies in real-world ALOHA environments."""

import os

import imageio
import numpy as np
from PIL import Image

from experiments.robot.aloha.real_env import make_real_env
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_next_task_label(task_label):
    """Prompt the user to input the next task."""
    if task_label == "":
        user_input = ""
        while user_input == "":
            user_input = input("Enter the task name: ")
        task_label = user_input
    else:
        user_input = input("Enter the task name (or leave blank to repeat the previous task): ")
        if user_input == "":
            pass  # Do nothing -> Let task_label be the same
        else:
            task_label = user_input
    print(f"Task: {task_label}")
    return task_label


def get_aloha_env():
    """Initializes and returns the ALOHA environment."""
    env = make_real_env(init_node=True)
    return env


def resize_image_for_preprocessing(img):
    """
    Takes numpy array corresponding to a single image and resizes to 256x256, exactly as done
    in the ALOHA data preprocessing script, which is used before converting the dataset to RLDS.
    """
    ALOHA_PREPROCESS_SIZE = 256
    img = np.array(
        Image.fromarray(img).resize((ALOHA_PREPROCESS_SIZE, ALOHA_PREPROCESS_SIZE), resample=Image.BICUBIC)
    )  # BICUBIC is default; specify explicitly to make it clear
    return img


def get_aloha_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    # obs: dm_env._environment.TimeStep
    img = obs.observation["images"]["cam_high"]
    img = resize_image_for_preprocessing(img)
    return img


def get_aloha_wrist_images(obs):
    """Extracts both wrist camera images from observations and preprocesses them."""
    # obs: dm_env._environment.TimeStep
    left_wrist_img = obs.observation["images"]["cam_left_wrist"]
    right_wrist_img = obs.observation["images"]["cam_right_wrist"]
    left_wrist_img = resize_image_for_preprocessing(left_wrist_img)
    right_wrist_img = resize_image_for_preprocessing(right_wrist_img)
    return left_wrist_img, right_wrist_img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, notes=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    filetag = f"{rollout_dir}/{DATE_TIME}--openvla--episode={idx}--success={success}--task={processed_task_description}"
    if notes is not None:
        filetag += f"--{notes}"
    mp4_path = f"{filetag}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=25)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path
