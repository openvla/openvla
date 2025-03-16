"""
run_aloha_eval.py

Evaluates a model in a real-world ALOHA environment.
"""

import logging
import os
import socket
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import tqdm

# Append current directory so that interpreter can find experiments.robot
sys.path.append(".")
from experiments.robot.aloha.aloha_utils import (
    get_aloha_env,
    get_aloha_image,
    get_aloha_wrist_images,
    get_next_task_label,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_from_server,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    set_seed_everywhere,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 25                    # Number of actions to execute open-loop before requerying policy

    use_vla_server: bool = True                      # Whether to query remote VLA server for actions
    vla_server_url: Union[str, Path] = ""            # Remote VLA server URL (set to 127.0.0.1 if on same machine)

    #################################################################################################################
    # ALOHA environment-specific parameters
    #################################################################################################################
    num_rollouts_planned: int = 50                   # Number of test rollouts
    max_steps: int = 1500                            # Max number of steps per rollout
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.use_vla_server, (
        "Must use VLA server (server-client interface) to query model and get actions! Please set --use_vla_server=True"
    )


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file."""
    # Create run ID
    run_id = f"EVAL-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    print(message)
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def get_server_endpoint(cfg: GenerateConfig):
    """Get the server endpoint for remote inference."""
    ip_address = socket.gethostbyname(cfg.vla_server_url)
    return f"http://{ip_address}:8777/act"


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_aloha_image(obs)
    left_wrist_img, right_wrist_img = get_aloha_wrist_images(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    left_wrist_img_resized = resize_image_for_policy(left_wrist_img, resize_size)
    right_wrist_img_resized = resize_image_for_policy(right_wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "left_wrist_image": left_wrist_img_resized,
        "right_wrist_image": right_wrist_img_resized,
        "state": obs.observation["qpos"],
    }

    return observation, img_resized, left_wrist_img_resized, right_wrist_img_resized


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    server_endpoint: str,
    resize_size,
    log_file=None,
):
    """Run a single episode in the ALOHA environment."""
    # Define control frequency
    STEP_DURATION_IN_SEC = 1.0 / 25.0

    # Reset environment
    obs = env.reset()

    # Initialize action queue
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    curr_state = None
    replay_images = []
    replay_images_resized = []
    replay_images_left_wrist_resized = []
    replay_images_right_wrist_resized = []

    log_message("Prepare the scene, and then press Enter to begin...", log_file)
    input()

    # Reset environment again to fetch first timestep observation
    obs = env.reset()

    # Fetch initial robot state (but sleep first so that robot stops moving)
    time.sleep(2)
    curr_state = env.get_qpos()

    episode_start_time = time.time()
    total_model_query_time = 0.0

    try:
        while t < cfg.max_steps:
            # Get step start time (used to compute how much to sleep between steps)
            step_start_time = time.time()

            # Get observation
            obs = env.get_observation(t=t)

            # Save raw high camera image for replay video
            replay_images.append(obs.observation["images"]["cam_high"])

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Prepare observation
                observation, img_resized, left_wrist_resized, right_wrist_resized = prepare_observation(obs, resize_size)
                observation["instruction"] = task_description

                # Save processed images for replay
                replay_images_resized.append(img_resized)
                replay_images_left_wrist_resized.append(left_wrist_resized)
                replay_images_right_wrist_resized.append(right_wrist_resized)

                # Query model to get action
                log_message("Requerying model...", log_file)
                model_query_start_time = time.time()
                actions = get_action_from_server(observation, server_endpoint)
                actions = actions[: cfg.num_open_loop_steps]
                total_model_query_time += time.time() - model_query_start_time
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()
            log_message("-----------------------------------------------------", log_file)
            log_message(f"t: {t}", log_file)
            log_message(f"action: {action}", log_file)

            # Execute action in environment
            if cfg.use_relative_actions:
                # Get absolute joint angles from relative action
                rel_action = action
                target_state = curr_state + rel_action
                obs = env.step(target_state.tolist())
                # Update current state (assume it is the commanded target state)
                curr_state = target_state
            else:
                obs = env.step(action.tolist())
            t += 1

            # Sleep until next timestep
            step_elapsed_time = time.time() - step_start_time
            if step_elapsed_time < STEP_DURATION_IN_SEC:
                time_to_sleep = STEP_DURATION_IN_SEC - step_elapsed_time
                log_message(f"Sleeping {time_to_sleep} sec...", log_file)
                time.sleep(time_to_sleep)

    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            log_message("\nCaught KeyboardInterrupt: Terminating episode early.", log_file)
        else:
            log_message(f"\nCaught exception: {e}", log_file)

    episode_end_time = time.time()

    # Get success feedback from user
    user_input = input("Success? Enter 'y' or 'n': ")
    success = True if user_input.lower() == "y" else False

    # Calculate episode statistics
    episode_stats = {
        "success": success,
        "total_steps": t,
        "model_query_time": total_model_query_time,
        "episode_duration": episode_end_time - episode_start_time,
    }

    return (
        episode_stats,
        replay_images,
        replay_images_resized,
        replay_images_left_wrist_resized,
        replay_images_right_wrist_resized,
    )


def save_episode_videos(
    replay_images,
    replay_images_resized,
    replay_images_left_wrist,
    replay_images_right_wrist,
    episode_idx,
    success,
    task_description,
    log_file=None,
):
    """Save videos of the episode from different camera angles."""
    # Save main replay video
    save_rollout_video(replay_images, episode_idx, success=success, task_description=task_description, log_file=log_file)

    # Save processed view videos
    save_rollout_video(
        replay_images_resized,
        episode_idx,
        success=success,
        task_description=task_description,
        log_file=log_file,
        notes="resized",
    )
    save_rollout_video(
        replay_images_left_wrist,
        episode_idx,
        success=success,
        task_description=task_description,
        log_file=log_file,
        notes="left_wrist_resized",
    )
    save_rollout_video(
        replay_images_right_wrist,
        episode_idx,
        success=success,
        task_description=task_description,
        log_file=log_file,
        notes="right_wrist_resized",
    )


@draccus.wrap()
def eval_aloha(cfg: GenerateConfig) -> None:
    """Main function to evaluate a trained policy in a real-world ALOHA environment."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Get ALOHA environment
    env = get_aloha_env()

    # Get server endpoint for remote inference
    server_endpoint = get_server_endpoint(cfg)

    # Initialize task description
    task_description = ""

    # Start evaluation
    num_rollouts_completed, total_successes = 0, 0

    for episode_idx in tqdm.tqdm(range(cfg.num_rollouts_planned)):
        # Get task description from user
        task_description = get_next_task_label(task_description)
        log_message(f"\nTask: {task_description}", log_file)

        log_message(f"Starting episode {num_rollouts_completed + 1}...", log_file)

        # Run episode
        episode_stats, replay_images, replay_images_resized, replay_images_left_wrist, replay_images_right_wrist = (
            run_episode(cfg, env, task_description, server_endpoint, resize_size, log_file)
        )

        # Update counters
        num_rollouts_completed += 1
        if episode_stats["success"]:
            total_successes += 1

        # Save videos
        save_episode_videos(
            replay_images,
            replay_images_resized,
            replay_images_left_wrist,
            replay_images_right_wrist,
            num_rollouts_completed,
            episode_stats["success"],
            task_description,
            log_file,
        )

        # Log results
        log_message(f"Success: {episode_stats['success']}", log_file)
        log_message(f"# episodes completed so far: {num_rollouts_completed}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / num_rollouts_completed * 100:.1f}%)", log_file)
        log_message(f"Total model query time: {episode_stats['model_query_time']:.2f} sec", log_file)
        log_message(f"Total episode elapsed time: {episode_stats['episode_duration']:.2f} sec", log_file)

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(num_rollouts_completed) if num_rollouts_completed > 0 else 0

    # Log final results
    log_message("\nFinal results:", log_file)
    log_message(f"Total episodes: {num_rollouts_completed}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_aloha()
