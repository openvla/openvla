import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import json
import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    fig_draw_dir: str = "./results"

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under
    
    num_seg: int = 10
    seed: int = 7                                    # Random Seed (for reproducibility)
    device: str = 'cuda:0'
    # fmt: on

def _cosine(u, v, eps: float = 1e-12):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    num = np.sum(u * v, axis=-1)
    den = (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)) + eps
    cos = num / den
    if cos.ndim == 0:
        return float(cos)
    return cos

def _row_normalize(W):
    s = W.sum(axis=1, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    return W / s

def _entropy_vec(W):
    Wc = np.clip(W, 1e-12, 1.0)
    return -(Wc * np.log(Wc)).sum(axis=1)

def _swap_ratio_pairwise(W):
    if W.shape[0] <= 1:
        return np.zeros(0, dtype=float)
    diffs = np.abs(W[1:] - W[:-1]).sum(axis=1)
    s_prev = np.sort(W[:-1], axis=1)[:, ::-1]
    s_next = np.sort(W[1:],  axis=1)[:, ::-1]
    diffs_shape = np.abs(s_next - s_prev).sum(axis=1)
    num = np.maximum(0.0, diffs - diffs_shape)
    den = np.maximum(1e-12, diffs)
    return num / den

def _segment_means(E: np.ndarray, S: int = 10):
    E = np.asarray(E, dtype=np.float64)
    if E.ndim != 2 or E.shape[0] == 0:
        raise ValueError("E must be (T, D) with T > 0")

    T, D = E.shape
    if S <= 0:
        raise ValueError("S must be a positive integer")

    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

    seg_ids = np.clip(((np.arange(T) + 1) * S // T) - 1, 0, S - 1).astype(int)

    seg_means = np.zeros((S, D), dtype=np.float64)
    seg_counts = np.zeros(S, dtype=int)
    for s in range(S):
        idx = np.where(seg_ids == s)[0]
        if idx.size > 0:
            m = En[idx].mean(axis=0)
            seg_means[s] = m / (np.linalg.norm(m) + 1e-12)
            seg_counts[s] = idx.size
        else:
            seg_means[s] = np.nan 

    if np.isnan(seg_means).any():
        global_mean = En.mean(axis=0)
        global_mean /= (np.linalg.norm(global_mean) + 1e-12)

        for s in range(S):
            if not np.isnan(seg_means[s]).any():
                continue
            left = next((k for k in range(s - 1, -1, -1) if seg_counts[k] > 0), None)
            right = next((k for k in range(s + 1, S) if seg_counts[k] > 0), None)

            if left is not None and right is not None:
                m = 0.5 * (seg_means[left] + seg_means[right])
            elif left is not None:
                m = seg_means[left]
            elif right is not None:
                m = seg_means[right]
            else:
                m = global_mean
            seg_means[s] = m / (np.linalg.norm(m) + 1e-12)

    return seg_means, seg_counts

def analyze_episode_two_metrics(ep_name, eps_buf, save_dir, N=10):
    os.makedirs(save_dir, exist_ok=True)
    if "w_seq" not in eps_buf or len(eps_buf["w_seq"]) == 0:
        raise ValueError("eps_buf['w_seq'] is empty or does not exist.")

    W = np.stack(eps_buf["w_seq"], axis=0).astype(np.float64)  # (T, M)
    if W.ndim != 2:
        raise ValueError(f"W shape is illegal: {W.shape}")
    T, M = W.shape
    if T < N:
        raise ValueError(f"T={T} is less than N={N}")
    W = _row_normalize(W)

    H_t  = _entropy_vec(W)    
    SR_t = _swap_ratio_pairwise(W) 

    desired_keys = ["p1", "p3", "p5", "p10"]
    focus_pair_map = {}        
    focus_overall_mean_map = {}     
    focus_seg_mean_map = {}        
    focus_seg_count_map = {}     

    raw_pair_map = eps_buf.get("focus_shift_pair_t_map", {}) or {}
    idx = np.linspace(0, T, N + 1, dtype=int)

    for k in desired_keys:
        fs_seq = np.asarray(raw_pair_map.get(k, []), dtype=np.float64)
        focus_pair_map[k] = fs_seq 

        if fs_seq.size > 0:
            v = fs_seq[np.isfinite(fs_seq)]
            focus_overall_mean_map[k] = float(np.mean(v)) if v.size > 0 else float("nan")
        else:
            focus_overall_mean_map[k] = float("nan")

        seg_means = np.full(N, np.nan, dtype=np.float64)
        seg_counts = np.zeros(N, dtype=int)
        for s in range(N):
            a, b = idx[s], idx[s+1]
            if fs_seq.size == 0:
                continue
            end = min(b - 1, fs_seq.size)
            start = min(a, end)
            if end - start >= 1:
                v = fs_seq[start:end]
                v = v[np.isfinite(v)]
                if v.size > 0:
                    seg_means[s] = float(np.mean(v))
                    seg_counts[s] = int(v.size)
        focus_seg_mean_map[k] = seg_means.tolist()
        focus_seg_count_map[k] = seg_counts.tolist()

    rows = []
    for s in range(N):
        a, b = idx[s], idx[s+1]
        Hs = H_t[a:b]
        H_mean = float(np.mean(Hs))
        H_std  = float(np.std(Hs, ddof=1)) if Hs.size > 1 else 0.0
        SR_mean = float(np.mean(SR_t[a:b-1])) if (b - a) >= 2 else float("nan")

        rows.append({
            "segment": int(s),
            "t_start": int(a),
            "t_end":   int(b - 1),
            "H_mean": H_mean,
            "H_std":  H_std,
            "swapratio_mean": SR_mean,
            "focus_p1_mean":  focus_seg_mean_map["p1"][s],
            "focus_p3_mean":  focus_seg_mean_map["p3"][s],
            "focus_p5_mean":  focus_seg_mean_map["p5"][s],
            "focus_p10_mean": focus_seg_mean_map["p10"][s],
        })

    out_json = {
        "episode": ep_name,
        "T": int(T),
        "N": int(N),
        "rows": rows, 
        "H_t": H_t.tolist(),
        "swapratio_pair_t": SR_t.tolist(),
        "focus_top_pcts": [0.01, 0.03, 0.05, 0.10],
        "focus_shift_pair_t_map": {k: focus_pair_map.get(k, np.array([], dtype=np.float64)).tolist() for k in desired_keys}, 
        "focus_shift_overall_mean_map": {k: float(focus_overall_mean_map[k]) for k in desired_keys},
        "focus_shift_seg_mean_map": {k: focus_seg_mean_map[k] for k in desired_keys},
        "focus_shift_seg_count_map": {k: focus_seg_count_map[k] for k in desired_keys},  
    }

    out_path = os.path.join(save_dir, f"{ep_name}_two_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)

    return {
        "rows": rows,
        "H_t": H_t,
        "swapratio_pair_t": SR_t,
        "focus_shift_pair_t_map": focus_pair_map,
        "focus_shift_overall_mean_map": focus_overall_mean_map,
        "focus_shift_seg_mean_map": focus_seg_mean_map,
        "focus_shift_seg_count_map": focus_seg_count_map,
        "json_path": out_path,
    }

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            
            eps_buf = dict()

            # Reset environment
            env.reset()

            # Set initial states

            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        task_id,
                        episode_idx,
                        t,
                        eps_buf,
                        processor=processor,
                    )

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break
            
            analyze_episode_two_metrics(ep_name=f"ep_{task_id+1}_{episode_idx+1}", eps_buf=eps_buf, save_dir=os.path.join(cfg.fig_draw_dir, cfg.task_suite_name, 'seed_'+str(cfg.seed)), N=cfg.num_seg)
            task_episodes += 1
            total_episodes += 1

            save_rollout_video(
                cfg.task_suite_name, cfg.seed, replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
