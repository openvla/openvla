"""
convert_nomagic_ur5e_raw.py

Convert raw CSV/MP4 data from Nomagic's UR5e robot arm into a LeRobotDataset.

The data should be in a format similar to the following:

raw/trajectories/
urXPose_20241219_133722.csv
urXPose_20241219_133814.csv
  ...

raw/videos/
  2024-12-19-12:37:26:897865_d5fb919d-2b3a-4d4b-b885-1f890a255b66.mp4
  2024-12-19-12:38:19:058352_99a53149-eb87-4a0d-979f-01b1283c5804.mp4
  ...

The final directory will look like:

data/my_lerobot_dataset/
  data/
    chunk-000/
      episode_000000.parquet
      episode_000001.parquet
      ...
  meta/
    info.json
    stats.json
    episodes.jsonl
    tasks.jsonl
  videos/
    chunk-000/
      observation.images.side/
        episode_000000.mp4
        episode_000001.mp4
        ...

Notes:
  - This script assumes that the CSV files have a matching MP4 file
    by a shared timestamp in the filename, e.g."urXPose_20241219_133722.csv" 
    ↔ "2024-12-19-12:37:26:897865_d5fb919d-2b3a-4d4b-b885-1f890a255b66.mp4"
"""

import os
import re
import json
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import cv2

from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Add a stream handler to show messages in console
    ]
)

@dataclass
class LeRobotFrame:
    """
    Represents a single entry (frame) in the final LeRobot parquet data.
    For example:
      timestamp (sec),
      episode_index,
      next.done (bool),
      task_index,
      index,        # global index in the episode
      frame_index,  # might be same as index, or you can offset if needed
      action        # a list of [dx, dy, dz, dox, doy, doz, grip]
    """
    timestamp: float
    episode_index: int
    next_done: bool
    task_index: int
    index: int
    frame_index: int
    action: List[float]

def find_pairs(raw_traj_dir: Path, raw_video_dir: Path):
    csv_files = sorted(raw_traj_dir.glob("*.csv"))
    mp4_files = sorted(raw_video_dir.glob("*.mp4"))

    logging.debug(f"Found CSV files: {[f.name for f in csv_files]}")
    logging.debug(f"Found MP4 files: {[f.name for f in mp4_files]}")

    # Just pair them up in order since they're already sorted chronologically
    pairs = list(zip(csv_files, mp4_files))
    
    for csv_f, mp4_f in pairs:
        logging.debug(f"Paired {csv_f.name} with {mp4_f.name}")

    return pairs

def compute_actions_from_rows(rowA: pd.Series, rowB: pd.Series):
    """
    Convert two consecutive CSV rows into a delta action (dx, dy, dz, dox, doy, doz, grip).
    Calculate Euler angle difference (dox, doy, doz) from quaternion pair using scipy.Rotation.
    """
    # Position deltas
    dx = float(rowB["PositionX"] - rowA["PositionX"]) 
    dy = float(rowB["PositionY"] - rowA["PositionY"])
    dz = float(rowB["PositionZ"] - rowA["PositionZ"])

    # Get quaternions in [x,y,z,w] format for scipy.Rotation
    q1 = [rowA["OrientationX"], rowA["OrientationY"], 
          rowA["OrientationZ"], rowA["OrientationW"]]
    q2 = [rowB["OrientationX"], rowB["OrientationY"],
          rowB["OrientationZ"], rowB["OrientationW"]]
    
    # Calculate orientation difference using quaternion_difference logic
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r_diff = r2 * r1.inv()
    euler_diff = r_diff.as_euler("xyz")
    dox, doy, doz = euler_diff
    
    # Gripper - map Gripper::Action values to float
    grip_action_map = {
        "Gripper::Action::NONE": 0.0,
        "Gripper::Action::RELEASE": -1.0, 
        "Gripper::Action::GRAB": 1.0
    }
    if rowA["GripperAction"] not in grip_action_map:
        raise ValueError(f"Unknown gripper action: {rowA['GripperAction']}")
    grip_val = grip_action_map[rowA["GripperAction"]]
    
    return [dx, dy, dz, dox, doy, doz, grip_val]

def convert_single_episode(
    csv_file: Path,
    mp4_file: Path,
    episode_index: int,
    out_dir: Path
) -> None:
    """
    Convert one CSV + MP4 into a single "episode_{:06d}.parquet" and
    copy the MP4 to "episode_{:06d}.mp4" in observation.images.side subdir.
    """
    # Load CSV data
    df = pd.read_csv(csv_file)
    # Convert timestamps to relative seconds from start of episode
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    first_time = df["Timestamp"].iloc[0]
    df["timestamp"] = (df["Timestamp"] - first_time).dt.total_seconds()

    # For convenience, define a small list of frames. We'll fill them up.
    final_frames: List[LeRobotFrame] = []

    for i in range(len(df) - 1):
        rowA = df.iloc[i]
        rowB = df.iloc[i + 1]
        # Build an action
        action_vals = compute_actions_from_rows(rowA, rowB)
        # Use the rowA's timestamp (relative seconds from start of episode)
        ts_val = float(rowA["timestamp"])
        # next.done is usually False unless e.g. i == len(df) - 2
        next_done = i == (len(df) - 2)
        # Build the frame record
        final_frames.append(
            LeRobotFrame(
                timestamp=ts_val,
                episode_index=episode_index,
                next_done=next_done,
                task_index=0,
                index=i,
                frame_index=i,
                action=action_vals
            )
        )

    # Write out as a parquet file
    out_parquet = out_dir / f"episode_{episode_index:06d}.parquet"
    pa_frames = pa.Table.from_pydict({
        "timestamp": [f.timestamp for f in final_frames],
        "episode_index": [f.episode_index for f in final_frames],
        "next.done": [f.next_done for f in final_frames],
        "task_index": [f.task_index for f in final_frames],
        "index": [f.index for f in final_frames],
        "frame_index": [f.frame_index for f in final_frames],
        "action": pa.array([f.action for f in final_frames], type=pa.list_(pa.float32()))
    })
    pq.write_table(pa_frames, out_parquet)
    print(f"[Episode {episode_index}] Saved parquet =>  {out_parquet}")

    # Copy MP4 to the expected output location
    video_outdir = out_dir.parent.parent \
        / "videos" \
        / "chunk-000" \
        / "observation.images.side"
    video_outdir.mkdir(parents=True, exist_ok=True)
    episode_mp4 = video_outdir / f"episode_{episode_index:06d}.mp4"
    shutil.copy(mp4_file, episode_mp4)
    print(f"[Episode {episode_index}] Saved MP4 => {episode_mp4}")

def build_meta_files(out_root: Path, total_episodes: int, episode_lengths: List[int]):
    meta_dir = out_root / "meta"
    meta_dir.mkdir(exist_ok=True)

    # Read trajectory fps, average over all episodes
    out_parquet_dir = out_root / "data" / "chunk-000"
    parquet_files = os.listdir(out_parquet_dir)
    parquet_files = [f for f in parquet_files if f.endswith(".parquet")]
    parquet_file_fps = []
    for parquetf in parquet_files:
        df = pd.read_parquet(out_parquet_dir / parquetf)
        inv_fps = df['timestamp'].diff().mean()
        fps = 1.0 / inv_fps
        parquet_file_fps.append(fps)
        print(f"{fps=}")
    mean_trajectory_fps = int(sum(parquet_file_fps) / len(parquet_file_fps))

    # Get total number of video frames
    video_dir = out_root / "videos" / "chunk-000" / "observation.images.side"
    video_frame_counts = []
    for episode_idx in range(total_episodes):
        video_path = video_dir / f"episode_{episode_idx:06d}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_counts.append(frame_count)
        cap.release()
    total_video_frames = sum(video_frame_counts)

    # Read video metadata from first video file
    first_video = next((out_root / "videos" / "chunk-000" / "observation.images.side").glob("*.mp4"))
    cap = cv2.VideoCapture(str(first_video))
    
    # Get basic video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read the first frame to get number of channels
    ret, frame = cap.read()
    channels = frame.shape[2] if ret else 3
    
    # Get codec information
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    cap.release()

    # info.json
    info_data = {
        "codebase_version": "v2.0",
        "robot_type": "UR5e",
        "total_episodes": total_episodes,
        "total_frames": total_video_frames, 
        "total_tasks": 1,
        "total_videos": total_episodes,
        "total_chunks": 1,
        "chunks_size": total_episodes,
        "fps": mean_trajectory_fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.side": {
                "dtype": "video",
                "shape": [height, width, channels],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": video_fps,
                    "video.codec": codec,
                    "pix_fmt": "yuv420p",  # This is still hardcoded as it's not easily accessible via OpenCV
                    "has_audio": False      # OpenCV doesn't expose audio info, but these are known to be video-only
                }
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": [
                    "PositionX", "PositionY", "PositionZ",
                    "OrientationX", "OrientationY", "OrientationZ",
                    "GripperAction"
                ]
            }
        }
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info_data, f, indent=2)

    # stats.json
    stats_data = {
        "action": {
            "q01": [-0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -1],
            "q99": [0.03, 0.03, 0.03, 0.03, 0.03, 0.03,  1]
        }
    }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats_data, f, indent=2)

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for eidx, frame_count in enumerate(video_frame_counts):
            row = {
                "episode_index": eidx,
                "tasks": ["Pick up the object"],
                "length": frame_count
            }
            f.write(json.dumps(row) + "\n")

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        row = {
            "task_index": 0,
            "task": "Demonstration from raw robot data"
        }
        f.write(json.dumps(row) + "\n")

def main(raw_data_prefix: Path = None, out_root: Path = None):
    print(f"\nStarting conversion with raw_data_prefix: {raw_data_prefix}")
    logging.debug("Starting the conversion process.")  # Logging line
    # Where is your raw data?
    if raw_data_prefix is None:
        raw_data_prefix = Path(".")
    raw_traj_dir = raw_data_prefix / "raw/trajectories" 
    raw_video_dir = raw_data_prefix / "raw/videos"
    print(f"Looking for data in:\n  {raw_traj_dir}\n  {raw_video_dir}")
    
    # Where do you want the new dataset to live?
    if out_root is None:
        out_root = raw_data_prefix / "data/my_lerobot_dataset"
    data_out_dir = out_root / "data" / "chunk-000"
    data_out_dir.mkdir(parents=True, exist_ok=True)

    # Pair up CSV + MP4
    pairs = find_pairs(raw_traj_dir, raw_video_dir)

    # Convert each episode
    episode_lengths = []
    for episode_index, (csv_f, mp4_f) in enumerate(pairs):
        logging.debug(f"Processing episode {episode_index} with CSV: {csv_f.name} and MP4: {mp4_f.name}")
        convert_single_episode(
            csv_file=csv_f,
            mp4_file=mp4_f,
            episode_index=episode_index,
            out_dir=data_out_dir
        )
        # Append the length of each episode
        episode_length = len(pd.read_csv(csv_f)) - 1  # Decrement by 1
        episode_lengths.append(episode_length)
        logging.debug(f"Episode {episode_index} length: {episode_length}")

    # Build meta files
    build_meta_files(out_root=out_root, total_episodes=len(pairs), episode_lengths=episode_lengths)
    print("\nDone creating LeRobot-style dataset at:", out_root)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_prefix", type=Path, default=None,
                       help="Path prefix to raw data directory")
    parser.add_argument("--out_root", type=Path, default=None,
                       help="Output directory for the dataset")
    args = parser.parse_args()
    print("Running with args:", args)
    main(raw_data_prefix=args.raw_data_prefix, out_root=args.out_root)