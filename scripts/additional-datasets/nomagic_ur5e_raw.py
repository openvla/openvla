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

import exiftool
import os
import re
import json
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
from typing import Optional

from pathlib import Path
from typing import List, Dict
import torch
import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

from datasets import Dataset
from lerobot.common.datasets.utils import (
    check_timestamps_sync,
    calculate_episode_data_index
)

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
class LeRobotTrajectoryStep:
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


def pair_trajectories_and_videos(
        csv_trajectories_dir: Path,  
        mp4_videos_dir: Path  
) -> list[tuple[Path, Path]]:
    """
    Pair up CSV trajectories and MP4 videos.
    Uses alphanumeric order on filenames.
    """ 

    csv_trajectories = sorted(csv_trajectories_dir.glob("*.csv"))
    mp4_videos = sorted(mp4_videos_dir.glob("*.mp4"))

    logging.debug(f"Found CSV files: {[f.name for f in csv_trajectories]}")
    logging.debug(f"Found MP4 files: {[f.name for f in mp4_videos]}")

    # We pair them up in order since they're already sorted chronologically
    # TODO: compare actual timestamps in filename, not alphanumeric order

    # We pair them up alphanumerically
    trajectory_video_pairs = list(zip(csv_trajectories, mp4_videos))
    
    for csv_trajectory, mp4_video in trajectory_video_pairs:
        logging.debug(f"Paired {csv_trajectory.name} with {mp4_video.name}")

    return trajectory_video_pairs

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
    
    # Calculate orientation difference
    r1 = R.from_rotvec(
        [
            rowA["OrientationX"],
            rowA["OrientationY"],
            rowA["OrientationZ"]
        ]
    )
    r2 = R.from_rotvec(
        [
            rowB["OrientationX"],
            rowB["OrientationY"],
            rowB["OrientationZ"]
        ]
    )
    r_diff = r2 * r1.inv()
    euler_diff = r_diff.as_euler("xyz")
    dox, doy, doz = euler_diff
    
    # Gripper - map Gripper::Action values to float
    grip_val = rowA["GripperAction"]
    
    return [dx, dy, dz, dox, doy, doz, grip_val]

def read_frame_list_from_path(
        video_path: Path | str
) -> list[cv2.typing.MatLike]:
    """
    Given a path to an MP4 file, read all frames from the file into a list.
    """
    cap = cv2.VideoCapture(str(video_path))
    frame_list = list()
    while (ret := cap.read())[0]:  # ret[0] is success flag, ret[1] is the frame
        frame_list.append(ret[1])
    return frame_list

def make_video_timestamps(
        mp4_filepath: Path | str, 
        frame_list: list[cv2.typing.MatLike],
) -> list[int]:
    """
    Given a path to an MP4 file and a list of frames from that file,
    use ExifTool to read a timestamp of as many frames as possible from the
    video metadata, then interpolate evenly to remaining frames.
    """
    # Read timestamps from metadata. Result is given in Unix nanoseconds.
    with exiftool.ExifToolHelper() as et:
        mp4_metadata = et.get_metadata(str(mp4_filepath))
        timestamp_list: list[int] = [
            int(t)  # From str
            for t in mp4_metadata[0]["XMP:Timestamps"]
        ]
    
    # Emit a warning if not all frames are retrieved from the metadata.
    if len(timestamp_list) < len(frame_list):
        logging.warning(
            f"ExifTool found {len(timestamp_list)} timestamps "
            f"for {len(frame_list)} frames of video at {mp4_filepath}. "
            "Will append interpolated timestamps to match frame count."    
        )

    # Interpolate timestamps.
    fps = mp4_metadata[0]["QuickTime:VideoFrameRate"]
    delta_t = int((1 / fps) * 1e9)  # In nanoseconds
    while len(timestamp_list) < len(frame_list):
        timestamp_list.append(timestamp_list[-1] + delta_t)

    # Convert timestamps to Unix miliseconds.
    timestamp_list = [
        int(t / 1e6)  # In miliseconds
        for t in timestamp_list
    ]

    # Check max deviation of frame timestamp deltas from period.
    timestamp_array = np.array(timestamp_list, dtype=np.float64) / 1e3
    diffs = np.diff(timestamp_array)
    logging.debug(
        "Max deviation of frame timestamp deltas from period "
        f"set by {fps=} is {np.max(np.abs(diffs - (1/fps))):.6f}s"
    )

    return timestamp_list

def csv_to_lerobot_trajectory(
    csv_trajectory_filepath: Path,
    mp4_filepath: Path,
    maybe_lerobot_episode_index: int,
    tolerance_s: float = 1e-5,
) -> tuple[pa.Table, np.ndarray]:
    """
    Convert one CSV + MP4 into a single "episode_{:06d}.parquet" and
    copy the MP4 to "episode_{:06d}.mp4" in observation.images.side subdir.

    Also save a video containing only those frames that were matched.
    """

    # Load CSV data.
    csv_trajectory_df = pd.read_csv(csv_trajectory_filepath)

    # --- Preprocess the trajectory data. ---
    # Drop non-synchronized rows entirely. "Synchronized" is a boolean column
    # intended to show if ViperLink was connected to UR5e at timestamp.
    sync_mask = csv_trajectory_df["IsSynchronized"] == 1
    csv_trajectory_df = csv_trajectory_df[sync_mask]

    # Append binary grip action to each row.
    # In ViperLink, `Gripper::Action::NONE` has the effect of taking previous
    # action (or `RELEASE` at trajectory start). We preserve this behavior.
    current_grip_action = -1.0  # Initial action is -1.0 (RELEASE)
    for idx, row in csv_trajectory_df.iterrows():
        if row["GripperAction"] == "Gripper::Action::RELEASE":
            current_grip_action = -1.0
        elif row["GripperAction"] == "Gripper::Action::GRAB":
            current_grip_action = 1.0
        csv_trajectory_df.at[idx, "GripperAction"] = current_grip_action
        
    # Append timestamp in Unix miliseconds to each CSV row.
    csv_trajectory_df["miliseconds"] = (
        pd
        .to_datetime(csv_trajectory_df["Timestamp"])
        .add(pd.Timedelta(hours=-1))  # UTC-1
        .apply(lambda x: x.timestamp() * 1e3)  # Miliseconds
        .astype(int)
    )

    # --- Preprocess the video. ---
    # Read video timestamp.
    mp4_frame_list = read_frame_list_from_path(mp4_filepath)
    mp4_timestamp_list: list[int] = make_video_timestamps(
        mp4_filepath=mp4_filepath,  # Should be seconds
        frame_list=mp4_frame_list
    )

    # --- Match trajectory steps to video frames. ---
    row_ts_begin = csv_trajectory_df["miliseconds"].min()
    row_ts_end = csv_trajectory_df["miliseconds"].max()
    matched_rows_by_frame: list[Optional[pd.Series]] \
         = [None] * len(mp4_timestamp_list)
    tolerance_unix_ms = tolerance_s * 1e3
    for frame_idx, frame_ts in enumerate(mp4_timestamp_list):
        if (
            frame_ts < row_ts_begin - tolerance_unix_ms or
            frame_ts > row_ts_end + tolerance_unix_ms
        ):
            continue  # Unmatchable within tolerance

        # Find a match by binary search on the trajectory timestamps.
        try:
            matching_row = csv_trajectory_df.iloc[
                csv_trajectory_df["miliseconds"].searchsorted(frame_ts)
            ]
        except IndexError:  # Likely means frame is outside trajectory,
            continue        # but within tolerance. Ignore it for now.

        # If the match is further than the tolerance, don't include it.
        if (
            frame_ts < matching_row["miliseconds"] - tolerance_unix_ms or
            frame_ts > matching_row["miliseconds"] + tolerance_unix_ms
        ):
            logging.warning(
                f"Matching candidate timestep further "
                f"({np.abs(frame_ts - matching_row['miliseconds']):.8f}ms) from frame than "
                f"allowed by tolerance of {tolerance_unix_ms}ms. Won't match this frame."
            )
            continue  # Unmatchable within tolerance

        # Save the match.
        matched_rows_by_frame[frame_idx] = matching_row
    
    # Verify that matched rows form a single, continuous trajectory.
    switch_count = int(matched_rows_by_frame[0] != None)
    for row_idx, row in enumerate(matched_rows_by_frame[:-1]):
        next_row = matched_rows_by_frame[row_idx + 1]
        if type(row) != type(next_row):
            switch_count += 1
    if switch_count > 2:
        raise ValueError(
            "Trajectory data forms multiple trajectories when "
            "matched against video."
        )
    
    # --- Build the trajectory. ---
    trajectory = [
        (row_frame_pair_idx, row)
        for row_frame_pair_idx, row
        in enumerate(matched_rows_by_frame)
        if row is not None  # If frame[frame_idx] matches some row
    ]
    # This is the timestamp of the initial frame in the trajectory.
    # It is (brittly) guaranteed to be >=0, since we match steps to
    # frames by binsearch on step timestamps, and skip index errors.
    # Since the timestep of the first frame is zeroed out by LeRobotDataset
    # initializer, we take it as the zero-point relative to step timestamps,
    # i.e. saved timestamp is `timestamp_isn_unix_ms - trajectory_ts_begin`.
    trajectory_init_frame_idx = trajectory[0][0]
    trajectory_ts_begin = mp4_timestamp_list[trajectory_init_frame_idx]
    lerobot_trajectory: List[LeRobotTrajectoryStep] = [
        LeRobotTrajectoryStep(                                        
            timestamp=float(row["miliseconds"] - trajectory_ts_begin) / 1e3, 
            episode_index=maybe_lerobot_episode_index,                # ^ seconds
            next_done=next_row is None or next_row_idx == len(trajectory) - 1,  # TODO: Not ideal
            task_index=0,  # TODO: Assumes (incorrectly) only single task in data
            index=step_idx,
            frame_index=step_idx,
            action=compute_actions_from_rows(row, next_row)
        )
        for step_idx, ((_, row), (next_row_idx, next_row))
        in enumerate(zip(trajectory[:-1], trajectory[1:]))
    ]
    matched_video = np.array([
        mp4_frame_list[frame_idx]
        for frame_idx, _ in trajectory[:-1]
    ], dtype=np.uint8)

    # Write trajectory out as a parquet.
    pa_trajectory = pa.Table.from_pydict({
        "timestamp": [f.timestamp for f in lerobot_trajectory],
        "episode_index": [f.episode_index for f in lerobot_trajectory],
        "next.done": [f.next_done for f in lerobot_trajectory],
        "task_index": [f.task_index for f in lerobot_trajectory],
        "index": [f.index for f in lerobot_trajectory],
        "frame_index": [f.frame_index for f in lerobot_trajectory],
        "action": pa.array([f.action for f in lerobot_trajectory], type=pa.list_(pa.float32()))
    })

    return pa_trajectory, matched_video

def save_single_trajectory(
        trajectory: pa.Table,
        out_dir: Path,
        episode_index: int,
) -> None:
    """
    Save a single trajectory to a parquet file at
    out_dir / data / chunk-000 / f"episode_{episode_index:06d}.parquet"
    """
    trajectory_outdir = out_dir \
        / "data" \
        / "chunk-000" \
        / f"episode_{episode_index:06d}.parquet"
    pq.write_table(trajectory, trajectory_outdir)
    logging.debug(
        f"[Episode {episode_index}] Saved parquet =>  {trajectory_outdir}"
    )

def save_single_video(
        video: np.ndarray,
        out_dir: Path,
        episode_index: int,
        fps: float,
) -> Path:
    """
    Save a single video, given a numpy array of frames (dtype=uint8),
    to the expected output location at
    out_dir / "videos" / "chunk-000" / "observation.images.side"
    / f"episode_{episode_index:06d}.mp4"
    """
    video_outdir = out_dir \
        / "videos" \
        / "chunk-000" \
        / "observation.images.side"
    video_outdir.mkdir(parents=True, exist_ok=True)
    episode_mp4 = video_outdir / f"episode_{episode_index:06d}.mp4"

    if len(video) == 0:
        logging.debug(f"[Episode {episode_index}] No frames to save for MP4 => {episode_mp4}")
        return

    # Assume video shape is (num_frames, height, width, channels)
    height, width, channels = video[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(episode_mp4), fourcc, fps, (width, height))

    for frame in video:
        writer.write(frame)

    writer.release()
    logging.debug(f"[Episode {episode_index}] Saved MP4 => {episode_mp4}")
    return episode_mp4

def build_meta_files(out_root: Path, total_episodes: int, episode_lengths: List[int]):
    meta_dir = out_root / "meta"
    meta_dir.mkdir(exist_ok=True)

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
        "fps": video_fps,
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
    # Collect all actions across episodes into a list
    all_actions = []
    for episode_idx in range(total_episodes):
        episode_path = out_root \
            / "data" \
            / "chunk-000" \
            / f"episode_{episode_idx:06d}.parquet"
        table = pq.read_table(episode_path)
        actions = table["action"].to_numpy()
        all_actions.extend(actions)
    
    # Convert to numpy array and compute quantiles along first axis
    all_actions = np.array(all_actions)
    q01 = np.quantile(all_actions, 0.01, axis=0).tolist()
    q99 = np.quantile(all_actions, 0.99, axis=0).tolist()
    
    stats_data = {
        "action": {
            "q01": q01,
            "q99": q99
        }
    }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats_data, f, indent=2)

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for eidx, length in enumerate(episode_lengths):
            row = {
                "episode_index": eidx,
                "tasks": ["Pick up the object"],
                "length": length
            }
            f.write(json.dumps(row) + "\n")

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        row = {
            "task_index": 0,
            "task": "Demonstration from raw robot data"
        }
        f.write(json.dumps(row) + "\n")

def main(
        raw_data_prefix: Path = None,
        out_root: Path = None,
        tolerance_s: float = 1e-5
):
    
    # Where is your raw data?
    if raw_data_prefix is None:
        raw_data_prefix = Path(".")
    raw_traj_dir = raw_data_prefix / "trajectories" 
    print(f"Looking for data in:\n  {raw_traj_dir}")
    
    # Where do you want the new dataset to live?
    if out_root is None:
        raise ValueError("give a outfile location")
    data_out_dir = out_root / "data" / "chunk-000"
    data_out_dir.mkdir(parents=True, exist_ok=True)

    # Pair up CSV + MP4
    pairs: list[tuple[Path, Path]] = []
    for ep_idx in os.listdir(raw_traj_dir):
        ep_files = os.listdir(raw_traj_dir / ep_idx)
        csv_file = [
            ep_file 
            for ep_file in ep_files
            if ep_file.endswith(".csv")
        ][0]
        mp4_file = [
            ep_file 
            for ep_file in ep_files
            if ("side_view" in ep_file) and ep_file.endswith(".mp4")
        ][0]
        pairs.append((raw_traj_dir / ep_idx / csv_file, raw_traj_dir / ep_idx / mp4_file))

    # Convert each episode
    lerobot_episode_lengths = []
    lerobot_episode_index = 0
    for episode_index, (csv_f, mp4_f) in enumerate(pairs):
        print("\n")
        logging.debug((
            f"Processing episode {episode_index} "
            f"with CSV: {csv_f.name} and MP4: {mp4_f.name}"
        ))
        lerobot_trajectory, matched_video = csv_to_lerobot_trajectory(
            csv_trajectory_filepath=csv_f,
            mp4_filepath=mp4_f,
            maybe_lerobot_episode_index=lerobot_episode_index,
            tolerance_s=tolerance_s,
        )

        # Check that the trajectory satisfies the tolerance.
        timestamps = lerobot_trajectory["timestamp"].to_numpy()
        diffs = np.diff(timestamps)
        fps = cv2.VideoCapture(mp4_f).get(cv2.CAP_PROP_FPS)
        within_tolerance = torch.tensor(
            np.abs(diffs - 1/fps) <= tolerance_s
        )
        if not torch.all(within_tolerance):
            # Find indices where tolerance check failed
            failed_indices = torch.where(~within_tolerance)[0]
            failed_diffs = diffs[failed_indices]
            expected_interval = 1/fps
            
            logging.debug(
                f"Episode {episode_index} failed tolerance check and will not be included in the LeRobot dataset.\n"
                f"Found {len(failed_indices)} timestamp intervals outside tolerance of {tolerance_s}s:\n"
                f"- Expected interval between frames: {expected_interval:.6f}s\n"
                f"- Number of failed indices: {failed_indices.shape[0]}\n"
                f"- Maximum deviation from expected: {np.max(np.abs(failed_diffs - expected_interval)):.6f}s\n"
                f"- Maximum allowed deviation: ±{tolerance_s:.6f}s"
            )
            continue

        # Save the trajectory and video
        save_single_trajectory(
            lerobot_trajectory,
            out_root,
            lerobot_episode_index
        )
        video_path = save_single_video(
            matched_video,
            out_root,
            lerobot_episode_index,
            fps=cv2.VideoCapture(mp4_f).get(cv2.CAP_PROP_FPS)
        )

        # Save the length of the trajectory.
        lerobot_episode_length = len(lerobot_trajectory)
        lerobot_episode_lengths.append(lerobot_episode_length)
        lerobot_frame_count = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)
        logging.debug((
            f"Episode {episode_index} "
            f"will be saved in LeRobot dataset as episode {lerobot_episode_index}.\n "
            f"trajectory length: {lerobot_episode_length} "
            f"number of frames: {int(lerobot_frame_count)}"
        ))

        lerobot_episode_index += 1

    lerobot_episodes_total = len(lerobot_episode_lengths)

    # Build meta files
    build_meta_files(
        out_root=out_root,
        total_episodes=lerobot_episodes_total,
        episode_lengths=lerobot_episode_lengths
    )
    print("\nDone creating LeRobot-style dataset at:", out_root)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_prefix", type=Path, default=None,
                       help="Path prefix to raw data directory")
    parser.add_argument("--out_root", type=Path, default=None,
                       help="Output directory for the dataset")
    parser.add_argument("--tolerance_s", type=float, default=1e-5,
                       help=("Maximum allowed deviation from expected "
                             "frame interval (in seconds)"))
    args = parser.parse_args()
    print("Running with args:", args)

    main(
        raw_data_prefix=args.raw_data_prefix,
        out_root=args.out_root,
        tolerance_s=args.tolerance_s
    )