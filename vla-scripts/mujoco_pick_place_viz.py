"""
mujoco_pick_place_viz.py

MuJoCo FetchPickAndPlace task visualization with OpenVLA in the loop.

Task:
  Pick the cube from the table and place it at the goal marker.

Control:
  - A scripted stage controller outputs task-level target motion.
  - OpenVLA predicts 7D action from (image, language instruction).
  - Final control blends scripted motion with OpenVLA offsets.
  - Optional virtual grasp mode attaches the cube when grasp stage is reached
    to provide a stable and clear pick/place visualization loop.
"""

from __future__ import annotations

import argparse
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForVision2Seq, AutoProcessor


def get_openvla_prompt(instruction: str, openvla_path: str) -> str:
    if "v01" in openvla_path:
        system_prompt = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        return f"{system_prompt} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


class InstructionBody(BaseModel):
    instruction: str


@dataclass
class Config:
    openvla_path: str
    unnorm_key: str
    env_id: str
    hz: float
    blend_openvla: float
    episode_steps: int
    virtual_grasp: bool
    disable_openvla: bool
    do_sample: bool
    temperature: float
    top_p: float
    host: str
    port: int


class PickPlaceService:
    STAGE_NAMES = [
        "approach_above_object",
        "descend_to_grasp",
        "close_gripper",
        "lift_object",
        "move_to_goal",
        "descend_to_place",
        "release_object",
        "retreat",
    ]

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("CUDA is required for OpenVLA inference.")

        self._openvla_enabled = not cfg.disable_openvla
        self._openvla_init_error = ""
        self.processor = None
        self.vla = None
        if self._openvla_enabled:
            try:
                self.processor = AutoProcessor.from_pretrained(cfg.openvla_path, trust_remote_code=True)
                self.vla = AutoModelForVision2Seq.from_pretrained(
                    cfg.openvla_path,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ).to(self.device)
            except Exception as exc:  # noqa: BLE001
                self._openvla_enabled = False
                self._openvla_init_error = str(exc)

        self.env = None
        self.obs = None
        self.last_info = {}
        self.stage = 0
        self.stage_steps = 0

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._frame_jpeg = b""
        self._instruction = "pick up the cube and place it on the goal marker"
        self._last_openvla = np.zeros(7, dtype=np.float32)
        self._last_script = np.zeros(4, dtype=np.float32)
        self._last_env_action = np.zeros(4, dtype=np.float32)
        self._last_reward = 0.0
        self._is_success = 0.0
        self._step = 0
        self._episode = 0
        self._episode_step = 0
        self._frame_mean = 0.0
        self._frame_delta_mean = 0.0
        self._last_error = ""
        self._prev_frame: Optional[np.ndarray] = None

        self._attached = False
        self._attach_offset = np.array([0.0, 0.0, -0.40], dtype=np.float32)

    def _make_env(self):
        if hasattr(gymnasium_robotics, "register_envs"):
            gymnasium_robotics.register_envs(gym)
        return gym.make(self.cfg.env_id, render_mode="rgb_array")

    @staticmethod
    def _parse_fetch_obs(obs_dict: dict) -> dict:
        obs = obs_dict["observation"]
        desired_goal = obs_dict["desired_goal"]
        achieved_goal = obs_dict["achieved_goal"]
        grip_pos = obs[0:3]
        object_pos = obs[3:6]
        object_rel = obs[6:9]
        gripper_state = obs[9:11] if obs.shape[0] >= 11 else np.zeros(2, dtype=np.float32)
        return {
            "grip_pos": grip_pos,
            "object_pos": object_pos,
            "object_rel": object_rel,
            "gripper_state": gripper_state,
            "desired_goal": desired_goal,
            "achieved_goal": achieved_goal,
        }

    def _predict_openvla(self, frame_rgb: np.ndarray, instruction: str) -> np.ndarray:
        if not self._openvla_enabled or self.processor is None or self.vla is None:
            return np.zeros(7, dtype=np.float32)
        prompt = get_openvla_prompt(instruction, self.cfg.openvla_path)
        image = Image.fromarray(frame_rgb.astype(np.uint8)).convert("RGB")
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        action = self.vla.predict_action(
            **inputs,
            unnorm_key=self.cfg.unnorm_key,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )
        return np.asarray(action, dtype=np.float32)

    @staticmethod
    def _openvla_to_fetch_action(openvla_action: np.ndarray) -> np.ndarray:
        xyz = np.tanh(openvla_action[:3] * 30.0)
        grip = float(np.clip(openvla_action[6] * 2.0 - 1.0, -1.0, 1.0))
        return np.array([xyz[0], xyz[1], xyz[2], grip], dtype=np.float32)

    def _scripted_pick_place(self, parsed: dict, is_success: float) -> tuple[np.ndarray, float, np.ndarray]:
        grip = parsed["grip_pos"]
        obj = parsed["object_pos"]
        goal = parsed["desired_goal"]

        grasp = np.array([obj[0], obj[1], obj[2] + 0.40], dtype=np.float32)
        above_obj = np.array([obj[0], obj[1], obj[2] + 0.50], dtype=np.float32)
        carry = np.array([obj[0], obj[1], max(goal[2] + 0.53, obj[2] + 0.56)], dtype=np.float32)
        above_goal = np.array([goal[0], goal[1], goal[2] + 0.50], dtype=np.float32)
        place = np.array([goal[0], goal[1], goal[2] + 0.41], dtype=np.float32)
        retreat = np.array([goal[0], goal[1], goal[2] + 0.58], dtype=np.float32)

        dist_grip_above = float(np.linalg.norm(grip - above_obj))
        dist_grip_grasp = float(np.linalg.norm(grip - grasp))
        dist_obj_goal = float(np.linalg.norm(obj - goal))
        dist_obj_goal_xy = float(np.linalg.norm(obj[:2] - goal[:2]))

        if self.stage == 0 and dist_grip_above < 0.035:
            self.stage, self.stage_steps = 1, 0
        elif self.stage == 1 and dist_grip_grasp < 0.028:
            self.stage, self.stage_steps = 2, 0
        elif self.stage == 2 and self.stage_steps >= 10:
            self.stage, self.stage_steps = 3, 0
        elif self.stage == 3 and (self._attached or self.stage_steps >= 22):
            self.stage, self.stage_steps = 4, 0
        elif self.stage == 4 and (dist_obj_goal_xy < 0.05 or self.stage_steps >= 28):
            self.stage, self.stage_steps = 5, 0
        elif self.stage == 5 and (dist_obj_goal < 0.06 or self.stage_steps >= 24):
            self.stage, self.stage_steps = 6, 0
        elif self.stage == 6 and (is_success > 0.5 or self.stage_steps >= 10):
            self.stage, self.stage_steps = 7, 0
        elif self.stage == 7 and self.stage_steps >= 8:
            self.stage, self.stage_steps = 0, 0
        self.stage_steps += 1

        if self.stage == 0:
            target, grip_cmd = above_obj, 1.0
        elif self.stage == 1:
            target, grip_cmd = grasp, 1.0
        elif self.stage == 2:
            target, grip_cmd = grasp, -1.0
        elif self.stage == 3:
            target, grip_cmd = carry, -1.0
        elif self.stage == 4:
            target, grip_cmd = above_goal, -1.0
        elif self.stage == 5:
            target, grip_cmd = place, -1.0
        elif self.stage == 6:
            target, grip_cmd = place, 1.0
        else:
            target, grip_cmd = retreat, 1.0

        pos_cmd = np.clip((target - grip) * 10.0, -1.0, 1.0)
        script_action = np.array([pos_cmd[0], pos_cmd[1], pos_cmd[2], grip_cmd], dtype=np.float32)
        return target, float(grip_cmd), script_action

    def _apply_target_control(self, target: np.ndarray, grip_cmd: float) -> None:
        uw = self.env.unwrapped
        target_clip = np.clip(
            target.astype(np.float32),
            np.array([1.0, 0.30, 0.35], dtype=np.float32),
            np.array([1.70, 1.00, 1.00], dtype=np.float32),
        )
        action = np.array([0.0, 0.0, 0.0, float(np.clip(grip_cmd, -1.0, 1.0))], dtype=np.float32)
        uw._set_action(action)
        uw.data.mocap_pos[:] = target_clip.reshape(1, 3)
        uw._mujoco_step(action)
        uw._step_callback()

    def _maybe_virtual_grasp(self, parsed: dict, grip_cmd: float) -> None:
        if not self.cfg.virtual_grasp:
            return
        uw = self.env.unwrapped
        grip = parsed["grip_pos"]
        obj = parsed["object_pos"]

        dist_xy = float(np.linalg.norm(grip[:2] - obj[:2]))
        dist_z = float(abs((obj[2] + 0.40) - grip[2]))
        closing = grip_cmd < -0.2

        if (not self._attached) and self.stage in (2, 3) and closing and dist_xy < 0.03 and dist_z < 0.05:
            self._attached = True
            self._attach_offset = (obj - grip).astype(np.float32)
            self._attach_offset[2] = float(np.clip(self._attach_offset[2], -0.46, -0.32))

        if self._attached and self.stage >= 6:
            self._attached = False

        if self._attached:
            object_q = uw._utils.get_joint_qpos(uw.model, uw.data, "object0:joint").copy()
            desired_obj = (grip + self._attach_offset).astype(np.float32)
            desired_obj[2] = max(0.008, float(desired_obj[2]))
            object_q[:3] = desired_obj
            uw._utils.set_joint_qpos(uw.model, uw.data, "object0:joint", object_q)
            uw._utils.set_joint_qvel(uw.model, uw.data, "object0:joint", np.zeros(6, dtype=np.float32))
            uw._mujoco.mj_forward(uw.model, uw.data)

    @staticmethod
    def _encode_jpeg(frame: np.ndarray) -> bytes:
        img = Image.fromarray(frame.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def _compute_reward_success(self, obs: dict) -> tuple[float, float]:
        uw = self.env.unwrapped
        success = float(uw._is_success(obs["achieved_goal"], uw.goal))
        info = {"is_success": success}
        reward = float(uw.compute_reward(obs["achieved_goal"], uw.goal, info))
        return reward, success

    def _reset_episode(self, seed: Optional[int] = None) -> None:
        self.obs, self.last_info = self.env.reset(seed=seed)
        self.stage = 0
        self.stage_steps = 0
        self._episode_step = 0
        self._attached = False

    def _loop(self) -> None:
        self.env = self._make_env()
        self._reset_episode(seed=0)
        _ = self.env.render()

        period = 1.0 / self.cfg.hz
        while self._running:
            t0 = time.time()
            try:
                with self._lock:
                    instruction = self._instruction

                frame = self.env.render()
                if frame is None:
                    raise RuntimeError("Render returned None")

                parsed_before = self._parse_fetch_obs(self.obs)
                is_success_prev = float(self.last_info.get("is_success", 0.0))
                script_target, script_grip, script_action = self._scripted_pick_place(parsed_before, is_success_prev)
                openvla_action = self._predict_openvla(frame, instruction)
                openvla_fetch = self._openvla_to_fetch_action(openvla_action)

                blend = float(np.clip(self.cfg.blend_openvla, 0.0, 1.0))
                target = script_target + blend * openvla_fetch[:3] * 0.03
                grip_cmd = float(np.clip((1.0 - blend) * script_grip + blend * openvla_fetch[3], -1.0, 1.0))
                env_action = np.array(
                    [float(target[0]), float(target[1]), float(target[2]), grip_cmd],
                    dtype=np.float32,
                )

                self._apply_target_control(target, grip_cmd)
                self.obs = self.env.unwrapped._get_obs()
                parsed_after = self._parse_fetch_obs(self.obs)
                self._maybe_virtual_grasp(parsed_after, grip_cmd)
                self.obs = self.env.unwrapped._get_obs()
                reward, is_success = self._compute_reward_success(self.obs)
                self.last_info = {"is_success": is_success}

                self._episode_step += 1
                if self._episode_step >= self.cfg.episode_steps:
                    self._episode += 1
                    self._reset_episode(seed=None)

                disp = self.env.render()
                if disp is None:
                    disp = frame

                frame_mean = float(disp.mean())
                if self._prev_frame is None:
                    frame_delta = 0.0
                else:
                    frame_delta = float(np.mean(np.abs(disp.astype(np.float32) - self._prev_frame.astype(np.float32))))
                self._prev_frame = disp.copy()

                with self._lock:
                    self._frame_jpeg = self._encode_jpeg(disp)
                    self._last_openvla = openvla_action
                    self._last_script = script_action
                    self._last_env_action = env_action
                    self._last_reward = float(reward)
                    self._is_success = float(is_success)
                    self._frame_mean = frame_mean
                    self._frame_delta_mean = frame_delta
                    self._step += 1
                    self._last_error = ""
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._last_error = str(exc)
                time.sleep(0.5)
                continue

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.env is not None:
            self.env.close()

    def app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(_: FastAPI):
            self.start()
            yield
            self.stop()

        app = FastAPI(title="Fetch Pick&Place + OpenVLA", lifespan=lifespan)

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Fetch Pick&Place + OpenVLA</title>
    <style>
      body { font-family: Arial, sans-serif; background: #111; color: #eee; margin: 20px; }
      .card { background: #1b1b1b; border: 1px solid #333; border-radius: 10px; padding: 12px; }
      .row { display: flex; gap: 16px; align-items: flex-start; }
      input { width: 580px; background: #222; color: #fff; border: 1px solid #666; border-radius: 6px; padding: 8px; }
      button { margin-left: 8px; padding: 8px 12px; background: #2f2f2f; color: #fff; border: 1px solid #666; border-radius: 6px; cursor: pointer; }
      #frame { width: 640px; max-width: 92vw; border: 1px solid #333; border-radius: 8px; }
      pre { white-space: pre-wrap; margin: 0; }
      .task { margin: 8px 0 14px 0; color: #8ec5ff; }
    </style>
  </head>
  <body>
    <h2>MuJoCo 抓取/放置任务可视化（FetchPickAndPlace）</h2>
    <div class="task">任务: 抓起桌面方块并放到目标点（Pick & Place）。</div>
    <div class="card" style="margin-bottom:12px;">
      <input id="instruction" value="pick up the cube and place it on the goal marker" />
      <button onclick="setInstruction()">更新指令</button>
    </div>
    <div class="row">
      <div class="card"><img id="frame" src="/frame.jpg" /></div>
      <div class="card" style="min-width: 420px;"><pre id="state"></pre></div>
    </div>
    <script>
      async function setInstruction() {
        const instruction = document.getElementById('instruction').value;
        await fetch('/instruction', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({instruction})
        });
      }
      async function refresh() {
        document.getElementById('frame').src = '/frame.jpg?t=' + Date.now();
        const r = await fetch('/state');
        const j = await r.json();
        document.getElementById('state').textContent = JSON.stringify(j, null, 2);
      }
      setInterval(refresh, 300);
      refresh();
    </script>
  </body>
</html>"""

        @app.get("/frame.jpg")
        def frame() -> Response:
            with self._lock:
                data = self._frame_jpeg
            if not data:
                black = np.zeros((480, 480, 3), dtype=np.uint8)
                buf = io.BytesIO()
                Image.fromarray(black).save(buf, format="JPEG", quality=80)
                data = buf.getvalue()
            return Response(content=data, media_type="image/jpeg")

        @app.get("/state")
        def state() -> JSONResponse:
            with self._lock:
                payload = {
                    "task": "Pick cube and place on goal marker",
                    "env_id": self.cfg.env_id,
                    "step": self._step,
                    "episode": self._episode,
                    "episode_step": self._episode_step,
                    "instruction": self._instruction,
                    "stage": self.stage,
                    "stage_name": self.STAGE_NAMES[self.stage],
                    "virtual_grasp": self.cfg.virtual_grasp,
                    "attached": self._attached,
                    "is_success": self._is_success,
                    "reward": self._last_reward,
                    "openvla_action_7d": self._last_openvla.tolist(),
                    "script_action_4d": self._last_script.tolist(),
                    "control_target_xyz_grip": self._last_env_action.tolist(),
                    "blend_openvla": self.cfg.blend_openvla,
                    "openvla_enabled": self._openvla_enabled,
                    "openvla_init_error": self._openvla_init_error,
                    "frame_mean": self._frame_mean,
                    "frame_delta_mean": self._frame_delta_mean,
                    "do_sample": self.cfg.do_sample,
                    "error": self._last_error,
                }
            return JSONResponse(payload)

        @app.post("/instruction")
        def set_instruction(body: InstructionBody) -> JSONResponse:
            with self._lock:
                self._instruction = body.instruction.strip()
            return JSONResponse({"ok": True, "instruction": self._instruction})

        return app


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Fetch Pick&Place + OpenVLA visualizer")
    parser.add_argument("--openvla_path", type=str, required=True)
    parser.add_argument("--unnorm_key", type=str, default="bridge_orig")
    parser.add_argument("--env_id", type=str, default="FetchPickAndPlace-v3")
    parser.add_argument("--hz", type=float, default=3.0)
    parser.add_argument(
        "--blend_openvla",
        type=float,
        default=0.10,
        help="0 means fully scripted controller, 1 means fully OpenVLA-guided controller",
    )
    parser.add_argument("--episode_steps", type=int, default=220)
    parser.add_argument("--virtual_grasp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable_openvla", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8016)
    args = parser.parse_args()
    return Config(
        openvla_path=args.openvla_path,
        unnorm_key=args.unnorm_key,
        env_id=args.env_id,
        hz=args.hz,
        blend_openvla=args.blend_openvla,
        episode_steps=args.episode_steps,
        virtual_grasp=args.virtual_grasp,
        disable_openvla=args.disable_openvla,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        host=args.host,
        port=args.port,
    )


def main() -> None:
    cfg = parse_args()
    service = PickPlaceService(cfg)
    uvicorn.run(service.app(), host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
