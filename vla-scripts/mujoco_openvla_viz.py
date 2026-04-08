"""
mujoco_openvla_viz.py

MuJoCo + OpenVLA closed-loop simulation web visualizer.

Pipeline:
  RGB frame (MuJoCo env) -> OpenVLA action (7D) -> mapped env action -> env.step()

This script is intended for algorithm validation before real robot deployment.
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

# Use EGL offscreen rendering by default to avoid GUI-context crashes in server mode.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym
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
    attn_implementation: str
    hz: float
    ctrl_scale: float
    do_sample: bool
    temperature: float
    top_p: float
    host: str
    port: int


class MujocoOpenVLAService:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("CUDA is required for OpenVLA inference.")

        self.processor = AutoProcessor.from_pretrained(cfg.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            cfg.openvla_path,
            attn_implementation=cfg.attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # NOTE: Create MuJoCo env in the worker thread to avoid GL-context issues
        # (black frames when context is created in one thread and rendered in another).
        self.env = None
        self.obs = None

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._frame_jpeg = b""
        self._instruction = "move end-effector towards the target"
        self._last_openvla_action = np.zeros(7, dtype=np.float32)
        self._last_env_action = np.zeros(1, dtype=np.float32)
        self._last_reward = 0.0
        self._step = 0
        self._episode = 0
        self._last_error = ""
        self._prev_openvla_action: Optional[np.ndarray] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._frame_min = 0
        self._frame_max = 0
        self._frame_mean = 0.0
        self._frame_std = 0.0
        self._frame_delta_mean = 0.0
        self._action_delta_l2 = 0.0

    def _make_env(self, env_id: str):
        tried = []
        for name in [env_id, "Reacher-v4", "HalfCheetah-v4"]:
            if name in tried:
                continue
            tried.append(name)
            try:
                return gym.make(name, render_mode="rgb_array")
            except Exception:
                continue
        raise RuntimeError(f"Failed to create MuJoCo env. Tried: {tried}")

    def _predict_openvla(self, frame_rgb: np.ndarray, instruction: str) -> np.ndarray:
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

    def _map_to_env_action(self, openvla_action: np.ndarray) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("Environment is not initialized yet.")
        # Map 7D OpenVLA action to arbitrary MuJoCo action dims.
        target_dim = int(np.prod(self.env.action_space.shape))
        src = openvla_action.reshape(-1)
        if target_dim <= src.shape[0]:
            raw = src[:target_dim]
        else:
            reps = int(np.ceil(target_dim / src.shape[0]))
            raw = np.tile(src, reps)[:target_dim]

        # Normalize and scale for stable control.
        normalized = np.tanh(raw) * self.cfg.ctrl_scale
        low = self.env.action_space.low.reshape(-1)
        high = self.env.action_space.high.reshape(-1)
        if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
            mid = 0.5 * (low + high)
            half = 0.5 * (high - low)
            out = mid + np.clip(normalized, -1.0, 1.0) * half
        else:
            out = normalized
        return out.astype(np.float32)

    def _encode_jpeg(self, frame_rgb: np.ndarray) -> bytes:
        img = Image.fromarray(frame_rgb.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def _loop(self) -> None:
        self.env = self._make_env(self.cfg.env_id)
        self.obs, _ = self.env.reset(seed=0)
        _ = self.env.render()  # warmup renderer
        with self._lock:
            self._last_env_action = np.zeros(int(np.prod(self.env.action_space.shape)), dtype=np.float32)

        period = 1.0 / self.cfg.hz
        while self._running:
            t0 = time.time()
            try:
                with self._lock:
                    instruction = self._instruction

                infer_frame = self.env.render()
                if infer_frame is None:
                    raise RuntimeError("MuJoCo render() returned None.")

                openvla_action = self._predict_openvla(infer_frame, instruction)
                env_action = self._map_to_env_action(openvla_action)
                self.obs, reward, terminated, truncated, _ = self.env.step(env_action)
                if terminated or truncated:
                    self.obs, _ = self.env.reset()
                    self._episode += 1
                display_frame = self.env.render()
                if display_frame is None:
                    display_frame = infer_frame

                # Runtime diagnostics for "black screen" / "static frame" debugging.
                frame_min = int(display_frame.min())
                frame_max = int(display_frame.max())
                frame_mean = float(display_frame.mean())
                frame_std = float(display_frame.std())
                if self._prev_frame is None:
                    frame_delta_mean = 0.0
                else:
                    frame_delta_mean = float(
                        np.mean(
                            np.abs(display_frame.astype(np.float32) - self._prev_frame.astype(np.float32))
                        )
                    )
                if self._prev_openvla_action is None:
                    action_delta_l2 = 0.0
                else:
                    action_delta_l2 = float(np.linalg.norm(openvla_action - self._prev_openvla_action))

                self._prev_frame = display_frame.copy()
                self._prev_openvla_action = openvla_action.copy()

                with self._lock:
                    self._frame_jpeg = self._encode_jpeg(display_frame)
                    self._last_openvla_action = openvla_action
                    self._last_env_action = env_action
                    self._last_reward = float(reward)
                    self._frame_min = frame_min
                    self._frame_max = frame_max
                    self._frame_mean = frame_mean
                    self._frame_std = frame_std
                    self._frame_delta_mean = frame_delta_mean
                    self._action_delta_l2 = action_delta_l2
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

    def build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(_: FastAPI):
            self.start()
            yield
            self.stop()

        app = FastAPI(title="MuJoCo OpenVLA Visualizer", lifespan=lifespan)

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>MuJoCo + OpenVLA</title>
    <style>
      body { font-family: Arial, sans-serif; background: #111; color: #eee; margin: 20px; }
      .card { background: #1b1b1b; border: 1px solid #333; border-radius: 10px; padding: 12px; }
      .row { display: flex; gap: 16px; align-items: flex-start; }
      input { width: 440px; background: #222; color: #fff; border: 1px solid #666; border-radius: 6px; padding: 8px; }
      button { margin-left: 8px; padding: 8px 12px; background: #2f2f2f; color: #fff; border: 1px solid #666; border-radius: 6px; cursor: pointer; }
      #frame { width: 640px; max-width: 92vw; border: 1px solid #333; border-radius: 8px; }
      pre { white-space: pre-wrap; margin: 0; }
    </style>
  </head>
  <body>
    <h2>MuJoCo + OpenVLA 仿真闭环</h2>
    <div class="card" style="margin-bottom:12px;">
      <input id="instruction" value="move end-effector towards the target" />
      <button onclick="setInstruction()">更新指令</button>
    </div>
    <div class="row">
      <div class="card"><img id="frame" src="/frame.jpg" /></div>
      <div class="card" style="min-width: 360px;"><pre id="state"></pre></div>
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
                    "env_id": self.cfg.env_id,
                    "step": self._step,
                    "episode": self._episode,
                    "instruction": self._instruction,
                    "openvla_action": self._last_openvla_action.tolist(),
                    "env_action": self._last_env_action.tolist(),
                    "last_reward": self._last_reward,
                    "frame_min": self._frame_min,
                    "frame_max": self._frame_max,
                    "frame_mean": self._frame_mean,
                    "frame_std": self._frame_std,
                    "frame_delta_mean": self._frame_delta_mean,
                    "action_delta_l2": self._action_delta_l2,
                    "ctrl_scale": self.cfg.ctrl_scale,
                    "do_sample": self.cfg.do_sample,
                    "error": self._last_error,
                    "device": str(self.device),
                    "unnorm_key": self.cfg.unnorm_key,
                }
            return JSONResponse(payload)

        @app.post("/instruction")
        def set_instruction(body: InstructionBody) -> JSONResponse:
            with self._lock:
                self._instruction = body.instruction.strip()
            return JSONResponse({"ok": True, "instruction": self._instruction})

        return app


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="MuJoCo + OpenVLA web visualizer")
    parser.add_argument("--openvla_path", type=str, required=True, help="HF path or local model dir")
    parser.add_argument("--unnorm_key", type=str, default="bridge_orig")
    parser.add_argument("--env_id", type=str, default="Reacher-v4")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--hz", type=float, default=2.0, help="Closed-loop frequency")
    parser.add_argument("--ctrl_scale", type=float, default=30.0, help="OpenVLA action scaling before env mapping")
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stochastic decoding to avoid static actions on nearly-static frames.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()
    return Config(
        openvla_path=args.openvla_path,
        unnorm_key=args.unnorm_key,
        env_id=args.env_id,
        attn_implementation=args.attn_implementation,
        hz=args.hz,
        ctrl_scale=args.ctrl_scale,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        host=args.host,
        port=args.port,
    )


def main() -> None:
    cfg = parse_args()
    service = MujocoOpenVLAService(cfg)
    app = service.build_app()
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
