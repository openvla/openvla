"""
sim_web_viz.py

OpenVLA + PyBullet minimal simulation web visualizer.

Features:
- Runs a simple PyBullet scene (KUKA iiwa + cube) in DIRECT mode.
- Renders an RGB camera frame from simulation.
- Uses OpenVLA to infer a 7D action from (instruction, image).
- Applies translational/rotational delta action to the robot via IK.
- Serves a lightweight web page for visualization and instruction editing.

Usage:
  conda run --no-capture-output -n openvla python vla-scripts/sim_web_viz.py \
    --openvla_path /path/to/openvla-7b \
    --host 127.0.0.1 \
    --port 8010
"""

from __future__ import annotations

import argparse
import io
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data
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


@dataclass
class SimConfig:
    openvla_path: str
    unnorm_key: str
    attn_implementation: str
    hz: float
    width: int
    height: int
    pos_scale: float
    rot_scale: float
    host: str
    port: int


class InstructionBody(BaseModel):
    instruction: str


class OpenVLASimServer:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("CUDA is required for real-time OpenVLA inference.")

        self.processor = AutoProcessor.from_pretrained(cfg.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            cfg.openvla_path,
            attn_implementation=cfg.attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._frame_jpeg: bytes = b""
        self._step = 0
        self._instruction = "pick up the red cube"
        self._last_action = np.zeros(7, dtype=np.float32)
        self._last_error = ""

        self._sim_init()

    def _sim_init(self) -> None:
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", [0.5, 0, -0.65], useFixedBase=True)
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        self.ee_link = 6

        # Simple target cube
        self.cube = p.loadURDF("cube_small.urdf", [0.65, 0.0, 0.05], useFixedBase=False)

        # Camera setup
        self.view = p.computeViewMatrix(
            cameraEyePosition=[1.1, 0.0, 0.7],
            cameraTargetPosition=[0.45, 0.0, 0.1],
            cameraUpVector=[0, 0, 1],
        )
        self.proj = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=float(self.cfg.width) / float(self.cfg.height),
            nearVal=0.02,
            farVal=3.0,
        )

    def _render_rgb(self) -> np.ndarray:
        _, _, rgba, _, _ = p.getCameraImage(
            width=self.cfg.width,
            height=self.cfg.height,
            viewMatrix=self.view,
            projectionMatrix=self.proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgba = np.reshape(rgba, (self.cfg.height, self.cfg.width, 4))
        rgb = rgba[:, :, :3].astype(np.uint8)
        return rgb

    def _predict_action(self, rgb: np.ndarray, instruction: str) -> np.ndarray:
        prompt = get_openvla_prompt(instruction, self.cfg.openvla_path)
        pil_img = Image.fromarray(rgb).convert("RGB")
        inputs = self.processor(prompt, pil_img).to(self.device, dtype=torch.bfloat16)
        action = self.vla.predict_action(
            **inputs,
            unnorm_key=self.cfg.unnorm_key,
            do_sample=False,
        )
        return np.asarray(action, dtype=np.float32)

    def _apply_action(self, action: np.ndarray) -> None:
        dx, dy, dz, droll, dpitch, dyaw, _ = action.tolist()
        cur_pos, cur_orn = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)[4:6]
        cur_euler = p.getEulerFromQuaternion(cur_orn)

        tgt_pos = [
            cur_pos[0] + self.cfg.pos_scale * dx,
            cur_pos[1] + self.cfg.pos_scale * dy,
            max(0.02, cur_pos[2] + self.cfg.pos_scale * dz),
        ]
        tgt_euler = [
            cur_euler[0] + self.cfg.rot_scale * droll,
            cur_euler[1] + self.cfg.rot_scale * dpitch,
            cur_euler[2] + self.cfg.rot_scale * dyaw,
        ]
        tgt_orn = p.getQuaternionFromEuler(tgt_euler)

        joint_targets = p.calculateInverseKinematics(self.robot, self.ee_link, tgt_pos, tgt_orn)
        for j in range(7):
            p.setJointMotorControl2(
                self.robot,
                j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_targets[j],
                force=200.0,
            )

    def _encode_frame(self, rgb: np.ndarray, action: np.ndarray, instruction: str) -> bytes:
        # Add a small info strip for easier page-side debugging.
        info_h = 44
        canvas = np.zeros((rgb.shape[0] + info_h, rgb.shape[1], 3), dtype=np.uint8)
        canvas[: rgb.shape[0]] = rgb
        canvas[rgb.shape[0] :] = np.array([20, 20, 20], dtype=np.uint8)
        img = Image.fromarray(canvas)
        # Keep frame raw and send details as JSON; front-end overlays text.
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def _loop(self) -> None:
        dt = 1.0 / self.cfg.hz
        while self._running:
            t0 = time.time()
            try:
                with self._lock:
                    instruction = self._instruction

                rgb = self._render_rgb()
                action = self._predict_action(rgb, instruction)
                self._apply_action(action)
                for _ in range(4):
                    p.stepSimulation()

                frame = self._encode_frame(rgb, action, instruction)
                with self._lock:
                    self._frame_jpeg = frame
                    self._last_action = action
                    self._step += 1
                    self._last_error = ""
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._last_error = str(exc)
                # Avoid hot looping if inference fails
                time.sleep(0.5)
                continue

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

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
        try:
            p.disconnect(self.cid)
        except Exception:  # noqa: BLE001
            pass

    def app(self) -> FastAPI:
        app = FastAPI(title="OpenVLA Simulation Web Visualizer")

        @app.on_event("startup")
        def _startup() -> None:
            self.start()

        @app.on_event("shutdown")
        def _shutdown() -> None:
            self.stop()

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>OpenVLA Simulation Visualizer</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; background: #111; color: #eee; }
      .row { display: flex; gap: 16px; align-items: flex-start; }
      .card { background: #1b1b1b; padding: 12px; border-radius: 10px; border: 1px solid #333; }
      input { width: 360px; padding: 8px; border-radius: 6px; border: 1px solid #666; background: #222; color: #fff; }
      button { margin-left: 8px; padding: 8px 12px; border-radius: 6px; border: 1px solid #666; background: #2f2f2f; color: #fff; cursor: pointer; }
      #frame { width: 768px; max-width: 90vw; border-radius: 8px; border: 1px solid #333; }
      pre { white-space: pre-wrap; word-break: break-word; margin: 0; }
    </style>
  </head>
  <body>
    <h2>OpenVLA 仿真可视化</h2>
    <div class="card" style="margin-bottom: 12px;">
      <input id="instruction" value="pick up the red cube" />
      <button onclick="setInstruction()">更新指令</button>
    </div>
    <div class="row">
      <div class="card">
        <img id="frame" src="/frame.jpg" />
      </div>
      <div class="card" style="min-width: 320px;">
        <h3 style="margin-top:0;">状态</h3>
        <pre id="state"></pre>
      </div>
    </div>
    <script>
      async function setInstruction() {
        const instruction = document.getElementById('instruction').value;
        await fetch('/instruction', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ instruction }),
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
                frame = self._frame_jpeg
            if not frame:
                black = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
                buf = io.BytesIO()
                Image.fromarray(black).save(buf, format="JPEG", quality=80)
                frame = buf.getvalue()
            return Response(content=frame, media_type="image/jpeg")

        @app.get("/state")
        def state() -> JSONResponse:
            with self._lock:
                payload = {
                    "step": self._step,
                    "instruction": self._instruction,
                    "action": self._last_action.tolist(),
                    "error": self._last_error,
                    "device": str(self.device),
                    "unnorm_key": self.cfg.unnorm_key,
                }
            return JSONResponse(payload)

        @app.post("/instruction")
        def update_instruction(body: InstructionBody) -> JSONResponse:
            with self._lock:
                self._instruction = body.instruction.strip()
            return JSONResponse({"ok": True, "instruction": self._instruction})

        return app


def parse_args() -> SimConfig:
    parser = argparse.ArgumentParser(description="OpenVLA PyBullet simulation web visualizer")
    parser.add_argument("--openvla_path", type=str, required=True, help="HF path or local model directory")
    parser.add_argument("--unnorm_key", type=str, default="bridge_orig")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--hz", type=float, default=2.0, help="Inference+control loop frequency")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--pos_scale", type=float, default=0.04)
    parser.add_argument("--rot_scale", type=float, default=0.25)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    return SimConfig(
        openvla_path=args.openvla_path,
        unnorm_key=args.unnorm_key,
        attn_implementation=args.attn_implementation,
        hz=args.hz,
        width=args.width,
        height=args.height,
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        host=args.host,
        port=args.port,
    )


def main() -> None:
    cfg = parse_args()
    server = OpenVLASimServer(cfg)
    app = server.app()
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()

