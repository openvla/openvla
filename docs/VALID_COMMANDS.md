# OpenVLA（A5000 + Ubuntu 20.04）已验证命令清单

## 0) 基础路径与 Conda 路径

```bash
export WORK_ROOT=/mnt/data/home/yqy/YANGCHENX/openvla_sim
export OPENVLA_REPO=$WORK_ROOT/openvla
export MODEL_DIR=$WORK_ROOT/models/openvla-7b
export PATH=/mnt/data/home/yqy/miniforge3/bin:$PATH
export OPENVLA_PY=/mnt/data/home/yqy/miniforge3/envs/openvla/bin/python
```

## 1) 创建工作目录并克隆官方仓库

```bash
mkdir -p $WORK_ROOT
cd $WORK_ROOT
git clone https://github.com/openvla/openvla.git
```

## 2) 创建 Python 3.10 环境

```bash
conda create -n openvla python=3.10 -y
```

## 3) 安装核心依赖（GPU 推理）

```bash
conda run -n openvla pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
conda run -n openvla pip install "numpy<2"
cd $OPENVLA_REPO
conda run -n openvla pip install -r requirements-min.txt
conda run -n openvla pip install draccus==0.8.0 json-numpy fastapi uvicorn pillow accelerate==0.30.1 peft==0.11.1 einops sentencepiece==0.1.99
conda run -n openvla pip install packaging ninja
conda run -n openvla pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
cd $OPENVLA_REPO
conda run -n openvla pip install -e . --no-deps
```

## 4) 本地下载 OpenVLA-7B 权重

```bash
mkdir -p $MODEL_DIR
conda run -n openvla huggingface-cli download openvla/openvla-7b --local-dir $MODEL_DIR
```

## 5) 校验本地模型文件

```bash
ls -lah $MODEL_DIR/model-*.safetensors
python3 - <<'PY'
import os, glob, json
base="/mnt/data/home/yqy/YANGCHENX/openvla_sim/models/openvla-7b"
files=sorted(glob.glob(base+"/model-*-of-*.safetensors"))
print("num_shards", len(files))
print("sum_bytes", sum(os.path.getsize(f) for f in files))
with open(base+"/model.safetensors.index.json") as f:
    print("index_total_size", json.load(f)["metadata"]["total_size"])
PY
```

## 6) 本地推理连通性检查（已验证）

```bash
conda run --no-capture-output -n openvla python - <<'PY'
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np
import torch

model_path = "/mnt/data/home/yqy/YANGCHENX/openvla_sim/models/openvla-7b"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")

image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
prompt = "In: What action should the robot take to pick up the block?\nOut:"
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print("ok_action_shape", np.array(action).shape)
print("ok_action", action)
PY
```

## 7) 启动 REST 服务（已验证）

```bash
cd $OPENVLA_REPO
conda run --no-capture-output -n openvla python vla-scripts/deploy.py \
  --openvla_path $MODEL_DIR \
  --host 127.0.0.1 \
  --port 8000
```

## 8) 客户端请求测试（已验证）

```bash
conda run --no-capture-output -n openvla python - <<'PY'
import numpy as np
import requests
import json_numpy
json_numpy.patch()

payload = {
    "image": np.zeros((224, 224, 3), dtype=np.uint8),
    "instruction": "pick up the block",
    "unnorm_key": "bridge_orig",
}
r = requests.post("http://127.0.0.1:8000/act", json=payload, timeout=120)
print("status", r.status_code)
print("resp", r.text)
PY
```

## 9) 常用诊断命令

```bash
nvidia-smi
conda run -n openvla python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 10) MuJoCo 仿真页面可视化（已验证）

```bash
# 首次使用需要安装 MuJoCo 依赖
$OPENVLA_PY -m pip install "gymnasium[mujoco]==0.29.1"

# 启动 OpenVLA + MuJoCo 网页可视化闭环
cd $OPENVLA_REPO
$OPENVLA_PY vla-scripts/mujoco_openvla_viz.py \
  --openvla_path $MODEL_DIR \
  --env_id Reacher-v4 \
  --ctrl_scale 30 \
  --do_sample \
  --host 127.0.0.1 \
  --port 8012
```

打开浏览器访问：

```text
http://127.0.0.1:8012
```

可选调试（确认不是黑屏/静帧）：

```bash
curl -s http://127.0.0.1:8012/state | python -m json.tool
```

如果遇到 `ModuleNotFoundError: torch`，先执行：

```bash
$OPENVLA_PY -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

## 11) 官方 LIBERO 基准复现（OpenVLA README 对应流程）

### 11.1 克隆并安装 LIBERO（官方）

```bash
export LIBERO_ROOT=$WORK_ROOT/LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git $LIBERO_ROOT
cd $LIBERO_ROOT
$OPENVLA_PY -m pip install -e . --config-settings editable_mode=compat
```

### 11.2 安装 LIBERO 评测所需依赖（官方 + 运行补齐）

```bash
cd $OPENVLA_REPO
$OPENVLA_PY -m pip install -r experiments/robot/libero/libero_requirements.txt
$OPENVLA_PY -m pip install wandb
$OPENVLA_PY -m pip install jsonlines matplotlib rich tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3 "dlimp @ git+https://github.com/moojink/dlimp_openvla"
$OPENVLA_PY -m pip install "numpy<2" "opencv-python<4.12"
```

### 11.3 预置 LIBERO 配置（避免首次 import 交互提问）

```bash
mkdir -p /mnt/data/home/yqy/.libero
cat > /mnt/data/home/yqy/.libero/config.yaml <<'YAML'
benchmark_root: /mnt/data/home/yqy/YANGCHENX/openvla_sim/LIBERO/libero/libero
bddl_files: /mnt/data/home/yqy/YANGCHENX/openvla_sim/LIBERO/libero/libero/./bddl_files
init_states: /mnt/data/home/yqy/YANGCHENX/openvla_sim/LIBERO/libero/libero/./init_files
datasets: /mnt/data/home/yqy/YANGCHENX/openvla_sim/LIBERO/libero/../datasets
assets: /mnt/data/home/yqy/YANGCHENX/openvla_sim/LIBERO/libero/libero/./assets
YAML
mkdir -p /mnt/data/home/yqy/YANGCHENX/openvla_sim/LIBERO/datasets
```

### 11.4 官方评测脚本连通性检查

```bash
cd $OPENVLA_REPO
$OPENVLA_PY experiments/robot/libero/run_libero_eval.py --help
```

### 11.5 官方 checkpoint 复现（四套任务）

> 说明：以下是 OpenVLA README 中对应命令；会自动下载 checkpoint。  
> 默认每套是 10 任务 × 每任务 50 次（共 500 rollouts），耗时较长。

```bash
cd $OPENVLA_REPO
export HF_HOME=$WORK_ROOT/.hf_cache

# LIBERO-Spatial
$OPENVLA_PY experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True

# LIBERO-Object
$OPENVLA_PY experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True

# LIBERO-Goal
$OPENVLA_PY experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True

# LIBERO-10
$OPENVLA_PY experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True
```

### 11.6 快速 smoke test（先小规模确认可跑）

```bash
cd $OPENVLA_REPO
export HF_HOME=$WORK_ROOT/.hf_cache
$OPENVLA_PY experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 1 \
  --use_wandb False \
  --seed 7 \
  --run_id_note smoke_spatial
```

### 11.7 评测日志位置

```bash
ls -lah $OPENVLA_REPO/experiments/logs
```
