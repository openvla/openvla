# OpenVLA 在 Ubuntu 20.04 + RTX A5000 上的完整部署与复现
  
内容基于官方 OpenVLA/LIBERO 文档，并结合本机复现部署过程整理而成。

---

## 1. 复现目标与范围

### 1.1 数据集

1. 在本地单机 GPU 上成功加载 OpenVLA 模型并推理。
2. 按官方方式，在 LIBERO 四个任务套件上运行评测脚本复现：
   - `LIBERO-Spatial`
   - `LIBERO-Object`
   - `LIBERO-Goal`
   - `LIBERO-10`

### 1.2 具体过程

1. 系统与硬件检查
2. Conda 环境与依赖安装
3. OpenVLA 与 LIBERO 仓库准备
4. 权重与数据下载（含断点续传、代理、报错补救）
5. 官方评测命令（smoke test + full eval）
6. 推理结果读取方法
7. 常见错误与修复
8. 模型架构与每一步操作作用解释

---

## 2. 机器配置（本机实测）

### 2.1 操作系统与内核

- Ubuntu 20.04.6 LTS
- Kernel: `5.15.0-139-generic`

### 2.2 GPU 与驱动

- GPU: `NVIDIA RTX A5000`（24GB 显存）
- Driver: `535.183.01`
- `nvidia-smi` 显示 CUDA Runtime: `12.2`

### 2.3 复现环境版本（建议保持一致）

- Python: `3.10.20`
- PyTorch: `2.2.0+cu121`
- transformers: `4.40.1`
- tokenizers: `0.19.1`
- timm: `0.9.10`
- flash-attn: `2.5.5`
- tensorflow: `2.15.0`
- tensorflow_datasets: `4.9.3`
- mujoco: `3.1.6`
- robosuite: `1.4.1`

> 这些版本组合是当前已验证可跑通的组合。OpenVLA 对版本比较敏感，特别是 `torch / transformers / flash-attn / numpy`。

---

## 3. 目录规划（建议）

建议统一工作根目录：

```bash
export WORK_ROOT=/mnt/data/home/yqy/YANGCHENX/openvla_sim
export OPENVLA_REPO=$WORK_ROOT/openvla
export LIBERO_ROOT=$WORK_ROOT/LIBERO
export OPENVLA_PY=/mnt/data/home/yqy/miniforge3/envs/openvla/bin/python
export HF_HOME=$WORK_ROOT/.hf_cache
```

建议目录结构：

```text
openvla_sim/
├── openvla/                      # OpenVLA 主仓库
├── LIBERO/                       # LIBERO 仓库
├── models/                       # 本地模型权重
│   ├── openvla-7b/
│   ├── openvla-7b-finetuned-libero-spatial/
│   ├── openvla-7b-finetuned-libero-object/
│   ├── openvla-7b-finetuned-libero-goal/
│   └── openvla-7b-finetuned-libero-10/
└── .hf_cache/                    # HuggingFace 下载缓存
```
---

## 4. 从零安装环境

### 4.1 克隆 OpenVLA

```bash
mkdir -p $WORK_ROOT
cd $WORK_ROOT
git clone https://github.com/openvla/openvla.git
```

### 4.2 创建 Conda 环境

```bash
conda create -n openvla python=3.10 -y
```

### 4.3 安装核心推理依赖（GPU）

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
  
> - `torch + cu121`：确保 GPU 计算可用。  
> - `requirements-min.txt`：最小推理依赖。  
> - `flash-attn`：降低显存占用并加速推理。  
> - `pip install -e .`：把当前仓库作为可编辑包安装，脚本可直接调用内部模块。

### 4.4 检查 GPU 与 Python 环境是否正常

```bash
nvidia-smi
conda run -n openvla python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 5. 安装 LIBERO 评测环境

### 5.1 克隆并安装 LIBERO

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git $LIBERO_ROOT
cd $LIBERO_ROOT
$OPENVLA_PY -m pip install -e . --config-settings editable_mode=compat
```

### 5.2 安装 OpenVLA-LIBERO 额外依赖

```bash
cd $OPENVLA_REPO
$OPENVLA_PY -m pip install -r experiments/robot/libero/libero_requirements.txt
$OPENVLA_PY -m pip install wandb
$OPENVLA_PY -m pip install jsonlines matplotlib rich tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3 "dlimp @ git+https://github.com/moojink/dlimp_openvla"
$OPENVLA_PY -m pip install "numpy<2" "opencv-python<4.12"
```

> - `tensorflow_datasets==4.9.3` 与 `dlimp` 是已知兼容组合。  
> - `numpy<2` 可规避一批旧依赖兼容问题。  
> - `opencv-python<4.12` 避免和 numpy 版本冲突。

### 5.3 初始化 LIBERO 配置文件（避免首次交互中断）

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

> 作用说明：LIBERO 首次导入会询问路径；提前写好配置后，脚本可非交互运行。

---

## 6. 模型权重下载（官方 checkpoint）

### 6.1 需要下载哪些权重

1. 基础模型（可选但推荐）：`openvla/openvla-7b`
2. LIBERO 四个官方微调模型（复现核心）：
   - `openvla/openvla-7b-finetuned-libero-spatial`
   - `openvla/openvla-7b-finetuned-libero-object`
   - `openvla/openvla-7b-finetuned-libero-goal`
   - `openvla/openvla-7b-finetuned-libero-10`

### 6.2 下载前建议环境变量

```bash
export WORK_ROOT=/mnt/data/home/yqy/YANGCHENX/openvla_sim
export HF_HOME=$WORK_ROOT/.hf_cache
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
```

如果走代理，另外加上：

```bash
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTPS_PROXY
```

> - `HF_HUB_ENABLE_HF_TRANSFER=0`：下载失败时错误信息更清晰，且常比 hf_transfer 稳。  
> - `HF_HUB_DISABLE_XET=1`：避免部分网络环境对 xet/cas 链路不稳定。

### 6.3 下载命令（逐个执行，支持断点续传）

```bash
mkdir -p $WORK_ROOT/models

/mnt/data/home/yqy/miniforge3/envs/openvla/bin/huggingface-cli download openvla/openvla-7b --local-dir $WORK_ROOT/models/openvla-7b
/mnt/data/home/yqy/miniforge3/envs/openvla/bin/huggingface-cli download openvla/openvla-7b-finetuned-libero-spatial --local-dir $WORK_ROOT/models/openvla-7b-finetuned-libero-spatial
/mnt/data/home/yqy/miniforge3/envs/openvla/bin/huggingface-cli download openvla/openvla-7b-finetuned-libero-object --local-dir $WORK_ROOT/models/openvla-7b-finetuned-libero-object
/mnt/data/home/yqy/miniforge3/envs/openvla/bin/huggingface-cli download openvla/openvla-7b-finetuned-libero-goal --local-dir $WORK_ROOT/models/openvla-7b-finetuned-libero-goal
/mnt/data/home/yqy/miniforge3/envs/openvla/bin/huggingface-cli download openvla/openvla-7b-finetuned-libero-10 --local-dir $WORK_ROOT/models/openvla-7b-finetuned-libero-10
```

> 说明：如果你的环境没有 `hf` 命令，用 `huggingface-cli` 是正常的。

### 6.4 如何判断权重是否下载完整

检查某个 checkpoint 是否具备全部分片和索引文件：

```bash
ls -lh $WORK_ROOT/models/openvla-7b-finetuned-libero-spatial/model-0000*-of-00004.safetensors
ls -lh $WORK_ROOT/models/openvla-7b-finetuned-libero-spatial/model.safetensors.index.json
```

如果 4 个分片 + `model.safetensors.index.json` 都存在，通常即可本地加载。

---

## 7. 官方复现命令（LIBERO）

### 7.1 先做最小 smoke test

```bash
cd $OPENVLA_REPO
export HF_HOME=$WORK_ROOT/.hf_cache
$OPENVLA_PY experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint $WORK_ROOT/models/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 1 \
  --use_wandb False \
  --seed 7 \
  --run_id_note smoke_spatial
```

> 先验证链路无误（模型加载、环境创建、动作推理、视频保存）再跑 500 次完整评测。

### 7.2 四套任务的正式复现

#### LIBERO-Spatial

```bash
cd $OPENVLA_REPO
$OPENVLA_PY experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint $WORK_ROOT/models/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True
```

#### LIBERO-Object

```bash
cd $OPENVLA_REPO
$OPENVLA_PY experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint $WORK_ROOT/models/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True
```

#### LIBERO-Goal

```bash
cd $OPENVLA_REPO
$OPENVLA_PY experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint $WORK_ROOT/models/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True
```

#### LIBERO-10（LIBERO-Long）

```bash
cd $OPENVLA_REPO
$OPENVLA_PY experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint $WORK_ROOT/models/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True
```

> 作用说明：  
> - `--center_crop True` 非常重要，官方 LIBERO 微调时使用过随机裁剪增强，推理时要做中心裁剪对齐分布。  
> - 默认 `10 tasks x 50 episodes = 500` 次 rollout，耗时较长。

---

## 8. 推理过程中每一步在做什么

下面对应 `experiments/robot/libero/run_libero_eval.py` 的主流程：

1. 读取配置与随机种子  
   作用：保证实验可复现（同一 seed 下结果波动更可控）。

2. 加载 OpenVLA 模型和 Processor  
   作用：把图像+文本 prompt 编码成模型输入，再生成动作 token。

3. 根据任务套件设置 `unnorm_key`  
   作用：把模型输出的归一化动作反归一化回该任务数据分布范围。

4. 初始化 LIBERO 环境与任务描述  
   作用：逐任务进行模拟，任务文本会注入提示词中。

5. 每回合开始先空转若干步（`num_steps_wait`）  
   作用：等待环境中物体稳定下落，减少初始抖动影响。

6. 取图像，必要时做中心裁剪，拼接 prompt  
   作用：让输入分布接近训练时分布，提高成功率。

7. `predict_action()` 输出 7 维动作  
   作用：得到末端位姿增量+夹爪开合控制。

8. 夹爪动作规范化 + 符号翻转  
   作用：适配 LIBERO 环境动作定义，避免“动作看似合理但执行方向反了”。

9. 环境 `step(action)` 执行并判断 done/success  
   作用：产生奖励、终止信号与下一帧观测。

10. 保存回放视频与日志  
    作用：便于可视化审查失败样本和统计汇报。

---

## 9. 模型架构

### 9.1 OpenVLA

OpenVLA 是 Vision-Language-Action（视觉-语言-动作）模型。  
它把“图像 + 指令文本”映射成“机器人动作”。

### 9.2 核心结构

1. 视觉编码器（Vision Backbone）  
   OpenVLA-7B 使用 Prismatic 的 `prism-dinosiglip-224px` 路线，融合 DINOv2 + SigLIP 视觉特征。

2. 语言模型骨干（LLM）  
   基于 Llama-2 系列语言模型结构作为动作生成主干。

3. 动作离散化（Action Tokenization）  
   连续动作会离散到 `256` 个 bin（代码中 `n_action_bins=256`），并映射到词表尾部 token，模型以“生成 token”的方式输出动作。

4. 动作反归一化（Unnormalization）  
   生成的是标准化动作，推理时依赖 `norm_stats`（如 `q01/q99`）恢复到真实动作量纲。

### 9.3 为什么这样设计

1. 可复用大模型生态：把机器人控制转成“序列生成”问题，可直接利用 HF + LLM 训练推理工具链。
2. 文本任务泛化更强：任务以自然语言描述，跨任务迁移更方便。
3. 统一接口：相机图像 + 文本进，动作出，和真实机器人部署接口一致。

---

## 10. 推理与结果读取

### 10.1 日志与视频位置

- 日志文件：`$OPENVLA_REPO/experiments/logs/*.txt`
- 回放视频：`$OPENVLA_REPO/rollouts/<date>/*.mp4`

### 10.2 查看实时进度

```bash
tail -f $OPENVLA_REPO/experiments/logs/EVAL-libero_spatial-openvla-*.txt
```

### 10.3 本机当前已观测到的实际结果样例

- `# episodes completed : 500`
- `# successes: 411 (82.2%)`

该值已接近官方报告量级（Spatial 官方均值约 `84.7% ± 0.9%`）。

---

## 11. 常见问题与补救（按实际高频问题整理）

### 11.1 `ModuleNotFoundError: No module named 'torch'`

原因：命令没走 `openvla` 环境的 Python。  
修复：

```bash
which python
/mnt/data/home/yqy/miniforge3/envs/openvla/bin/python -c "import torch; print(torch.__version__)"
```

统一用：

```bash
export OPENVLA_PY=/mnt/data/home/yqy/miniforge3/envs/openvla/bin/python
```

### 11.2 卡在 `[*] Loading in BF16 with Flash-Attention Enabled`

常见原因：

1. 正在首次下载大模型分片（不是死机）。
2. 网络或代理不稳定导致下载无速度。

排查：

```bash
watch -n 10 "date '+%F %T'; ls -lh $WORK_ROOT/models/openvla-7b-finetuned-libero-spatial/.cache/huggingface/download | tail -n 5"
```

### 11.3 `hf_transfer` 报错 / CAS 链路报错

典型报错：`RuntimeError: An error occurred while downloading using hf_transfer`  
修复：

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
```

然后重试下载命令，自动断点续传。

### 11.4 SSL 报错 `UNEXPECTED_EOF_WHILE_READING`

原因：代理链路不稳或 TLS 中断。  
修复建议：

1. 先确认代理端口可用（`127.0.0.1:7897`）。
2. 切换更稳定网络后重试。
3. 临时取消代理做对比测试。

### 11.5 `Address already in use`

原因：端口被占用。  
修复：

```bash
lsof -i :8016
kill -9 <PID>
```

或直接换端口。

### 11.6 TensorFlow / Gym / robosuite Warning 

以下通常不是致命错误：

1. TensorFlow oneDNN/cuDNN 注册 warning
2. Gym unmaintained 提示
3. robosuite 宏文件 warning

只要脚本继续推进 episode 并输出成功率，通常可忽略这些 warning。

### 11.7 下载到 0% 很久不动

先确认不是刚建立连接。若长时间 0B/s：

1. 换代理节点
2. 关闭 `HF_HUB_ENABLE_HF_TRANSFER`
3. 保留 `HF_HOME` 固定缓存，重复执行同一下载命令进行续传

---

## 12. 官方数据集说明

1. 跑官方 `run_libero_eval.py` 评测，核心依赖是 LIBERO 环境、任务定义和官方 checkpoint。
2. OpenVLA README 中提到的 `modified_libero_rlds`（约 10GB）主要用于微调训练，不是评测必须。

可选下载命令：

```bash
cd $WORK_ROOT
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds
```

> 作当后续要做“再训练/微调”时，这份 RLDS 数据是关键

---

## 13. 执行前后自检

### 13.1 执行前

1. `nvidia-smi` 正常
2. `openvla` conda 环境可用
3. `python -c "import torch; print(torch.cuda.is_available())"` 为 `True`
4. LIBERO config 文件存在
5. checkpoint 分片完整

### 13.2 执行后

1. 日志文件生成在 `experiments/logs/`
2. `rollouts/` 下生成 mp4
3. 日志中持续出现 `# successes: ...`
4. 完整 500 rollout 结束后有稳定总成功率

---

## 14. 推荐复现实验顺序（节省时间）

1. 先下载 `libero_spatial` checkpoint，跑 smoke test（1 trial/task）。
2. 再跑 `libero_spatial` 全量 500 rollout，确认链路稳定。
3. 按同样流程跑 `object -> goal -> libero_10`。
4. 最后统一汇总四套结果。

---

## 15. 参考入口（仓库内）

1. 官方 LIBERO 评测脚本：`experiments/robot/libero/run_libero_eval.py`
2. OpenVLA 推理工具：`experiments/robot/openvla_utils.py`
3. 机器人评测公共函数：`experiments/robot/robot_utils.py`
4. 动作离散器（256 bins）：`prismatic/vla/action_tokenizer.py`

---

## 16. 总结
在 `Ubuntu 20.04 + RTX A5000` 上，本流程已经可以打通：
1. OpenVLA 本地加载与 GPU 推理
2. LIBERO 官方四套任务的标准评测入口
3. 日志与视频证据链输出
4. 常见网络/依赖/端口错误的可执行补救方案
