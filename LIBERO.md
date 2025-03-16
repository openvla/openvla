# OpenVLA-OFT in the LIBERO Simulation Benchmark

## Relevant Files

Evaluation
* `experiments/robot/libero/`: LIBERO eval files
  * `run_libero_eval.py`: LIBERO eval script
  * `libero_utils.py`: LIBERO eval utils
* `experiments/robot/`: General eval utils files
  * `openvla_utils.py`: OpenVLA-specific eval utils
  * `robot_utils.py`: Other eval utils

Training
* `vla-scripts/finetune.py`: VLA fine-tuning script


## Setup

Set up a conda environment (see instructions in [SETUP.md](SETUP.md)).

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO) and required packages:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt  # From openvla-oft base dir
```

(Optional, if you plan to launch training) To download the [LIBERO datasets](https://huggingface.co/datasets/openvla/modified_libero_rlds) that we used in our fine-tuning
experiments, run the command below. This will download the LIBERO-Spatial, LIBERO-Object, LIBERO-Goal,
and LIBERO-10 datasets in RLDS data format (~10 GB total). You can use these to fine-tune OpenVLA or
train other methods. This step is optional since we provide pretrained OpenVLA-OFT checkpoints below.
Note that these are the same datasets used in the original OpenVLA project. If needed, see details on how to download the original non-RLDS datasets [here](https://github.com/openvla/openvla?tab=readme-ov-file#libero-setup).
```bash
git lfs clone git@hf.co:datasets/openvla/modified_libero_rlds
```

## Launching LIBERO Evaluations

We fine-tuned OpenVLA via LoRA (r=32) with our OFT recipe on four LIBERO task suites independently: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 (also called LIBERO-Long).
The four OpenVLA-OFT checkpoints for LIBERO are available on Hugging Face:
* [moojink/openvla-7b-oft-finetuned-libero-spatial](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial)
* [moojink/openvla-7b-oft-finetuned-libero-object](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-object)
* [moojink/openvla-7b-oft-finetuned-libero-goal](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-goal)
* [moojink/openvla-7b-oft-finetuned-libero-10](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-10)

To start evaluations with one of these checkpoints, run one of the commands below. Each will automatically download the appropriate checkpoint listed above. You can set the `TRANSFORMERS_CACHE` and `HF_HOME` environment variable to change where the checkpoint files get cached.

```bash
# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial

# Launch LIBERO-Object evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-object \
  --task_suite_name libero_object

# Launch LIBERO-Goal evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
  --task_suite_name libero_goal

# Launch LIBERO-10 (LIBERO-Long) evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10
```

Notes:
* The evaluation script will run 500 trials by default (10 tasks x 50 episodes each). You can modify the number of
  trials per task by setting `--num_trials_per_task`. You can also change the random seed via `--seed`. There are
  other arguments in the script; we set them to the default values that work with the OpenVLA-OFT checkpoints above.
* **NOTE: Setting `--center_crop True` is important** because we fine-tuned OpenVLA with random crop augmentations
  (we took a random crop with 90% area in every training sample, so at test time we simply take the center 90% crop).
* The evaluation script logs results locally. You can also log results in Weights & Biases
  by setting `--use_wandb True` and specifying `--wandb_project <PROJECT>` and `--wandb_entity <ENTITY>`.
* The results reported in our paper were obtained using **Python 3.10.14, PyTorch 2.2.0, and our
  [custom transformers v4.40.1 fork](https://github.com/moojink/transformers-openvla-oft.git)**
  on an **NVIDIA A100 GPU**, averaged over three random seeds. Please stick to these package versions if possible.
  Note that results may vary slightly if you use a different GPU than the A100. If the discrepancy is large,
  please post a GitHub issue, and we will look into it.

## Fine-Tuning on LIBERO Datasets

First, download the LIBERO datasets as mentioned above in the Setup section above: `libero_spatial_no_noops`, `libero_object_no_noops`, `libero_goal_no_noops`, `libero_10_no_noops`. (`"_no_noops"` stands for no no-op actions, i.e., training samples with near-zero actions are filtered out).

Then, launch the fine-tuning script with the OFT configuration below, replacing `X` in the first line with the number of GPUs. The command below launches fine-tuning on LIBERO-Spatial with the hyperparameters that we used in our paper. Here, batch size 8 per GPU will require ~62 GB VRAM, and batch size 1 per GPU will require ~25 GB VRAM.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /PATH/TO/RLDS/DATASETS/DIR/ \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /YOUR/CHECKPOINTS/AND/LOG/DIR/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state
```

The above training command should reproduce our OpenVLA-OFT results if `X = 8` and the 150K step checkpoint is evaluated.

You can replace `libero_spatial_no_noops` with `libero_object_no_noops`, `libero_goal_no_noops`, or `libero_10_no_noops`. You can also modify other args — e.g., if you want to train with just one input image from the third-person camera and disable proprio state input, you can set `--num_images_in_input 1` and `--use_proprio False`.

In general, we recommend fine-tuning until training L1 loss goes below 0.01 and starts to plateau (with the above configuration, it should reach ~0.006 L1 loss on LIBERO-Spatial after 150K gradient steps with 10x LR decay after 100K steps). However, for LIBERO-Goal only, we found that the 50K checkpoint (which was at ~0.02 L1 loss) performed best for unknown reasons. For all other task suites though, we found that the 150K checkpoint performed best.

Please be sure to test your policy with the same device/GPU used to train it! Otherwise, performance may drop substantially. You may be able to avoid the performance drop if you merge the LoRA weights into the base model on the downstream device used for testing (e.g., if you train on H100 and then merge on A100 before testing on A100). You can see our script [vla-scripts/merge_lora_weights_and_save.py](vla-scripts/merge_lora_weights_and_save.py) for merging the LoRA adapter into the base model offline. It's okay if you already merged LoRA weights into the base OpenVLA model during fine-tuning; you can always redownload the base model and merge again as long as you still have the LoRA adapter (`merge_lora_weights_and_save.py` will handle this for you).

If you run into any issues, please open a new GitHub issue. If you do not receive a response within 2 business days, please email Moo Jin Kim (moojink@cs.stanford.edu) to bring the issue to his attention.
