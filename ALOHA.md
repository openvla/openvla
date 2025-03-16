# OpenVLA-OFT+ in Real-World ALOHA Robot Tasks

## Relevant Files

Evaluation
* `experiments/robot/aloha/`: ALOHA training and eval files
  * `run_aloha_eval.py`: ALOHA eval script (CLIENT SIDE; see "SERVER SIDE" below)
  * `aloha_utils.py`: ALOHA eval utils
  * Other ALOHA robot environment files copied from the original [ALOHA GitHub repo](https://github.com/tonyzhaozh/aloha):
    * `constants.py`
    * `real_env.py`
    * `robot_utils.py`
* `experiments/robot/`: General eval utils files
  * `openvla_utils.py`: OpenVLA-specific eval utils
  * `robot_utils.py`: Other eval utils
* `vla-scripts/deploy.py`: VLA server deploy script (SERVER SIDE)

Note: Unlike the LIBERO evaluation setup, we use a server-client interface here. This is particularly useful if the user's machine which commands the robot does not have access to a local GPU with sufficient specs to run the fine-tuned VLA policies.

Training
* `experiments/robot/aloha/`: ALOHA training and eval files
  * `preprocess_split_aloha_data.py`: ALOHA data preprocessing script
* `vla-scripts/finetune.py`: VLA fine-tuning script

## Setup

Set up a conda environment for training policies and deploying them on the VLA server (see instructions in [SETUP.md](SETUP.md)).

## Fine-Tuning on ALOHA Robot Data

We assume that you have collected a set of expert demonstrations on the ALOHA robot already.

First, use our `preprocess_split_aloha_data.py` script to preprocess the raw ALOHA dataset: downsize images from 480x640 to 256x256 and split into training and validation sets. Below are examples for the `put X into pot` task in our paper (which has 3 possible target objects, 1 per episode):

```bash
python experiments/robot/aloha/preprocess_split_aloha_data.py \
  --dataset_path /scr/moojink/data/aloha1_raw/put_green_pepper_into_pot/ \
  --out_base_dir /scr/moojink/data/aloha1_preprocessed/ \
  --percent_val 0.05
python experiments/robot/aloha/preprocess_split_aloha_data.py \
  --dataset_path /scr/moojink/data/aloha1_raw/put_red_pepper_into_pot/ \
  --out_base_dir /scr/moojink/data/aloha1_preprocessed/ \
  --percent_val 0.05
python experiments/robot/aloha/preprocess_split_aloha_data.py \
  --dataset_path /scr/moojink/data/aloha1_raw/put_yellow_corn_into_pot/ \
  --out_base_dir /scr/moojink/data/aloha1_preprocessed/ \
  --percent_val 0.05
```

Then, convert the preprocessed ALOHA datasets into a single RLDS dataset that is compatible with OpenVLA fine-tuning. This process is the same as in the original OpenVLA repo. See instructions for converting to RLDS [here](https://github.com/moojink/rlds_dataset_builder) (a sample ALOHA preprocessed-to-RLDS conversion script is available [here](https://github.com/moojink/rlds_dataset_builder/blob/main/aloha1_put_X_into_pot_300_demos/aloha1_put_X_into_pot_300_demos_dataset_builder.py); this script converts the three preprocessed datasets above into one unified RLDS dataset, with train/val splits).

After converting to RLDS, register the dataset (which, for the example task above, would be called `aloha1_put_X_into_pot_300_demos`) with our dataloader by adding an entry for it in `configs.py` ([here](prismatic/vla/datasets/rlds/oxe/configs.py#L680)), `transforms.py` ([here](prismatic/vla/datasets/rlds/oxe/transforms.py#L928)), and `mixtures.py` ([here](prismatic/vla/datasets/rlds/oxe/mixtures.py#L216)). For reference, in each of these files, there are sample entries for the ALOHA datasets that we used in our paper.

Before fine-tuning, set the desired ALOHA action chunk size in [`prismatic/vla/constants.py`](prismatic/vla/constants.py) (see `NUM_ACTIONS_CHUNK` in `ALOHA_CONSTANTS`). We set it to 25 by default because we used a control frequency of 25 Hz in our ALOHA setup to reduce storage costs and training time (while still maintaining smoothness in the robot's motions). If you use 50 Hz, we recommend setting `NUM_ACTIONS_CHUNK` to `50`. In general, 1 second-long action chunks are a good default. Do NOT modify `ACTION_PROPRIO_NORMALIZATION_TYPE`: Since the ALOHA robot action space is absolute joint angles, we do not want to use a normalization scheme that clips outlier values (like the Q1-Q99 normalization we used with the relative end-effector pose actions for LIBERO), since that would prevent the model from outputting certain robot joint angles that are crucial for solving the task.

Now begin fine-tuning! Below is a sample command to fine-tune OpenVLA using our OFT+ recipe on the `put X into pot` task above ("+" in "OFT+" means FiLM is included for enhanced language grounding). Replace `X` in the first line with the number of GPUs available to you.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /PATH/TO/RLDS/DATASETS/DIR/ \
  --dataset_name aloha1_put_X_into_pot_300_demos \
  --run_root_dir /YOUR/CHECKPOINTS/AND/LOG/DIR/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --use_val_set True \
  --val_freq 10000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_note parallel_dec--25_acts_chunk--continuous_acts--L1_regression--3rd_person_img--left_right_wrist_imgs--proprio_state--film
```

The above training command should reproduce our OpenVLA-OFT+ results on the `put X into pot` task if `X = 8` and the 100K step checkpoint is evaluated. It will fine-tune OpenVLA using 3 input images (1 third-person image + 2 wrist camera images). Note that we use learning rate decay after a certain point (50K steps in the command above) since doing so speeds up training convergence (train L1 loss spikes down from our experience).

Best practices for fine-tuning:
* In general, we recommend fine-tuning until training L1 loss goes below 0.01 and starts to plateau.
  * One way to achieve this is to fine-tune using our default learning rate of `5e-4` until the loss starts to decrease very slowly, and then decay the learning rate by 10x to `5e-5` (which should make the loss spike down) and train until the training L1 loss finally plateaus.
* Depending on your dataset size, you may need to adjust some hyperparameters. For example, if you use a large dataset with over 300 demos, you may need to decay the learning rate later and train for longer for best performance. Decaying too earlier can lead to a suboptimal policy.
* If your task does not require good langauge grounding (e.g., if there is only one language instruction), FiLM is not necessary; consider setting `--use_film False` to train fewer model parameters.
* Please be sure to test your policy with the same device/GPU used to train it! Otherwise, performance may drop substantially. You may be able to avoid the performance drop if you merge the LoRA weights into the base model on the downstream device used for testing (e.g., if you train on H100 and then merge on A100 before testing on A100). You can see our script [vla-scripts/merge_lora_weights_and_save.py](vla-scripts/merge_lora_weights_and_save.py) for merging the LoRA adapter into the base model offline. It's okay if you already merged LoRA weights into the base OpenVLA model during fine-tuning; you can always redownload the base model and merge again as long as you still have the LoRA adapter (`merge_lora_weights_and_save.py` will handle this for you).

If you run into any issues, please open a new GitHub issue.

## Launching ALOHA Robot Evaluations

In the primary conda environment (`openvla-oft`) which you will use to launch the VLA server, install a few packages for the server-client interface:

```bash
conda activate openvla-oft
pip install uvicorn fastapi json-numpy
```

On the machine that you will use to command the robot, set up a second conda environment that will be used to run the robot environment, query the VLA server, and execute actions in the environment:

```bash
# Create and activate client conda environment
conda create -n openvla-oft-aloha python=3.10 -y
conda activate openvla-oft-aloha

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# Install packages needed for the ALOHA robot environment
pip install -r experiments/robot/aloha/requirements_aloha.txt
```

Launch the VLA server on the machine that has the GPU you will use to run model inference (using the `openvla-oft` conda environment). Below is a sample command for this (change as needed):

```bash
python vla-scripts/deploy.py \
  --pretrained_checkpoint /PATH/TO/FINETUNED/MODEL/CHECKPOINT/DIR/ \
  --use_l1_regression True \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --center_crop True \
  --num_open_loop_steps 25 \
  --unnorm_key aloha1_put_X_into_pot_300_demos
```

Then, run the ALOHA evaluation script. Specify the VLA server URL or IP address in the `vla_server_url` argument. Below is a sample command:

```bash
python experiments/robot/aloha/run_aloha_eval.py \
  --center_crop True \
  --num_open_loop_steps 25 \
  --use_vla_server True \
  --vla_server_url <URL OF VLA SERVER> \
  --num_rollouts_planned <NUM TEST ROLLOUTS> \
  --max_steps <MAX NUM STEPS PER ROLLOUT>
```

If you run into any issues, please open a new GitHub issue.

## Troubleshooting Tips

* Tip #1: If you run into a ROS error such as `ImportError: /lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0`, try running the following command in your client conda environment (`openvla-oft-aloha`):

    ```
    conda install -c conda-forge libffi
    ```
