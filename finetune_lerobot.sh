#!/usr/bin/env bash
PYTHONPATH="$PYTHONPATH:$(pwd)/lerobot" && \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
	--vla_path "openvla/openvla-7b" \
	--data_root_dir "data/simple_task_lerobot" \
	--dataset_name "main" \
	--run_root_dir ".runs/" \
	--adapter_tmp_dir ".adapter/" \
	--lora_rank 32 \
	--batch_size 2 \
	--grad_accumulation_steps 8 \
	--learning_rate 5e-4 \
	--image_aug True \
	--max_steps 10000 \
	--save_steps 1000 \
	--save_latest_checkpoint_only False \
	--tolerance_s 0.01 \
	--wandb_entity robotgeneralist \
	--wandb_project ur5e
