"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"  # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")  # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"  # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")  # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")  # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16  # Fine-tuning batch size
    max_steps: int = 200_000  # Max number of fine-tuning steps
    save_steps: int = 5000  # Interval for checkpoint saving
    learning_rate: float = 5e-4  # Fine-tuning learning rate
    grad_accumulation_steps: int = 1  # Number of batches to accumulate gradients over before performing
                                     # an optimization step. Effectively multiplies the batch_size by this
                                     # value while using less memory. Example: if batch_size=16 and
                                     # grad_accumulation_steps=4, this simulates training with
                                     # batch_size=64 but only requires memory for 16 samples at a time.
    image_aug: bool = True  # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000  # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True  # Whether to save only one checkpoint per run and
    #   continually overwrite the latest checkpoint
    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True  # Whether to use LoRA fine-tuning
    lora_rank: int = 32  # Rank of LoRA weight matrix
    lora_dropout: float = 0.0  # Dropout applied to LoRA weights
    use_quantization: bool = False  # Whether to 4-bit quantize VLA for LoRA fine-tuning
    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"  # Name of entity to log under
    run_id_note: Optional[str] = None  # Extra note for logging, Weights & Biases

    # Parameter for LeRobotDataset
    tolerance_s: float = 0.15

    # fmt: on



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    # TODO: FIgure out what ActionTokenizer and processor.tokenizer do
    # TODO: and how to duplicate this functionality in LeRobotDataset

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #       # TODO: Figure this out
    # ---
    from prismatic.vla.datasets.datasets import OpenVLALeRobotDataset

    vla_dataset = OpenVLALeRobotDataset(
        repo_id="NotRequired",
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=(
            PurePromptBuilder
            if "v01" not in cfg.vla_path
            else VicunaV15ChatPromptBuilder
        ),
        root=f"{cfg.data_root_dir}/{cfg.dataset_name}",
        tolerance_s=cfg.tolerance_s,
        image_transforms=None,
        download_videos=False,
        local_files_only=True,
    )


    # batch_transform = RLDSBatchTransform(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # vla_dataset = RLDSDataset(
    #     cfg.data_root_dir,
    #     cfg.dataset_name,
    #     batch_transform,
    #     resize_resolution=tuple(vla.module.config.image_sizes),
    #     shuffle_buffer_size=cfg.shuffle_buffer_size,
    #     image_aug=cfg.image_aug,
    # )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    sampler = RandomSampler(
        vla_dataset, num_samples=vla_dataset.num_frames
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,  # Set to 0 bc we don't use parallelism
                        # TODO: figure out if this is right?
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies_components = {
        action_dim_name: deque(maxlen=cfg.grad_accumulation_steps)
        for action_dim_name in ['dx', 'dy', 'dz', 'dox', 'doy', 'doz', 'grip']
    }
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_loss_components = {
        action_dim_name: deque(maxlen=cfg.grad_accumulation_steps)
        for action_dim_name in ['dx', 'dy', 'dz', 'dox', 'doy', 'doz', 'grip']
    }

    # Calculate number of epochs
    steps_per_epoch = len(dataloader)  # number of batches per epoch
    min_epochs = (cfg.max_steps * cfg.grad_accumulation_steps) // steps_per_epoch + 1
    num_epochs = min_epochs

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        # TODO: Debug

        # Compute the number of epochs needed to train for cfg.max_steps steps
        #   =>> This is used to set the number of epochs in the progress bar
        steps_per_epoch = len(dataloader)  # number of batches per epoch
        min_epochs = (cfg.max_steps * cfg.grad_accumulation_steps) // steps_per_epoch + 1
        num_epochs = min_epochs

        total_optimizer_steps = 0
        for _ in range(num_epochs):
            for batch_idx, batch in enumerate(dataloader):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Normalize loss to account for gradient accumulation
                normalized_loss = loss / cfg.grad_accumulation_steps

                # Backward pass
                normalized_loss.backward()

                # Compute Accuracy and L1 Loss for Logging
                # Will vla have a ".module"?
                assert isinstance(vla, DDP)
                # Is featurizer a vision transformer?
                assert hasattr(vla.module.vision_backbone.featurizer, "patch_embed")
                action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches: -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()
                action_accuracy_components = {
                    action_dim_name: (
                        (
                            (action_preds[:, i::7] == action_gt[:, i::7]) & mask[:, i::7]
                        ).sum().float() 
                        / mask[:, i::7].sum().float()
                    )
                    for i, action_dim_name in enumerate(['dx', 'dy', 'dz', 'dox', 'doy', 'doz', 'grip'])
                }

                # Compute L1 Loss on Predicted (Continuous) Actions

                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                action_l1_loss_components = {
                    action_dim_name: torch.nn.functional.l1_loss(
                        continuous_actions_pred[i::7],
                        continuous_actions_gt[i::7]
                    )
                    for i, action_dim_name in enumerate(['dx', 'dy', 'dz', 'dox', 'doy', 'doz', 'grip'])
                }

                # Store recent train metrics
                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())
                for action_dim_name in ['dx', 'dy', 'dz', 'dox', 'doy', 'doz', 'grip']:
                    recent_action_accuracies_components[action_dim_name].append(action_accuracy_components[action_dim_name].item())
                recent_l1_losses.append(action_l1_loss.item())
                for action_dim_name in ['dx', 'dy', 'dz', 'dox', 'doy', 'doz', 'grip']:
                    recent_l1_loss_components[action_dim_name].append(action_l1_loss_components[action_dim_name].item())

                # Compute gradient step index
                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
                wandb.log({
                    "batch_idx": batch_idx,
                    "grad_accumulation_steps": cfg.grad_accumulation_steps,
                    "gradient_step_idx": gradient_step_idx
                })

                # Compute smoothened train metrics
                #   =>> Equal to current step metrics when not using gradient accumulation
                #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
                smoothened_action_accuracy_components = {
                    action_dim_name: sum(recent_action_accuracies_components[action_dim_name]) / len(recent_action_accuracies_components[action_dim_name])
                    for action_dim_name in ['dx', 'dy', 'dz', 'dox', 'doy', 'doz', 'grip']
                }
                smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)
                smoothened_l1_loss_components = {
                    action_dim_name: sum(recent_l1_loss_components[action_dim_name]) / len(recent_l1_loss_components[action_dim_name])
                    for action_dim_name in ['dx', 'dy', 'dz', 'dox', 'doy', 'doz', 'grip']
                }

                # Optimizer Step
                if (
                    (batch_idx + 1) % cfg.grad_accumulation_steps == 0
                    or batch_idx == len(dataloader) - 1
                ):
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update()
                    total_optimizer_steps += 1
                    # Convert recent actions to tensors for logging, split by action dimension
                    action_pred_by_dim = {
                        action_dim_name: continuous_actions_pred[i::7].detach().cpu()
                        for i, action_dim_name in enumerate(['dx', 'dy', 'dz', 'dox', 'doy', 'doz', 'grip'])
                    }
                    action_gt_by_dim = {
                        action_dim_name: continuous_actions_gt[i::7].detach().cpu()
                        for i, action_dim_name in enumerate(['dx', 'dy', 'dz', 'dox', 'doy', 'doz', 'grip'])
                    }
                    
                    wandb.log({
                        "total_optimizer_steps": total_optimizer_steps,
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                        **{
                            f"l1_loss_{action_dim_name}": smoothened_l1_loss_components[action_dim_name]
                            for action_dim_name in ['dx', 'dy', 'dz', 'dox', 'doy', 'doz', 'grip']
                        },
                        **{
                            f"action_accuracy_{action_dim_name}": smoothened_action_accuracy_components[action_dim_name]
                            for action_dim_name in ['dx', 'dy', 'dz', 'dox', 'doy', 'doz', 'grip']
                        },
                        **{
                            f"recent_actions_pred_{action_dim_name}": wandb.Histogram(action_pred_by_dim[action_dim_name].numpy())
                            for action_dim_name in ['dx', 'dy', 'dz', 'dox', 'doy', 'doz', 'grip']
                        },
                        **{
                            f"recent_actions_gt_{action_dim_name}": wandb.Histogram(action_gt_by_dim[action_dim_name].numpy())
                            for action_dim_name in ['dx', 'dy', 'dz', 'dox', 'doy', 'doz', 'grip']
                        }
                    })

                # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
                if total_optimizer_steps > 0 and total_optimizer_steps % cfg.save_steps == 0:
                    if distributed_state.is_main_process:
                        print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                        # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                        save_dir = f"{adapter_dir}-{total_optimizer_steps}_chkpt" if cfg.use_lora else run_dir

                        # Save Processor & Weights
                        processor.save_pretrained(f"run_dir--{total_optimizer_steps}_chkpt")
                        vla.module.save_pretrained(save_dir)

                    # Wait for processor and adapter weights to be saved by main process
                    dist.barrier()

                    # Merge LoRA weights into model backbone for faster inference
                    #   =>> Note that merging is slow and can be done post-hoc to speed up training
                    if cfg.use_lora:
                        continue
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                        )
                        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                        merged_vla = merged_vla.merge_and_unload()
                        if distributed_state.is_main_process:
                            if cfg.save_latest_checkpoint_only:
                                # Overwrite latest checkpoint
                                # Save locally
                                # merged_vla.save_pretrained(run_dir)

                                # NOTE: don't save, we'll merge later
                                continue

                                print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                            else:
                                # Prepare to save checkpoint in new directory
                                checkpoint_dir = Path(str(run_dir) + f"--{total_optimizer_steps}_chkpt")
                                os.makedirs(checkpoint_dir, exist_ok=True)

                                # Save dataset statistics to new directory
                                save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                                # Save training statistics to new directory
                                with open(checkpoint_dir / "training_stats.json", "w") as f:
                                    json.dump({
                                        "total_optimizer_steps": total_optimizer_steps,
                                        "train_loss": smoothened_loss,
                                        "action_accuracy": smoothened_action_accuracy,
                                        "l1_loss": smoothened_l1_loss,
                                    }, f, indent=4)

                                # Save processor and model weights to new directory
                                processor.save_pretrained(checkpoint_dir)
                                merged_vla.save_pretrained(checkpoint_dir)

                                print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                    # Block on Main Process Checkpointing
                    dist.barrier()

                # Stop training when max_steps is reached
                if total_optimizer_steps == cfg.max_steps:
                    print(f"Max step {cfg.max_steps} reached! Stopping training...")
                    break

            if total_optimizer_steps == cfg.max_steps:
                break


if __name__ == "__main__":
    finetune()
