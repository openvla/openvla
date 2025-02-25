from pathlib import Path
import os
import json
import torch
from copy import deepcopy
import time
import argparse
import logging
import psutil
import GPUtil
from tqdm import tqdm

from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction, MoEOpenVLAForActionPrediction
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig, MoEOpenVLAConfig
from prismatic.overwatch import initialize_overwatch

# Initialize logger
overwatch = initialize_overwatch(__name__)

def convert_to_moe_model(
    model_path_or_id: str,
    output_path: str = None,
    num_experts: int = 4,
    num_selected_experts: int = 2,
    expert_dropout: float = 0.1,
    load_balancing_loss_weight: float = 0.01,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_progress: bool = True,
    torch_dtype=torch.bfloat16,
    use_flash_attention: bool = True,
    low_cpu_mem_usage: bool = True,
):
    """
    Convert an existing OpenVLA model to MoE version
    
    Args:
        model_path_or_id: Path to original model or HuggingFace model ID
        output_path: Path to save converted model (defaults to model_path + "-moe")
        num_experts: Number of expert networks
        num_selected_experts: Number of experts to use for each forward pass
        expert_dropout: Dropout probability for experts
        load_balancing_loss_weight: Weight for the load balancing auxiliary loss
        device: Device to load model on ("cuda" or "cpu")
        log_progress: Whether to log detailed progress
        torch_dtype: Torch data type for model weights
        use_flash_attention: Whether to use flash attention for faster processing
        low_cpu_mem_usage: Whether to minimize CPU memory usage during loading
        
    Returns:
        MoEOpenVLAForActionPrediction: The converted model
    """
    start_time = time.time()
    
    # Default output path - check if model_path_or_id is a HF model ID
    is_hf_model = "/" in model_path_or_id and not os.path.exists(model_path_or_id)
    
    if output_path is None:
        if is_hf_model:
            # For HF models, use the model ID as part of the output path
            model_name = model_path_or_id.split("/")[-1]
            output_path = f"{model_name}-moe"
        else:
            output_path = f"{model_path_or_id}-moe"
    
    if log_progress:
        if is_hf_model:
            overwatch.info(f"Starting conversion of HuggingFace model: {model_path_or_id}")
        else:
            overwatch.info(f"Starting conversion of local model at {model_path_or_id}")
        overwatch.info(f"Output will be saved to {output_path}")
        overwatch.info(f"Using {num_experts} experts with {num_selected_experts} selected per forward pass")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Check available GPU memory
    if device == "cuda" and torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        available_mem = gpu.memoryFree
        if log_progress:
            overwatch.info(f"Available GPU memory: {available_mem} MB")
            overwatch.info(f"Converting model on {device}")
    
    # Log memory usage before loading
    if log_progress:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # in MB
        overwatch.info(f"Memory usage before loading model: {mem_before:.2f} MB")
    
    # Load original model
    if log_progress:
        overwatch.info("Loading original model...")
    
    load_start = time.time()
    
    # Configure flash attention if requested
    attn_implementation = "flash_attention_2" if use_flash_attention else None
    
    # Load the model - either from local path or HuggingFace
    original_model = AutoModelForVision2Seq.from_pretrained(
        model_path_or_id,
        trust_remote_code=True,
        low_cpu_mem_usage=low_cpu_mem_usage,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    
    # Move to device
    original_model = original_model.to(device)
    
    # Also load the processor if it's a HF model (we'll save it with the output)
    processor = None
    if is_hf_model:
        if log_progress:
            overwatch.info(f"Loading processor from {model_path_or_id}...")
        processor = AutoProcessor.from_pretrained(model_path_or_id, trust_remote_code=True)
    
    load_time = time.time() - load_start
    if log_progress:
        overwatch.info(f"Original model loaded in {load_time:.2f} seconds")
        
        # Log model size
        param_count = sum(p.numel() for p in original_model.parameters())
        overwatch.info(f"Original model has {param_count/1e6:.2f}M parameters")
        
        # Log memory after loading
        process = psutil.Process(os.getpid())
        mem_after = process.memory_info().rss / 1024 / 1024  # in MB
        overwatch.info(f"Memory usage after loading model: {mem_after:.2f} MB")
        overwatch.info(f"Model loading required {mem_after - mem_before:.2f} MB additional memory")
    
    # Create MoE config
    if log_progress:
        overwatch.info("Creating MoE configuration...")
    
    original_config = original_model.config
    moe_config = MoEOpenVLAConfig(
        **vars(original_config),
        num_experts=num_experts,
        num_selected_experts=num_selected_experts,
        expert_dropout=expert_dropout,
        load_balancing_loss_weight=load_balancing_loss_weight,
    )
    
    # Create MoE model with this config
    if log_progress:
        overwatch.info("Creating MoE model from configuration...")
    
    moe_model = MoEOpenVLAForActionPrediction(moe_config).to(device)
    
    # Copy weights from original model
    if log_progress:
        overwatch.info("Copying weights from original model...")
    
    copy_start = time.time()
    moe_model.load_state_dict(original_model.state_dict(), strict=False)
    
    # Initialize MoE experts from the original model's weights
    if log_progress:
        overwatch.info("Initializing MoE experts from pretrained weights...")
    
    if hasattr(original_model, 'model') and hasattr(original_model.model, 'lm_head'):
        moe_model.action_moe.from_pretrained_layers(original_model.model.lm_head)
    
    copy_time = time.time() - copy_start
    if log_progress:
        overwatch.info(f"Model weights copied and initialized in {copy_time:.2f} seconds")
        
        # Log MoE model size
        moe_param_count = sum(p.numel() for p in moe_model.parameters())
        overwatch.info(f"MoE model has {moe_param_count/1e6:.2f}M parameters")
        overwatch.info(f"Parameter increase: {(moe_param_count - param_count)/1e6:.2f}M ({(moe_param_count/param_count - 1)*100:.2f}%)")
        
        # Log memory for MoE model
        process = psutil.Process(os.getpid())
        moe_mem = process.memory_info().rss / 1024 / 1024  # in MB
        overwatch.info(f"Memory usage after creating MoE model: {moe_mem:.2f} MB")
        overwatch.info(f"Additional memory for MoE: {moe_mem - mem_after:.2f} MB")
    
    # Check forward pass with dummy input
    if log_progress:
        overwatch.info("Testing MoE model with dummy input...")
    
    # Create dummy input for testing
    try:
        batch_size = 1
        seq_len = 20
        dummy_input_ids = torch.ones((batch_size, seq_len), dtype=torch.long).to(device)
        
        # Add the special token at the end that triggers action prediction
        dummy_input_ids[:, -1] = 29871
        
        with torch.no_grad():
            moe_start = time.time()
            outputs = moe_model(input_ids=dummy_input_ids)
            moe_end = time.time()
            
        inference_time = moe_end - moe_start
        if log_progress:
            overwatch.info(f"MoE model forward pass successful in {inference_time:.4f} seconds")
    except Exception as e:
        if log_progress:
            overwatch.error(f"Error during MoE model test: {e}")
            overwatch.warning("Continuing with model saving despite test failure")
    
    # Save the converted model
    if log_progress:
        overwatch.info("Saving MoE model...")
    
    save_start = time.time()
    # Move to CPU before saving to avoid CUDA OOM
    moe_model = moe_model.cpu()
    moe_model.save_pretrained(output_path)
    
    # Also save the processor if we loaded it
    if processor is not None:
        processor.save_pretrained(output_path)
        
    save_time = time.time() - save_start
    
    if log_progress:
        overwatch.info(f"Model saved in {save_time:.2f} seconds")
    
    # Copy dataset statistics to ensure action prediction works
    dataset_stats_path = None
    if is_hf_model:
        # Try to download dataset_statistics.json from HF
        try:
            from huggingface_hub import hf_hub_download
            dataset_stats_path = hf_hub_download(
                repo_id=model_path_or_id,
                filename="dataset_statistics.json"
            )
            if log_progress:
                overwatch.info(f"Downloaded dataset statistics from HuggingFace Hub")
        except Exception as e:
            if log_progress:
                overwatch.warning(f"Could not download dataset_statistics.json from Hub: {e}")
    else:
        dataset_stats_path = os.path.join(model_path_or_id, "dataset_statistics.json")
    
    if dataset_stats_path and os.path.exists(dataset_stats_path):
        if log_progress:
            overwatch.info("Copying dataset statistics...")
        
        with open(dataset_stats_path, "r") as f:
            stats = json.load(f)
        with open(os.path.join(output_path, "dataset_statistics.json"), "w") as f:
            json.dump(stats, f)
    
    total_time = time.time() - start_time
    if log_progress:
        overwatch.info(f"Total conversion time: {total_time:.2f} seconds")
        
        # Estimate GPU memory requirements for a 50GB GPU
        max_mem_used = max(mem_after, moe_mem) - mem_before
        estimated_max_memory = max_mem_used * 1.5  # Add buffer for computation
        if estimated_max_memory < 50 * 1024:  # 50GB in MB
            overwatch.info(f"Estimated max memory usage: {estimated_max_memory:.2f} MB (fits on a 50GB GPU)")
        else:
            overwatch.warning(f"Estimated max memory usage: {estimated_max_memory:.2f} MB (may exceed 50GB GPU)")
    
    return moe_model

def run():
    """
    Command-line entry point for model conversion
    """
    parser = argparse.ArgumentParser(description='Convert OpenVLA model to MoE version')
    parser.add_argument('--model_path_or_id', type=str, required=True, 
                        help='Path to the original OpenVLA model or HuggingFace model ID (e.g., "openvla/openvla-7b")')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save the converted model (default: model_name + "-moe")')
    parser.add_argument('--num_experts', type=int, default=4,
                        help='Number of expert networks (default: 4)')
    parser.add_argument('--num_selected_experts', type=int, default=2,
                        help='Number of experts to use for each forward pass (default: 2)')
    parser.add_argument('--expert_dropout', type=float, default=0.1,
                        help='Dropout probability for experts (default: 0.1)')
    parser.add_argument('--load_balancing_weight', type=float, default=0.01,
                        help='Weight for the load balancing auxiliary loss (default: 0.01)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to load model on ("cuda" or "cpu")')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Data type for model weights (default: bfloat16)')
    parser.add_argument('--disable_flash_attention', action='store_true',
                        help='Disable flash attention (reduces memory efficiency but improves compatibility)')
    parser.add_argument('--high_cpu_mem_usage', action='store_true',
                        help='Allow high CPU memory usage during loading (may be faster)')
    parser.add_argument('--quiet', action='store_true',
                        help='Disable detailed progress logging')
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    torch_dtype = dtype_map[args.dtype]
    
    # Print banner
    print("=" * 80)
    print(f"OpenVLA to MoE Converter")
    print("=" * 80)
    print(f"Source model: {args.model_path_or_id}")
    print(f"Target model: {args.output_path or args.model_path_or_id.split('/')[-1] + '-moe'}")
    print(f"Configuration: {args.num_experts} experts with {args.num_selected_experts} selected per forward pass")
    print(f"Device: {args.device}, Dtype: {args.dtype}")
    print(f"Flash Attention: {'Disabled' if args.disable_flash_attention else 'Enabled'}")
    print("=" * 80)
    
    try:
        # Run conversion
        convert_to_moe_model(
            model_path_or_id=args.model_path_or_id,
            output_path=args.output_path,
            num_experts=args.num_experts,
            num_selected_experts=args.num_selected_experts,
            expert_dropout=args.expert_dropout,
            load_balancing_loss_weight=args.load_balancing_weight,
            device=args.device,
            log_progress=not args.quiet,
            torch_dtype=torch_dtype,
            use_flash_attention=not args.disable_flash_attention,
            low_cpu_mem_usage=not args.high_cpu_mem_usage
        )
        print("\n✅ Conversion completed successfully!")
        print(f"The MoE model is saved at: {args.output_path or args.model_path_or_id.split('/')[-1] + '-moe'}")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(run()) 

"""
# Convert OpenVLA directly from HuggingFace Hub
python -m prismatic.models.conversion --model_path_or_id "openvla/openvla-7b" --output_path "./openvla-moe" --num_experts 4

# Convert OpenVLA from local path
python -m prismatic.models.conversion --model_path_or_id "./openvla-7b" --output_path "./openvla-moe" --num_experts 4

Supports loading directly from HuggingFace Hub
Uses bfloat16 precision by default (same as your example)
Enables flash attention for faster processing
Also downloads and saves the processor
Attempts to get dataset statistics from HuggingFace Hub
Provides detailed timing and memory usage logs

"""