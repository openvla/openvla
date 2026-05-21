"""
Run one OpenVLA action prediction from a local image or a generated toy image.

This example uses the lightweight Hugging Face AutoClass inference path from the README and keeps model loading optional
via --dry-run, so the prompt and image plumbing can be checked before downloading weights.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single OpenVLA offline inference example.")
    parser.add_argument("--model-path", default="openvla/openvla-7b", help="HF model id or local checkpoint path.")
    parser.add_argument(
        "--image-path", type=Path, default=None, help="Optional RGB image path. Uses a toy image if omitted."
    )
    parser.add_argument(
        "--instruction", default="put the spoon on the towel", help="Robot instruction to place in the prompt."
    )
    parser.add_argument(
        "--unnorm-key", default="bridge_orig", help="Dataset statistics key used to unnormalize actions."
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help='Attention implementation passed to from_pretrained. Use "flash_attention_2" if installed.',
    )
    parser.add_argument(
        "--device", default="auto", help='Device for non-quantized inference: "auto", "cuda:0", or "cpu".'
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Load the model with 4-bit bitsandbytes quantization."
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Load the model with 8-bit bitsandbytes quantization."
    )
    parser.add_argument("--local-files-only", action="store_true", help="Only use local model files / HF cache entries.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the prompt and image size without loading model weights."
    )
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path for saving the predicted action.")
    return parser.parse_args()


def get_openvla_prompt(instruction: str, model_path: str) -> str:
    instruction = instruction.lower()
    if "openvla-v01" in model_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction}? ASSISTANT:"
    return f"In: What action should the robot take to {instruction}?\nOut:"


def load_image(image_path: Optional[Path]) -> Image.Image:
    if image_path is not None:
        return Image.open(image_path).convert("RGB")

    # A deterministic toy image keeps the example runnable before a robot camera frame is available.
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(40, 220, 256, dtype=np.uint8)[None, :]
    image[..., 1] = np.linspace(220, 40, 256, dtype=np.uint8)[:, None]
    image[96:160, 96:160, 2] = 255
    return Image.fromarray(image)


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose at most one of --load-in-4bit and --load-in-8bit.")

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    model_kwargs: Dict[str, Any] = {
        "attn_implementation": args.attn_implementation,
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if importlib.util.find_spec("accelerate") is not None:
        model_kwargs["low_cpu_mem_usage"] = True
    elif args.load_in_4bit or args.load_in_8bit:
        raise ImportError(
            "Quantized loading requires `accelerate`; install it before using --load-in-4bit/--load-in-8bit."
        )

    if args.load_in_4bit or args.load_in_8bit:
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForVision2Seq.from_pretrained(args.model_path, **model_kwargs)
    else:
        device = resolve_device(args.device)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model_kwargs["torch_dtype"] = dtype
        model = AutoModelForVision2Seq.from_pretrained(args.model_path, **model_kwargs).to(device)

    return processor, model


def get_input_device(model, device_arg: str):
    if device_arg != "auto" and not (
        getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
    ):
        return resolve_device(device_arg)
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def main() -> None:
    args = parse_args()
    prompt = get_openvla_prompt(args.instruction, args.model_path)
    image = load_image(args.image_path)

    print(f"Prompt:\n{prompt}")
    print(f"Image size: {image.size}")

    if args.dry_run:
        print("Dry run complete; model weights were not loaded.")
        return

    import torch

    processor, model = load_model(args)
    input_device = get_input_device(model, args.device)
    input_dtype = torch.bfloat16 if input_device.type == "cuda" else torch.float32
    inputs = processor(prompt, image).to(input_device, dtype=input_dtype)

    action = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
    action_array = np.asarray(action)
    print(f"Predicted action shape: {action_array.shape}")
    print(action_array)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(
                {
                    "model_path": args.model_path,
                    "instruction": args.instruction,
                    "unnorm_key": args.unnorm_key,
                    "action": action_array.tolist(),
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
