"""
verify_openvla.py

Given an HF-exported OpenVLA model, attempt to load via AutoClasses, and verify forward() and predict_action().
"""

import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# === Verification Arguments
MODEL_PATH = "openvla/openvla-7b"
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
INSTRUCTION = "put spoon on towel"


def get_openvla_prompt(instruction: str) -> str:
    if "v01" in MODEL_PATH:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


@torch.inference_mode()
def verify_openvla() -> None:
    print(f"[*] Verifying OpenVLAForActionPrediction using Model `{MODEL_PATH}`")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    # Load Processor & VLA
    print("[*] Instantiating Processor and Pretrained OpenVLA")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # === BFLOAT16 + FLASH-ATTN MODE (Flash Attention 2 needs CUDA; other devices use PyTorch SDPA) ===
    attn_implementation = "flash_attention_2" if device.type == "cuda" else "sdpa"
    print(f"[*] Loading in BF16 with attention implementation `{attn_implementation}` on {device}")
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    # === 8-BIT QUANTIZATION MODE (`pip install bitsandbytes`) :: [~9GB of VRAM Passive || 10GB of VRAM Active] ===
    # print("[*] Loading in 8-Bit Quantization Mode")
    # vla = AutoModelForVision2Seq.from_pretrained(
    #     MODEL_PATH,
    #     attn_implementation="flash_attention_2",
    #     torch_dtype=torch.float16,
    #     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )

    # === 4-BIT QUANTIZATION MODE (`pip install bitsandbytes`) :: [~6GB of VRAM Passive || 7GB of VRAM Active] ===
    # print("[*] Loading in 4-Bit Quantization Mode")
    # vla = AutoModelForVision2Seq.from_pretrained(
    #     MODEL_PATH,
    #     attn_implementation="flash_attention_2",
    #     torch_dtype=torch.float16,
    #     quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )

    print("[*] Iterating with Randomly Generated Images")
    for _ in range(100):
        prompt = get_openvla_prompt(INSTRUCTION)
        image = Image.fromarray(np.asarray(np.random.rand(256, 256, 3) * 255, dtype=np.uint8))

        # === BFLOAT16 MODE ===
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

        # === 8-BIT/4-BIT QUANTIZATION MODE ===
        # inputs = processor(prompt, image).to(device, dtype=torch.float16)

        # Run OpenVLA Inference
        start_time = time.time()
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        print(f"\t=>> Time: {time.time() - start_time:.4f} || Action: {action}")


if __name__ == "__main__":
    verify_openvla()
