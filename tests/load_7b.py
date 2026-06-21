import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

model_id = "openvla/openvla-7b"

print("1. Loading processor...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

print("2. Loading OpenVLA model...")
print("Note: This will download ~14GB of model weights on the first run. Please wait.")

try:
    vla = AutoModelForVision2Seq.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda:0")
    
    print("Success! OpenVLA successfully loaded into VRAM using Flash Attention 2.")
    
except Exception as e:
    print(f"Error loading model: {e}")