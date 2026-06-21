import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

model_id = "openvla/openvla-7b"

print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")

# 1. Prepare Image
# Using a dummy blue image for this test. 
# In production, use: image = Image.open("path_to_your_camera_frame.jpg").convert("RGB")
print("Preparing image and prompt...")
image = Image.new("RGB", (224, 224), color="blue")

# 2. Prepare Prompt
# OpenVLA requires a strict prompt template: "In: What action should the robot take to [task]?\nOut:"
prompt = "In: What action should the robot take to grasp the blue object?\nOut:"

# 3. Process inputs
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

# 4. Predict Action
print("Running inference...")
with torch.inference_mode():
    # unnorm_key="bridge_orig" unnormalizes the actions to the original dataset scale
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print("\n--- Inference Complete ---")
print("Predicted robotic action tensor (x, y, z, roll, pitch, yaw, gripper):")
print(action)