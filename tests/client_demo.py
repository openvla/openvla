import requests
import json_numpy
import numpy as np

# Patch json module to support numpy arrays natively
json_numpy.patch()

# 1. Get image from Isaac Sim camera
# Here we simulate an image tensor (e.g., shape 224x224x3)
image_array = np.zeros((224, 224, 3), dtype=np.uint8) 

instruction = "grasp the blue object"

# 2. Prepare payload
payload = {
    "image": image_array,
    "instruction": instruction,
    "unnorm_key": "bridge_orig"  # Optional key to specify normalization method on the server
}

# 3. Send request to WSL server
print("Sending request to OpenVLA server...")
try:
    response = requests.post("http://127.0.0.1:8000/act", json=payload)
    response.raise_for_status()
    
    # 4. Extract action
    action = response.json()
    print("Action received:", action)

except Exception as e:
    print(f"Failed to get action: {e}")