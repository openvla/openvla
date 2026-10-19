"""Minimal client: an RGB image and instruction in, decoded action out."""

import argparse
import json
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/act")
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    args = parser.parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    request = Request(args.url, data=json.dumps({"image": image.tolist(), "instruction": args.instruction}).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=120) as response:
        print(json.loads(response.read())["action"])


if __name__ == "__main__":
    main()
