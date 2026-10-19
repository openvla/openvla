"""Standalone QuIC tuner. The original LoRA tuner is finetune_lora.py."""

import sys
from quic_petf.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
