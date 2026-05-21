# Examples

Small, runnable examples for common OpenVLA workflows.

## Offline inference

Run a single-image action prediction from the Hugging Face checkpoint:

```bash
python examples/offline_inference.py \
  --image-path /path/to/robot_view.jpg \
  --instruction "put the spoon on the towel" \
  --unnorm-key bridge_orig
```

The script can also validate the prompt and image preprocessing path without loading model weights:

```bash
python examples/offline_inference.py --dry-run
```

For a low-memory GPU run, install `bitsandbytes` and pass `--load-in-4bit` or `--load-in-8bit`. Quantized models are loaded with `device_map="auto"` and are not moved with `.to(...)`.

After the model has already been downloaded, pass `--local-files-only` to use the local Hugging Face cache or a local checkpoint directory.
