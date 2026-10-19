"""Standalone console entry point; help/config loading does not import runtimes."""

import argparse
import json
from pathlib import Path
import sys


def parser():
    result = argparse.ArgumentParser(prog="openvla-quic", description="Standalone dense token OpenVLA with QuIC adapters")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("inventory", "prepare", "train", "evaluate", "predict", "serve"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True, help="Standalone JSON run configuration")
        command.add_argument("--checkpoint", help="Override the local dense checkpoint directory")
        command.add_argument("--revision", help="Immutable checkpoint source revision (40-character Git SHA)")
        command.add_argument("--unnorm-key", help="Action statistics key")
        command.add_argument("--device", help="Torch device, e.g. cuda:0 or cpu")
        command.add_argument("--json", action="store_true", help="Explicitly print the complete result document")
        if name in {"prepare", "train", "evaluate"}:
            command.add_argument("--output", type=Path, required=True, help="New output directory")
        if name in {"train", "evaluate"}:
            command.add_argument("--data-manifest", type=Path)
        if name == "train":
            command.add_argument("--fit-action-normalization", action="store_true")
        if name in {"evaluate", "predict", "serve"}:
            command.add_argument("--adapter", type=Path, required=name == "evaluate", help="Directory with adapter.json and adapter.safetensors")
        if name == "predict":
            command.add_argument("--image", type=Path, required=True)
            command.add_argument("--instruction", required=True)
        if name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8000)
    return result


def summary(document):
    for key in ("status", "optimizer_steps", "trainable_parameters", "adapter_path", "manifest", "output"):
        if key in document:
            print(f"{key.replace('_', ' ')}: {document[key]}")
    for key in ("baseline_validation", "adapter_validation"):
        metrics = document.get(key)
        if metrics:
            print(f"{key.replace('_', ' ')}: CE={metrics['action_token_cross_entropy']:.4f} "
                  f"accuracy={metrics['action_token_accuracy']:.2%}")
    if "action_token_cross_entropy" in document:
        print(f"Validation: CE={document['action_token_cross_entropy']:.4f} "
              f"accuracy={document['action_token_accuracy']:.2%}")
    if "action" in document:
        print("Action: " + " ".join(f"{value:.6g}" for value in document["action"]))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        from .configuration import load_config
        config = load_config(args.config)
        host = config.setdefault("host", {})
        for key, value in (("checkpoint", args.checkpoint), ("revision", args.revision),
                           ("unnorm_key", args.unnorm_key), ("device", args.device)):
            if value is not None:
                host[key] = str(Path(value).expanduser().resolve()) if key == "checkpoint" else value
        if getattr(args, "data_manifest", None) is not None:
            config.setdefault("data", {})["manifest"] = str(args.data_manifest.resolve())
        if args.command == "inventory":
            from .host import layer_inventory
            inventory, count = layer_inventory(host)
            if args.json:
                print(json.dumps({"language_layers": count, "linear_modules": inventory}))
            else:
                for item in inventory:
                    if item["used_in_policy"]:
                        print(f"{item['name']}  {item['in_features']} -> {item['out_features']}")
            return 0
        if args.command == "prepare":
            from .prepare import prepare
            document = {"status": "prepared", "manifest": str(prepare(config, args.output))}
        elif args.command == "train":
            from .tuning import train
            document = train(config, args.output, fit_action_normalization=args.fit_action_normalization)
        elif args.command == "evaluate":
            from .tuning import evaluate
            document = evaluate(config, args.adapter, args.output)
        elif args.command == "predict":
            import numpy as np
            from PIL import Image
            from .runtime import load_policy
            policy = load_policy(host, args.adapter)
            action = policy.predict(np.asarray(Image.open(args.image).convert("RGB")), args.instruction)
            document = {"action": action.tolist()}
        else:
            from .server import serve
            serve(host, args.adapter, bind=args.host, port=args.port)
            return 0
        if args.json:
            print(json.dumps(document, sort_keys=True, allow_nan=False))
        else:
            summary(document)
        return 0
    except KeyboardInterrupt:
        print("openvla-quic: interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"openvla-quic: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
