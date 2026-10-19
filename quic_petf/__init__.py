"""Standalone PETF QuIC runtime for dense token OpenVLA.

Importing this namespace or using CLI help does not load Torch, TensorFlow,
OVLAB, or a simulator.
"""

__all__ = ["load_policy"]


def __getattr__(name):
    if name == "load_policy":
        from .runtime import load_policy
        return load_policy
    raise AttributeError(name)
