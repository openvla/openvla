"""Prismatic APIs; importing the HF model does not load training/data stacks."""

__all__ = ["available_model_names", "available_models", "get_model_description", "load"]


def __getattr__(name):
    if name in __all__:
        from . import models
        return getattr(models, name)
    raise AttributeError(name)
