"""Public, dependency-neutral provider boundary for OVLAB QuIC integrations."""

from .ovlab_provider import PROVIDER_API_VERSION, create_provider

__all__ = ["PROVIDER_API_VERSION", "create_provider"]
