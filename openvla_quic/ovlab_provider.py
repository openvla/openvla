"""Versioned lazy provider factory; concrete variants remain independent."""

PROVIDER_API_VERSION = "1.0.0"


def create_provider(*, variant: str):
    if variant == "quic-peft":
        from .peft_provider import QuICPEFTProvider

        return QuICPEFTProvider()
    if variant == "quic-wc":
        from .wc_provider import QuICWCProvider

        return QuICWCProvider()
    raise ValueError(f"unknown QuIC variant: {variant!r}")
