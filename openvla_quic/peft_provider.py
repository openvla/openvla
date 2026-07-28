"""QuIC-PEFT provider bone connected only to the metadata bridge in Gate F."""

from .compound_peft_bridge import (
    CompoundPeftBackendBridge,
    CompoundPeftOpenVLAIntegrationIncompleteError,
)
from .ovlab_provider import PROVIDER_API_VERSION


class QuICPEFTProvider:
    variant_id = "quic-peft"

    def __init__(self) -> None:
        self.backend_bridge = CompoundPeftBackendBridge.from_repository_layout()

    def api_version(self) -> str:
        return PROVIDER_API_VERSION

    def describe(self) -> dict[str, object]:
        return {
            "family": "openvla_quic",
            "variant_id": self.variant_id,
            "provider_api_version": PROVIDER_API_VERSION,
            "backend": self.backend_bridge.describe(),
            "runtime_validated": False,
            "training_validated": False,
            "libero_validated": False,
            "compression_verified": False,
        }

    @staticmethod
    def _raise_incomplete():
        raise CompoundPeftOpenVLAIntegrationIncompleteError(
            "compound-peft legacy backend is available, but OpenVLA integration is incomplete"
        )

    def capability_description(self):
        self._raise_incomplete()

    def load(self, request):
        del request
        self._raise_incomplete()

    def reset_episode(self, request):
        del request
        self._raise_incomplete()

    def predict(self, request):
        del request
        self._raise_incomplete()

    def load_counts(self) -> dict[str, int]:
        return {"base_model": 0, "quic_artifact": 0, "processor": 0}

    def close(self) -> None:
        return None
