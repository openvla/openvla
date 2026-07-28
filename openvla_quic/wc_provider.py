"""Independent QuIC-WC skeleton with no compound-PEFT dependency."""

from .ovlab_provider import PROVIDER_API_VERSION


class QuICWCBackendIncompleteError(RuntimeError):
    """No QuIC-WC implementation or source artifact exists in Gate F."""


class QuICWCProvider:
    variant_id = "quic-wc"

    def api_version(self) -> str:
        return PROVIDER_API_VERSION

    def describe(self) -> dict[str, object]:
        return {
            "family": "openvla_quic",
            "variant_id": self.variant_id,
            "provider_api_version": PROVIDER_API_VERSION,
            "source_import_status": "absent",
            "implementation_status": "skeleton",
            "runtime_validated": False,
            "compression_verified": False,
            "dense_adapter_materialization_allowed": False,
            "persistent_dense_reconstruction_allowed": False,
            "replaced_dense_weights_required_at_deployment": False,
        }

    @staticmethod
    def _raise_incomplete():
        raise QuICWCBackendIncompleteError("QuIC-WC has no Gate F implementation or source backend")

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
