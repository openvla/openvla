"""Metadata-only bridge to the immutable user-supplied compound-PEFT snapshot.

Gate F deliberately does not import the bundled package named ``peft``.  Gate I
must provide an isolated loading strategy and validate the mathematics before
this bridge can construct an OpenVLA policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ARCHIVE_SHA256 = "b024ba61b852d83beec631b724489b3bc3055c4a883f2df0c05b6c9857103e9a"
MANIFEST_SHA256 = "8084213849149a47f9bf84dd0c9220b319faf7df8dba39cdef3894e85e00f845"
PEFT_VERSION = "0.12.1.dev0"


class CompoundPeftOpenVLAIntegrationIncompleteError(RuntimeError):
    """The legacy source exists but no validated OpenVLA integration exists."""


@dataclass(frozen=True, slots=True)
class CompoundPeftBackendBridge:
    """Locate and describe the snapshot without importing Torch or bundled PEFT."""

    source_root: Path

    @classmethod
    def from_repository_layout(cls) -> "CompoundPeftBackendBridge":
        external_root = Path(__file__).resolve().parents[2]
        return cls(external_root / "compound-peft")

    def describe(self) -> dict[str, object]:
        return {
            "backend_id": "compound-peft-legacy-reference",
            "source_origin": "user_supplied_archive",
            "source_revision_kind": "content_hash",
            "archive_sha256": ARCHIVE_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "package_name": "peft",
            "package_version": PEFT_VERSION,
            "source_import_status": "present" if self.source_root.is_dir() else "absent",
            "generic_compound_backend_status": "legacy_reference_available",
            "openvla_integration_status": "skeleton",
            "paper_equivalence_validated": False,
            "dense_adapter_materialization": True,
        }

    def load_openvla(self, request: dict[str, object]):
        del request
        raise CompoundPeftOpenVLAIntegrationIncompleteError(
            "compound-peft source is present, but its OpenVLA integration is deferred to Gate I"
        )
