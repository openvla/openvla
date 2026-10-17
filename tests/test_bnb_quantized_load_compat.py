"""Regression tests for bitsandbytes quantized-load compatibility (GH issues #286, #287).

transformers 4.40.1 (the pinned version) dispatches bitsandbytes-quantized loads through
accelerate's `dispatch_model()` with a synthesized single-device `device_map` *before* it sets
`model.is_loaded_in_4bit` / `model.is_loaded_in_8bit`. accelerate >= 1.2.0 decides between the
hooks path and a plain `model.to()` based on exactly those (still unset) flags, so dispatch
falls through to `model.to()` and transformers rejects it:

    ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.

These tests pin the two layers of protection against that crash:
  1. the declared `accelerate` requirement must exclude the broken versions (>= 1.2.0);
  2. `get_vla()` must fail fast with an actionable message when an incompatible `accelerate`
     is already installed in the environment.
"""

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]

# accelerate 1.2.0 removed the `is_bnb_quantized` force-dispatch term from `dispatch_model()`;
# every release from that one onward crashes the transformers 4.40.1 quantized-load path.
MIN_BREAKING_ACCELERATE_VERSION = "1.2.0"
BROKEN_ACCELERATE_VERSIONS = ("1.2.0", "1.2.1", "1.4.0", "1.15.0")
WORKING_ACCELERATE_VERSIONS = ("0.25.0", "0.26.1", "1.1.1")


def _accelerate_requirement() -> str:
    """Returns the accelerate requirement string declared in pyproject.toml."""
    import tomllib

    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        dependencies = tomllib.load(f)["project"]["dependencies"]

    requirements = [dep for dep in dependencies if dep.startswith("accelerate")]
    assert len(requirements) == 1, f"expected exactly one accelerate requirement, got: {requirements}"
    return requirements[0]


class AccelerateRequirementTest(unittest.TestCase):
    """The declared dependency range must exclude the accelerate versions that break bnb dispatch."""

    def test_broken_accelerate_versions_are_excluded(self):
        requirement = _accelerate_requirement()
        specifier = SpecifierSet(requirement[len("accelerate") :])
        for version_string in BROKEN_ACCELERATE_VERSIONS:
            self.assertNotIn(
                Version(version_string),
                specifier,
                f"{requirement} must exclude accelerate {version_string} (GH #286/#287)",
            )

    def test_working_accelerate_versions_are_allowed(self):
        requirement = _accelerate_requirement()
        specifier = SpecifierSet(requirement[len("accelerate") :])
        for version_string in WORKING_ACCELERATE_VERSIONS:
            self.assertIn(
                Version(version_string),
                specifier,
                f"{requirement} must still allow accelerate {version_string}",
            )


def _import_openvla_utils():
    """Imports `experiments.robot.openvla_utils` with heavy third-party deps stubbed out.

    The module under test imports tensorflow / PIL / transformers / prismatic at import time,
    but the compatibility guard under test does not need them (and bitsandbytes itself is not
    installable on all platforms), so they are replaced with inert mocks for this test.
    """
    module_name = "experiments.robot.openvla_utils"
    if module_name in sys.modules:
        return sys.modules[module_name]

    for name in (
        "tensorflow",
        "numpy",
        "PIL",
        "PIL.Image",
        "transformers",
        "prismatic",
        "prismatic.extern",
        "prismatic.extern.hf",
        "prismatic.extern.hf.configuration_prismatic",
        "prismatic.extern.hf.modeling_prismatic",
        "prismatic.extern.hf.processing_prismatic",
    ):
        sys.modules[name] = mock.MagicMock()

    sys.path.insert(0, str(REPO_ROOT))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(REPO_ROOT))


class FailFastGuardTest(unittest.TestCase):
    """`_raise_if_accelerate_breaks_bnb_dispatch()` must reject broken accelerate installs."""

    def _guard_result(self, installed_version):
        openvla_utils = _import_openvla_utils()
        with mock.patch.object(openvla_utils, "_accelerate_distribution_version", return_value=installed_version):
            try:
                openvla_utils._raise_if_accelerate_breaks_bnb_dispatch()
            except EnvironmentError as exc:
                return str(exc)
        return None

    def test_guard_raises_for_broken_versions(self):
        for version_string in BROKEN_ACCELERATE_VERSIONS:
            message = self._guard_result(version_string)
            self.assertIsNotNone(message, f"guard must fail for accelerate {version_string}")
            self.assertIn("accelerate<1.2.0", message, "guard must include the fix command")

    def test_guard_allows_working_versions(self):
        for version_string in WORKING_ACCELERATE_VERSIONS:
            self.assertIsNone(self._guard_result(version_string), f"guard must allow accelerate {version_string}")


class ParseVersionTest(unittest.TestCase):
    """`_parse_version()` must produce tuples that order versions correctly."""

    def test_parse_version_ordering(self):
        openvla_utils = _import_openvla_utils()
        self.assertLess(
            openvla_utils._parse_version("1.1.1"),
            openvla_utils._parse_version(MIN_BREAKING_ACCELERATE_VERSION),
        )
        self.assertEqual(
            openvla_utils._parse_version("1.2.0rc1"),
            openvla_utils._parse_version(MIN_BREAKING_ACCELERATE_VERSION),
        )


if __name__ == "__main__":
    unittest.main()
