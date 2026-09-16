"""Regression tests for empty DynamicCache / prepare_inputs_for_generation (issue #346)."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_DIR = REPO_ROOT / "prismatic" / "extern" / "hf"


def _load_modeling_module():
    """Load modeling_prismatic without importing the heavy prismatic package __init__."""
    for name in ("prismatic", "prismatic.extern", "prismatic.extern.hf"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(REPO_ROOT / Path(*name.split(".")))]  # type: ignore[attr-defined]
            sys.modules[name] = mod

    def _load(mod_name: str, file_name: str):
        full = f"prismatic.extern.hf.{mod_name}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, HF_DIR / file_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    _load("configuration_prismatic", "configuration_prismatic.py")
    return _load("modeling_prismatic", "modeling_prismatic.py")


class _DummyModel:
    """Bare object to host prepare_inputs_for_generation without constructing weights."""


class TestPrepareInputsForGenerationCache(unittest.TestCase):
    modeling = None

    @classmethod
    def setUpClass(cls):
        cls.modeling = _load_modeling_module()

    def _call(self, **kwargs):
        # Call the unbound function explicitly so unittest does not bind `self`.
        return self.modeling.PrismaticForConditionalGeneration.prepare_inputs_for_generation(
            _DummyModel(), **kwargs
        )

    def test_supports_sdpa_is_class_attr(self):
        pre = self.modeling.PrismaticPreTrainedModel
        self.assertTrue(hasattr(pre, "_supports_sdpa"))
        self.assertIs(pre._supports_sdpa, True)
        self.assertNotIsInstance(vars(pre).get("_supports_sdpa"), property)

    def test_generation_mixin_in_mro(self):
        from transformers import GenerationMixin

        self.assertIn(GenerationMixin, self.modeling.PrismaticForConditionalGeneration.__mro__)

    def test_none_cache_keeps_full_prompt(self):
        input_ids = torch.arange(35).unsqueeze(0)
        pixel_values = torch.zeros(1, 3, 224, 224)
        attention_mask = torch.ones_like(input_ids)
        out = self._call(
            input_ids=input_ids,
            past_key_values=None,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=True,
        )
        self.assertEqual(tuple(out["input_ids"].shape), (1, 35))
        self.assertIsNone(out["past_key_values"])
        self.assertTrue(torch.equal(out["pixel_values"], pixel_values))
        self.assertTrue(torch.equal(out["attention_mask"], attention_mask))
        self.assertTrue(out["use_cache"])

    def test_empty_dynamic_cache_keeps_full_prompt_and_nulls_cache(self):
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            self.skipTest("DynamicCache not available in this transformers version")

        empty = DynamicCache()
        self.assertEqual(self.modeling._cache_length(empty), 0)
        self.assertIsNotNone(empty)  # truthy object that previously triggered slice

        input_ids = torch.arange(35).unsqueeze(0)
        pixel_values = torch.zeros(1, 3, 224, 224)
        out = self._call(
            input_ids=input_ids,
            past_key_values=empty,
            pixel_values=pixel_values,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
        )
        self.assertEqual(tuple(out["input_ids"].shape), (1, 35))
        self.assertIsNone(out["past_key_values"])
        self.assertIs(out["pixel_values"], pixel_values)

    def test_empty_cache_like_object_with_get_seq_length(self):
        class EmptyCache:
            def get_seq_length(self):
                return 0

        input_ids = torch.arange(10).unsqueeze(0)
        out = self._call(input_ids=input_ids, past_key_values=EmptyCache(), use_cache=True)
        self.assertEqual(tuple(out["input_ids"].shape), (1, 10))
        self.assertIsNone(out["past_key_values"])

    def test_nonempty_cache_slices_last_token(self):
        class NonEmptyCache:
            def get_seq_length(self):
                return 35

        cache = NonEmptyCache()
        input_ids = torch.arange(36).unsqueeze(0)
        pixel_values = torch.zeros(1, 3, 224, 224)
        out = self._call(
            input_ids=input_ids,
            past_key_values=cache,
            pixel_values=pixel_values,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
        )
        self.assertEqual(tuple(out["input_ids"].shape), (1, 1))
        self.assertEqual(int(out["input_ids"].item()), 35)
        self.assertIs(out["past_key_values"], cache)

    def test_legacy_tuple_cache_length(self):
        k = torch.zeros(1, 2, 8, 4)
        v = torch.zeros(1, 2, 8, 4)
        legacy = ((k, v),)
        self.assertEqual(self.modeling._cache_length(legacy), 8)

        input_ids = torch.arange(9).unsqueeze(0)
        out = self._call(input_ids=input_ids, past_key_values=legacy, use_cache=True)
        self.assertEqual(tuple(out["input_ids"].shape), (1, 1))
        self.assertIs(out["past_key_values"], legacy)

    def test_batch_size_gt_1_raises(self):
        input_ids = torch.arange(6).view(2, 3)
        with self.assertRaises(ValueError):
            self._call(input_ids=input_ids, past_key_values=None)


if __name__ == "__main__":
    unittest.main()
