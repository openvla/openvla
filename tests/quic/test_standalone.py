"""Standalone contracts use tiny dense hosts and local episode arrays only."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from quic_petf.artifacts import sha256_file
from quic_petf.data import fit_statistics, read_manifest
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

ROOT = Path(__file__).resolve().parents[2]


class Tokenizer:
    vocab_size = 512
    model_max_length = 128
    padding_side = 'right'
    pad_token_id = 0
    eos_token_id = 2
    eos_token = '</s>'

    def decode(self, values):
        return ' '.join(str(int(value)) for value in values)

    def batch_decode(self, values):
        return [self.decode(row) for row in values]

    def __call__(self, text, **kwargs):
        action = [int(value) for value in text.split('Out: ', 1)[1].replace('</s>', '').split()]
        return SimpleNamespace(input_ids=[1, 10, 11] + action + [2])


class Processor:
    tokenizer = Tokenizer()
    image_processor = SimpleNamespace(apply_transform=lambda image: torch.tensor(np.asarray(image).copy()).float().mean((0, 1)) / 255)

    def save_pretrained(self, path):
        Path(path).mkdir()
        (Path(path) / 'processor.json').write_text('{}')


def tiny_host(convolution=False):
    model = OpenVLAForActionPrediction.__new__(OpenVLAForActionPrediction)
    nn.Module.__init__(model)
    model.vision_backbone = nn.Module() if convolution else nn.Identity()
    if convolution:
        model.vision_backbone.proj = nn.Conv2d(3, 8, 1)
    model.projector = nn.Module()
    model.projector.fc2 = nn.Linear(8 if convolution else 3, 512)
    model.norm_stats = {'robot': {'action': {'q01': [-1.] * 7, 'q99': [1.] * 7, 'mask': [True] * 6 + [False]}}}

    def forward(self, input_ids, pixel_values, labels, **kwargs):
        if convolution:
            pixel_values = self.vision_backbone.proj(pixel_values[:, :, None, None]).flatten(1)
        logits = self.projector.fc2(pixel_values)[:, None].expand(-1, input_ids.shape[1], -1)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 512), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(logits=logits, loss=loss)
    model.forward = MethodType(forward, model)
    return model


class StandaloneTests(unittest.TestCase):
    def test_model_api_rejects_tensor_parallel_slicing_and_quantized_hosts(self):
        from quic_petf_vla import AdapterConfig
        target = {'projector.fc2': AdapterConfig(orders=(1,), block_count=1, base_size=4)}
        model = tiny_host()
        model.config = SimpleNamespace(text_config=SimpleNamespace(pretraining_tp=2))
        with self.assertRaisesRegex(ValueError, 'pretraining_tp'):
            model.enable_quic(target)
        self.assertIs(type(model.projector.fc2), nn.Linear)
        model.config.text_config.pretraining_tp = 1
        model.is_quantized = True
        with self.assertRaisesRegex(ValueError, 'dense'):
            model.load_quic_adapter('/not-loaded', {'checkpoint_sha256': 'a' * 64})
        model.is_quantized = False
        model.enable_quic(target)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'save_quic_adapter'):
                model.save_pretrained(Path(directory) / 'full-model')
            self.assertFalse((Path(directory) / 'full-model').exists())

    def create_data(self, directory):
        records = []
        for index, split in enumerate(('train', 'validation')):
            target = directory / f'{split}.npz'
            actions = np.full((2, 7), index * 1000., dtype=np.float32)
            actions[1] += 1
            np.savez(target, images=np.full((2, 8, 8, 3), 90, dtype=np.uint8), actions=actions,
                     instructions=np.asarray(['move the object'] * 2))
            records.append({'id': split, 'split': split, 'file': target.name,
                            'frames': 2, 'sha256': sha256_file(target)})
        path = directory / 'manifest.json'
        path.write_text(json.dumps({'schema_version': 'openvla-quic.episodes/v1', 'source': {'name': 'fixture'},
                                   'action_mask': [True] * 6 + [False], 'episodes': records}))
        return path

    def test_help_does_not_import_model_data_or_benchmark_runtimes(self):
        code = "import sys; from quic_petf.cli import main; "
        code += "\ntry: main(['--help'])\nexcept SystemExit: pass\n"
        code += "assert not any(k.split('.')[0] in {'torch','tensorflow','ovlab_core','libero','robosuite'} for k in sys.modules)"
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)

    def test_train_only_statistics_and_duplicate_episode_rejection(self):
        with tempfile.TemporaryDirectory() as name:
            path, manifest = read_manifest(self.create_data(Path(name)))
            statistics = fit_statistics(path, manifest)
            self.assertLess(max(statistics['q99']), 2)
            self.assertFalse(statistics['mask'][-1])
            manifest['episodes'][1]['file'] = manifest['episodes'][0]['file']
            manifest['episodes'][1]['sha256'] = manifest['episodes'][0]['sha256']
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'physical episode'):
                read_manifest(path)

    def test_real_adapter_tuning_identity_and_serialized_reload(self):
        already_imported = set(sys.modules)
        from quic_petf.host import apply_action_normalization
        from quic_petf.tuning import train, evaluate
        torch.manual_seed(31)
        pristine = tiny_host()
        identity = {'checkpoint_id': 'tiny', 'checkpoint_sha256': 'a' * 64}
        inventory = [{'name': 'projector.fc2', 'in_features': 3, 'out_features': 512, 'used_in_policy': True}]
        def load(host, training=False):
            model = copy.deepcopy(pristine)
            apply_action_normalization(model, host)
            return model, Processor(), {'device': 'cpu'}
        def identify(host):
            result = dict(identity)
            if 'normalization_contract_sha256' in host:
                result['normalization_contract_sha256'] = host['normalization_contract_sha256']
            return result
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest = self.create_data(root)
            config = {'host': {'device': 'cpu', 'unnorm_key': 'robot'}, 'data': {'manifest': str(manifest)},
                      'targets': {'projector.fc2': {'orders': [1], 'block_count': 1, 'base_size': 4}},
                      'training': {'max_steps': 2, 'learning_rate': .01, 'validation_batches': 1, 'gradient_checkpointing': False}}
            with patch('quic_petf.host.load_host', side_effect=load), \
                 patch('quic_petf.host.checkpoint_identity', side_effect=identify), \
                 patch('quic_petf.host.layer_inventory', return_value=(inventory, 0)):
                result = train(config, root / 'run', fit_action_normalization=True)
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(result['trainable_parameters'], 6)
                self.assertEqual(result['identity_logit_max_absolute_error'], 0)
                self.assertEqual(result['reload_logit_max_absolute_error'], 0)
                self.assertLess(max(result['normalization_contract']['action_statistics']['q99']), 2)
                restored = evaluate(config, root / 'run/adapter', root / 'evaluation')
                self.assertEqual(restored['role'], 'teacher_forced_validation')
                self.assertEqual(restored['action_tokens'], 7)
                self.assertTrue((root / 'run/adapter/adapter.safetensors').is_file())
                self.assertNotIn('tensorflow', set(sys.modules) - already_imported)
                self.assertNotIn('ovlab_core', set(sys.modules) - already_imported)

    def test_mixed_conv_linear_training_and_reload_with_frozen_host(self):
        from quic_petf.host import module_inventory
        from quic_petf.tuning import train
        torch.manual_seed(77)
        pristine = tiny_host(convolution=True)
        inventory = module_inventory(pristine)
        def load(host, training=False):
            return copy.deepcopy(pristine), Processor(), {'device': 'cpu'}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            config = {'host': {'device': 'cpu', 'unnorm_key': 'robot'},
                      'data': {'manifest': str(self.create_data(root))},
                      'targets': {key: {'orders': [1], 'block_count': 1, 'base_size': 4}
                                  for key in ('vision_backbone.proj', 'projector.fc2')},
                      'training': {'max_steps': 2, 'learning_rate': .01, 'validation_batches': 1,
                                   'gradient_checkpointing': False}}
            with patch('quic_petf.host.load_host', side_effect=load), \
                 patch('quic_petf.host.checkpoint_identity', return_value={'checkpoint_sha256': 'a' * 64}), \
                 patch('quic_petf.host.layer_inventory', return_value=(inventory, 0)):
                result = train(config, root / 'run')
            self.assertEqual(result['trainable_parameters'], 12)
            self.assertEqual(result['identity_logit_max_absolute_error'], 0)
            self.assertEqual(result['reload_logit_max_absolute_error'], 0)
            saved = json.loads((root / 'run/adapter/adapter.json').read_text())
            self.assertEqual(saved['format_version'], 2)
            self.assertEqual(saved['manifest'][0]['module_type'], 'Conv2d')

        # Conv-only adapters must also block accidental dense HF serialization.
        from quic_petf_vla import AdapterConfig
        model = tiny_host(convolution=True)
        model.enable_quic({'vision_backbone.proj': AdapterConfig(orders=[1], block_count=1, base_size=4)})
        with self.assertRaisesRegex(ValueError, 'save_quic_adapter'):
            model.save_pretrained('/unused-conv-dense-checkpoint')

    @unittest.skipUnless(importlib.util.find_spec('fastapi'), 'install the serve extra')
    def test_http_action_contract_uses_uint8_and_does_not_load_a_second_model(self):
        from quic_petf.server import create_app
        calls = []
        policy = SimpleNamespace(model=SimpleNamespace(), predict=lambda image, text: calls.append((image, text)) or np.zeros(7))
        app = create_app(policy)
        endpoint = next(route.endpoint for route in app.routes if route.path == '/act')
        self.assertEqual(endpoint({'image': [[[1, 2, 3]]], 'instruction': 'move'}), {'action': [0.] * 7})
        self.assertEqual(calls[0][0].dtype, np.uint8)
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):
            endpoint({'image': [[[-1, 2, 3]]], 'instruction': 'move'})


if __name__ == '__main__':
    unittest.main()
