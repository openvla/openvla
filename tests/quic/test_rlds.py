"""Optional real-TFDS preparation check; no simulator or downloaded dataset."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def temporal_standardizer(trajectory):
    import tensorflow as tf
    trajectory['action'] = tf.concat([tf.roll(trajectory['action'][:, :1], -1, axis=0),
                                      trajectory['action'][:, 1:]], axis=1)
    return trajectory


@unittest.skipUnless(os.environ.get('OPENVLA_QUIC_TEST_RLDS') == '1', 'opt into the installed rlds extra')
class RLDSPreparationTests(unittest.TestCase):
    def test_whole_trajectories_are_standardized_before_sampling_and_episode_split(self):
        import numpy as np
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        import tensorflow_datasets as tfds
        from prismatic.vla.datasets.rlds.oxe.configs import ActionEncoding, OXE_DATASET_CONFIGS
        from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
        from quic_petf.prepare import prepare
        from quic_petf.data import read_manifest

        class StandaloneEpisodeFixture(tfds.core.GeneratorBasedBuilder):
            VERSION = tfds.core.Version('1.0.0')

            def _info(self):
                return tfds.core.DatasetInfo(builder=self, features=tfds.features.FeaturesDict({
                    'steps': tfds.features.Dataset({
                        'observation': {'image': tfds.features.Image(shape=(8, 8, 3))},
                        'action': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                        'language_instruction': tfds.features.Text(),
                    }),
                }))

            def _split_generators(self, dl_manager):
                return {'train': self._generate_examples()}

            def _generate_examples(self):
                for episode in range(4):
                    yield episode, {'steps': [{
                        'observation': {'image': np.full((8, 8, 3), frame + episode, dtype=np.uint8)},
                        'action': np.asarray([frame / 10] + [0.] * 5 + [1.], dtype=np.float32),
                        'language_instruction': 'move the fixture',
                    } for frame in range(4)]}

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            builder = StandaloneEpisodeFixture(data_dir=str(root / 'source'))
            builder.download_and_prepare()
            config = {'data': {'tfds_directory': str(builder.data_dir), 'dataset_name': builder.name,
                'seed': 7, 'validation_fraction': .25, 'frame_stride': 2, 'max_frames_per_episode': 2,
                'image_size': 16}}
            mapping = {'image_obs_keys': {'primary': 'image'}, 'action_encoding': ActionEncoding.EEF_POS}
            with patch.dict(OXE_DATASET_CONFIGS, {builder.name: mapping}), \
                 patch.dict(OXE_STANDARDIZATION_TRANSFORMS, {builder.name: temporal_standardizer}):
                manifest_path = prepare(config, root / 'prepared')
            path, manifest = read_manifest(manifest_path)
            self.assertEqual(sum(row['split'] == 'train' for row in manifest['episodes']), 3)
            self.assertEqual(sum(row['split'] == 'validation' for row in manifest['episodes']), 1)
            self.assertEqual(manifest['action_mask'], [True] * 6 + [False])
            for row in manifest['episodes']:
                with np.load(path.parent / row['file'], allow_pickle=False) as arrays:
                    self.assertEqual(arrays['images'].shape, (2, 16, 16, 3))
                    np.testing.assert_allclose(arrays['actions'][:, 0], [.1, .3], atol=1e-6)
