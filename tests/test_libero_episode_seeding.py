from experiments.robot.libero.episode_utils import reset_libero_episode


class RecordingEnv:
    def __init__(self):
        self.calls = []

    def seed(self, seed):
        self.calls.append(("seed", seed))

    def reset(self):
        self.calls.append(("reset",))
        return len(self.calls)


def test_episode_seed_is_applied_before_reset():
    env = RecordingEnv()

    assert reset_libero_episode(env, seed=17) == 2
    assert env.calls == [("seed", 17), ("reset",)]


def test_each_episode_restarts_the_same_fixture_seed():
    env = RecordingEnv()

    reset_libero_episode(env, seed=5)
    reset_libero_episode(env, seed=5)
    reset_libero_episode(env, seed=5)

    assert [call for call in env.calls if call[0] == "seed"] == [("seed", 5), ("seed", 5), ("seed", 5)]
    assert env.calls[1][0] == "reset"
    assert env.calls[3][0] == "reset"


def test_none_seed_preserves_environment_default_rng():
    env = RecordingEnv()

    reset_libero_episode(env, seed=None)

    assert env.calls == [("reset",)]
