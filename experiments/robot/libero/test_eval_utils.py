"""Regression tests for deterministic LIBERO episode setup."""

from experiments.robot.libero.eval_utils import reset_libero_episode


class _RecordingEnv:
    def __init__(self):
        self.calls = []

    def seed(self, value):
        self.calls.append(("seed", value))

    def reset(self):
        self.calls.append(("reset",))

    def set_init_state(self, state):
        self.calls.append(("set_init_state", state))
        return {"state": state}


def test_reset_libero_episode_reseeds_before_fixture_reset():
    env = _RecordingEnv()

    result = reset_libero_episode(env, initial_state=(1, 2, 3), seed=17)

    assert env.calls == [("seed", 17), ("reset",), ("set_init_state", (1, 2, 3))]
    assert result == {"state": (1, 2, 3)}
