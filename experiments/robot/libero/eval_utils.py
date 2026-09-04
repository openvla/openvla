"""Small, dependency-free helpers for deterministic LIBERO evaluation."""


def reset_libero_episode(env, initial_state, seed):
    """Reset an environment with a reproducible fixture layout before restoring its initial state.

    LIBERO samples placements for fixed fixtures during ``reset``.  The simulator's
    initial-state setter restores robot state, but does not restore those fixture
    placements.  Reseeding immediately before every reset keeps repeated episodes
    paired while still allowing each episode to use its own initial state.
    """
    env.seed(seed)
    env.reset()
    return env.set_init_state(initial_state)
