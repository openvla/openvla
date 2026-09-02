"""Small helpers shared by LIBERO evaluation entry points."""


def reset_libero_episode(env, seed):
    """Reset an episode from an evaluation seed independent of prior rollouts.

    LIBERO randomizes task fixtures during ``reset``.  Re-seeding immediately
    before each reset makes the fixture depend only on the configured seed,
    rather than on random draws made by earlier policy rollouts.  This mirrors
    constructing a fresh seeded environment for every episode.
    """
    if seed is not None:
        env.seed(int(seed))
    return env.reset()
