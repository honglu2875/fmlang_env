def get_gym_version_in_float():
    """
    This tries to get the gym version in float number, but will not report error.
    """
    r = 0
    try:
        import gym

        r = float(".".join(gym.__version__.split(".")[:2]))
    finally:
        return r  # noqa: B012
