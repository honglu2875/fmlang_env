import gym
import pytest
import numpy as np
import fmlang_env


@pytest.mark.parametrize("obs_rew", [False, True])
@pytest.mark.parametrize("obs_target", [False, True])
def test_arith(obs_rew, obs_target):
    env = gym.make("fmlang/Arithmetic-v0", observe_rew=obs_rew, observe_target=obs_target)

    env.reset()
    action_space = env.action_space

    for i in range(10):
        action = action_space.sample()
        obs, rew, term, info = env.step(action)
        print(f"Reward: {rew}")
        env.render()

    if not obs_rew and not obs_target:
        assert isinstance(obs, np.ndarray)
    else:
        assert "state" in obs
        if obs_rew:
            assert "acc_rew" in obs
        if obs_target:
            assert "target" in obs
