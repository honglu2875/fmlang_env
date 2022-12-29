import gym
import pytest
import numpy as np
import fmlang_env


@pytest.mark.parametrize("obs_rew", [False, True])
@pytest.mark.parametrize("obs_target", [False, True])
@pytest.mark.parametrize("flatten_obs", [False, True])
def test_arith_run(obs_rew, obs_target, flatten_obs):
    env = gym.make("fmlang/Arithmetic-v0", observe_rew=obs_rew, observe_target=obs_target, flatten_obs=flatten_obs)

    obs = env.reset()
    action_space = env.action_space

    for i in range(10):
        action = action_space.sample()
        obs, rew, term, info = env.step(action)

    if not obs_rew and not obs_target or flatten_obs:
        assert isinstance(obs, np.ndarray)
        if flatten_obs and (obs_rew or obs_target):
            assert obs.dtype == np.float32
    else:
        assert "state" in obs
        if obs_rew:
            assert "acc_rew" in obs
        if obs_target:
            assert "target" in obs


def test_arith_env():
    env = gym.make("fmlang/Arithmetic-v0", max_len=100, observe_rew=False, observe_target=False, flatten_obs=False)

    obs = env.reset()
    t = np.zeros((100, ), dtype=np.uint8)
    assert np.all(obs == t)
    env.env.target_num = 2.5  # the real class is wrapped in `.env` attribute.

    action = 1
    t[0] = 2  # 0 -> empty. 1 -> '0', 2 -> '1', etc.
    obs, rew, term, info = env.step(action)
    assert np.all(obs == t)
    assert abs(rew - 1 / 1.5) < 1e-6
    assert not term

    action = env.char_num  # test backspace
    t[0] = 0
    obs, rew, term, info = env.step(action)
    assert np.all(obs == t)
    assert abs(rew - 0) < 1e-6  # eval over empty string -> syntax error
    assert not term
