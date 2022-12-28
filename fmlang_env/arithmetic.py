import random
from typing import TypeVar

import gym
import numpy as np

from fmlang_env.util import get_gym_version_in_float

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

GYM_VERSION = get_gym_version_in_float()


class Arithmetic(gym.Env):
    metadata = {}

    if GYM_VERSION >= 0.22:
        action_space: gym.spaces.Space[ActType]  # type: ignore
        observation_space: gym.spaces.Space[ObsType]  # type: ignore

    allowed_characters = list(map(str, range(10))) + list("()+-*/")
    char_num = len(allowed_characters)

    def __init__(self, max_len=100):
        self.observation_space = gym.spaces.Box(
            low=0, high=self.char_num, shape=(max_len,), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(self.char_num + 1)
        self.state, self.num_state, self.last_acc_rew, self.target_num = (
            None,
            None,
            None,
            None,
        )
        self.max_len = max_len

        self.reset()

    def step(self, action: int):
        if action == len(self.allowed_characters):  # go backwards
            if self.state:
                self.num_state[len(self.state) - 1] = 0
            self.state = self.state[:-1]
        else:
            if len(self.state) < self.max_len - 1:
                self.num_state[len(self.state)] = np.uint8(action + 1)
                self.state += self.allowed_characters[action]

        try:
            r = eval(self.state)
        except:  # noqa: E722
            r = None

        if r is not None:
            # 1 / (abs difference) is the accumulated reward (capped at 100).
            # the single-step reward is the current acc reward minus the last acc reward.
            acc_rew = 1 / max(0.01, abs(self.target_num - r))
            reward = acc_rew - self.last_acc_rew
            self.last_acc_rew = acc_rew
        else:
            reward = 0.0

        return self._get_obs(), reward, False, self._get_info()

    def reset(self, seed=None):
        if GYM_VERSION >= 0.22:
            super().reset(seed=seed)  # type: ignore

        self.state, self.num_state = "", np.zeros((self.max_len,), dtype=np.uint8)
        self.last_acc_rew = 0.0
        self.target_num = random.random() * self.max_len

        if GYM_VERSION >= 0.22:
            return self._get_obs(), self._get_info()
        else:
            return self._get_obs()  # no info for earlier version

    def render(self, mode="human"):
        print(self.state)

    def _get_obs(self):
        return self.num_state

    def _get_info(self):
        return {"text": self.state}
