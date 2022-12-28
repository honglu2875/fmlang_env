import random
from typing import TypeVar, Union

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

    def __init__(
        self, max_len=100, observe_rew=True, observe_target=False, flatten_obs=True
    ):
        """
        :param max_len: the maximal length of the expression.
        :param observe_rew: add accumulated reward into the observation.
        :param observe_target: add the target number into the observation.
        :param flatten_obs: if `observe_rew` or `observe_target` is on, instead of returning a dict,
            return a flattened array.
        """
        self.observe_rew = observe_rew
        self.observe_target = observe_target
        self.flatten_obs = flatten_obs

        if not observe_rew and not observe_target:
            self.observation_space = gym.spaces.Box(
                low=0, high=self.char_num, shape=(max_len,), dtype=np.uint8
            )
        elif flatten_obs:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=max(100, max_len),
                shape=(max_len + observe_target + observe_rew,),
                dtype=np.float32,
            )
        else:
            obs_sp = {
                "state": gym.spaces.Box(
                    low=0, high=self.char_num, shape=(max_len,), dtype=np.uint8
                ),
            }
            if observe_rew:
                obs_sp["acc_rew"] = gym.spaces.Box(
                    low=0.0, high=100.0, shape=(1,), dtype=np.float32
                )
            if observe_target:
                obs_sp["target"] = gym.spaces.Box(
                    low=0.0, high=max_len, shape=(1,), dtype=np.float32
                )
            self.observation_space = gym.spaces.Dict(obs_sp)

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
        if not self.observe_rew and not self.observe_target:
            return self.num_state
        else:
            obs = {"state": self.num_state}
            if self.observe_rew:
                obs["acc_rew"] = np.array([self.last_acc_rew])
            if self.observe_target:
                obs["target"] = np.array([self.target_num])
            if self.flatten_obs:
                obs = self._flatten_obs(obs)
            return obs

    def _get_info(self):
        return {"text": self.state}

    @staticmethod
    def _flatten_obs(obs: Union[dict, np.ndarray]) -> np.ndarray:
        if isinstance(obs, np.ndarray):
            return obs.astype(np.float32)

        # fixing keys and orders to make sure the flattened array represent data in a given order.
        keys = ["state", "acc_rew", "target"]

        res = []
        for key in keys:
            if key in obs:
                res.append(np.float32(obs[key]))

        return np.concatenate(res, axis=0)
