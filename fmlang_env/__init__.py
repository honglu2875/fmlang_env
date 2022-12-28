from gym.envs.registration import register

from .arithmetic import Arithmetic

register(
    id="fmlang/Arithmetic-v0",
    entry_point="fmlang_env:Arithmetic",
    max_episode_steps=300,
)

__all__ = ["Arithmetic"]
