import gym
import fmlang_env


def test_arith():
    env = gym.make("fmlang/Arithmetic-v0")

    env.reset()
    action_space = env.action_space

    for i in range(10):
        action = action_space.sample()
        obs, rew, term, info = env.step(action)
        print(f"Reward: {rew}")
        env.render()
