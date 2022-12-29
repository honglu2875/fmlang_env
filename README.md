![status](https://github.com/honglu2875/fmlang_env/actions/workflows/ci.yml/badge.svg?branch=main)

# fmlang_env
Toy gym env related to formal languages.

# Arithmetics

One can quickly try it with the following code.
```python
import gym
import fmlang_env

env = gym.make("fmlang/Arithmetic-v0")

env.reset()
action_space = env.action_space

for i in range(10):
    action = action_space.sample()
    obs, rew, term, info = env.step(action)
    print(f"Reward: {rew}")
    env.render()
```

## What it does
It starts with an empty string. Each action is either
- inserting a character which is either a digit or one or `"()+-*/"`, aiming to form an arithmetic expression,
- or deleting the previous character.

When env is initialized, one can set the max length of an expression `max_len`.
Every time the env is reset, there is a hidden target number.

The accumulated reward is

$$\text{min}(100, \dfrac{1}{|\text{eval} - \text{target}|}).$$

where `eval` is the eval result, and `target` is the target number. The step reward is what updates the accumulated
reward for the next observation. When the expression is invalid, the step reward is `0`.

## Specifications

### Observation space:
- `(max_len, )`, min = `0`, max = `16`.

`0` represents empty string, and `1-16` represents each possible character.

### Action space:
- Discrete(17)

`0-15` represents each character, and `16` means to delete the last character if any.

## A couple quirky things

- eval of empty string leads to syntax error. This is a great feature as to not exposing
the hidden target number as otherwise our agent can hit backspace and get the number right-away
- the definition of observation space representation and the action space can be confusing. Note
that `0` represents empty character in observation space and all digits are 1-based. But in action
space digits are 0-based with backspace being the last action button.
