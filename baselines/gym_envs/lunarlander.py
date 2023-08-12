__credits__ = ["Andrea PIERRÉ"]

import math
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np

import gym
from gym import error, spaces

class LunarLander():

    def __init__(
        self, step_max
    ):
        self.gym_env = gym.make('LunarLander-v2')
        self.step_max = step_max
        self.steps = 0
        self.done = False
        self.success = False
        self.num_episodes = 0
        self.total_reward = 0


        low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -1.5,
                -1.5,
                # velocity bounds is 5x rated speed
                -5.0,
                -5.0,
                -math.pi,
                -5.0,
                -0.0,
                -0.0,
            ]
        ).astype(np.float32)
        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                1.5,
                1.5,
                # velocity bounds is 5x rated speed
                5.0,
                5.0,
                math.pi,
                5.0,
                1.0,
                1.0,
            ]
        ).astype(np.float32)

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Discrete(4)


    def step(self, action):
        self.done = False

        next_state, reward, terminated, info = self.gym_env.step(action)
        # terminates when collides or is not awake i.e. lands
        
        self.steps += 1
        if self.steps == self.step_max:
            self.done = True
        else:
            self.done = terminated

        if not self.gym_env.lander.awake:
            self.success = True
        else:
            self.success = False

        if self.done:
            # self.render()
            self.num_episodes += 1
        self.total_reward += reward

        info["done"] = self.done
        info["succ"] = self.success
        info["reward"] = self.total_reward
        info["steps"] = self.steps
        info["num_episodes"] = self.num_episodes
        # return np.array(state, dtype=np.float32), reward, terminated, False, {}
        return next_state, reward, self.done, info

    def reset(self):
        self.steps = 0
        self.total_reward = 0
        self.done = False
        self.success = False 
        return self.gym_env.reset()

    


    