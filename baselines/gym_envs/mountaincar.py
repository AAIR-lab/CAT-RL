__credits__ = ["Andrea PIERRÉ"]

import math
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np

import gym
from gym import error, spaces

class MountainCar():

    def __init__(
        self, step_max
    ):
        self.gym_env = gym.make('MountainCar-v0')
        self.step_max = step_max
        self.steps = 0
        self.done = False
        self.success = False
        self.num_episodes = 0
        self.total_reward = 0

        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.metadata = self.gym_env.metadata

    def step(self, action):
        self.done = False

        # next_state, reward, terminated = self.gym_env.step(action)
        next_state, reward, terminated, info = self.gym_env.step(action)

        
        self.steps += 1
        if self.steps == self.step_max:
            self.done = True
        else:
            self.done = terminated

        if terminated and "TimeLimit.truncated" not in info:
            self.success = True
            reward = 1000
        else:
            self.success = False

        if self.done:
            # self.gym_env.render()
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

    


    