__credits__ = ["Andrea PIERRÉ"]

import math
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np

import gym
from gym import error, spaces

class CartPole():

    def __init__(
        self, step_max
    ):
        self.gym_env = gym.make('CartPole-v1')
        self.step_max = step_max
        self.steps = 0
        self.done = False
        self.success = False
        self.num_episodes = 0
        self.total_reward = 0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-high, high)
        self.action_space = spaces.Discrete(2)


    def step(self, action):
        self.done = False

        next_state, reward, terminated, info = self.gym_env.step(action)
        if terminated:
            reward = -10
        
        self.steps += 1
        if terminated:
            self.done = True
            self.success = False
        if self.steps >= self.step_max:
            self.done = True
            if terminated:  
                self.success = False
            else:
                self.success = True
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

    


    