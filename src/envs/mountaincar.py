import numpy as np
import math
import gym

class Mountain_Car:
    def __init__(self):
        self._env = gym.make ('MountainCar-v0')
        self._action_space = self._env.action_space
        self._action_size = 3
        self._n_state_variables = 2
        # self._state_ranges = [ (-1210, 610), 
        #                        (-710, 710)]
        self._original_state_ranges = [(-1.2, 0.6),
                                      (-0.07, 0.07)]
        self._gran = 0.001
        self._state_ranges = []
        for i in range (self._n_state_variables):
            low = math.floor(self._original_state_ranges[i][0] * 1/self._gran) 
            high = math.ceil(self._original_state_ranges[i][1] * 1/self._gran) + 1
            r = (low, high)
            self._state_ranges.append(r)
        self._vars_split_allowed = [1 for i in range(len(self._state_ranges))]
        self.log_r = []

    def seed(self, seed):
        self._env.seed(seed)

    def step (self, action_index):
        new_state, reward, done, _ = self._env.step(action_index)
        new_state = np.clip(new_state, a_min=self._env.observation_space.low, a_max=self._env.observation_space.high)
        self.log_r.append(reward)
        success = False
        if done and new_state[0] >= self._env.goal_position: 
            reward = 1000
            success = True
        return self.scale_state(new_state.tolist()), reward, done, success

    def render (self):
        self._env.render()

    def reset (self):
        start_state = self._env.reset()
        return self.scale_state(start_state.tolist())

    def scale_state (self, state):
        for i in range (len(state)):
            state[i] = math.ceil(state[i] * 1/self._gran)
        return state

    # action_index into action
    def index_to_action (self, action_index):
        return self._action_space [action_index]

    # state to state_index
    def state_to_index (self, state):
        return state[0]*self._dimension[0] + state[1]