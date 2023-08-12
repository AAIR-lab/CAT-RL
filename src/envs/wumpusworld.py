import numpy as np
import random 
import src.envs.map_maker as map_maker

class Simple_Grid:
    def __init__(self, env_name, start, goal):
        self._visited = []
        self._action_space = ['up','down','left','right']
        self._maze = map_maker.get_map(env_name)
        self._dimension = self._maze.shape
        self._state_size = self._dimension[0] * self._dimension[1]
        self._start = start
        self._goal = goal
        self._current_loc = self._start
        self._action_size = len(self._action_space)
        self._action_probs = {0:[2,3], 1:[2,3], 2:[0,1], 3:[0,1]}
        self._stoch_prob = 0.8
        self._visit_map = np.zeros_like(self._maze)
        self._n_state_variables = 2
        self._state_ranges = [ (0,self._dimension[0]), # robot y
                               (0,self._dimension[1]), # robot x
                                ]
        self._vars_split_allowed = [1 for i in range(len(self._state_ranges))]
        self._locations = []
        self._n_state_variables = 2
    def reset_visited (self):
        self._visit_map = np.zeros_like(self._maze)


    def action_stochastic (self, action_index):
        if random.uniform (0,1) > self._stoch_prob:
            if random.uniform (0,1) > 0.5 : 
                action_index_stoch = self._action_probs[action_index][0]
            else: action_index_stoch = self._action_probs[action_index][1]
        else: action_index_stoch = action_index
        return action_index_stoch


    def step (self, action_index_input):
        [a,b] = self._current_loc
        reward  = None # the episode's reward (-100 for pitfall, 0 for reaching the goal, and -1 otherwise)
        flag = False # termination flag is true if the agent falls in a pitfall or reaches to the goal
        flag_succ = False
        flag_pitfall = False
        action_index = self.action_stochastic (action_index_input)
        action = self.index_to_action (action_index)
        if action == 'up':
            next_loc = [a-1,b]
        elif action == 'down':
            next_loc = [a+1,b]
        elif action == 'left':
            next_loc = [a, b-1] 
        elif action == 'right':
            next_loc = [a, b+1] 

        if self._current_loc == self._goal:
            next_loc = self._current_loc
            x, y = next_loc[0], next_loc[1]
            self._visit_map[x][y] += 1
            reward = 500
            flag = True
            flag_succ = True
            return next_loc, reward, flag, flag_succ
        elif self._maze [self._current_loc[0]] [self._current_loc[1]] == -1:
            next_loc = self._current_loc
            x, y = next_loc[0], next_loc[1]
            self._visit_map[x][y] += 1
            flag_pitfall = True 
            reward = -1000
            flag = True
            return next_loc, reward, flag, flag_succ
        elif self.in_bound (next_loc) == False:
            reward = -2
            next_loc = self._current_loc
            x, y = next_loc[0], next_loc[1]
            self._visit_map[x][y] += 1
            flag = False
            return next_loc, reward, flag, flag_succ
        else:
            if self._maze [next_loc[0]] [next_loc[1]] == 1:
                next_loc = self._current_loc
                x, y = next_loc[0], next_loc[1]
                self._visit_map[x][y] += 1
                reward = -2
                flag = False
                return next_loc, reward, flag, flag_succ
            else:
                self._current_loc = next_loc
                x, y = next_loc[0], next_loc[1]
                self._visit_map[x][y] += 1
                reward = -1
                flag = False
                return next_loc, reward, flag, flag_succ

    def reset (self):
        self._current_loc = self._start
        return self._start

    def reset (self):
        self._current_loc = self._start
        return self._start

    # checks if a location is withing the env bound
    def in_bound (self, loc):
        flag = False
        if loc[0] < self._dimension[0] and loc[0] >= 0:
            if loc[1] < self._dimension[1] and loc[1] >= 0:
                flag = True
        return flag

    # action_index into action
    def index_to_action (self, action_index):
        return self._action_space [action_index]

    # state to state_index
    def state_to_index (self, state):
        return state[0]*self._dimension[0] + state[1]

    def update_visited (self, state):
        flag = True
        for i in self._visited:
            if state == i: flag = False
        if flag: self._visited.append(state)

    # state to state_index
    def state_to_index (self, state):
        return state[0]*self._dimension[0] + state[1]

    def transition (self, state, action_index_input):
        [a,b] = state
        reward  = None # the episode's reward (-100 for pitfall, 0 for reaching the goal, and -1 otherwise)
        flag = False # termination flag is true if the agent falls in a pitfall or reaches to the goal
        flag_succ = False
        flag_pitfall = False
        action_index = self.action_stochastic (action_index_input)
        action = self.index_to_action (action_index)
        if action == 'up':
            next_loc = [a-1,b]
        elif action == 'down':
            next_loc = [a+1,b]
        elif action == 'left':
            next_loc = [a, b-1] 
        elif action == 'right':
            next_loc = [a, b+1]


        if self.in_bound (next_loc) == False:
            reward = -2
            next_loc = state
            flag = False
            return next_loc, reward, flag, flag_succ, flag_pitfall
        else:
            if next_loc == self._goal:
                state = next_loc
                reward = 1000
                flag = True
                flag_succ = True
                return next_loc, reward, flag, flag_succ, flag_pitfall
            elif self._maze [next_loc[0]] [next_loc[1]] == -1:
                state = next_loc
                flag_pitfall = True 
                reward = -1000
                flag = True
                return next_loc, reward, flag, flag_succ, flag_pitfall
            elif self._maze [next_loc[0]] [next_loc[1]] == 1:
                next_loc = state
                reward = -2
                flag = False
                return next_loc, reward, flag, flag_succ, flag_pitfall
            else:
                state = next_loc
                reward = -1
                flag = False
                return next_loc, reward, flag, flag_succ, flag_pitfall