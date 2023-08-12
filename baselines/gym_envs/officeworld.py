from socket import has_dualstack_ipv6
import gym
import numpy as np
from gym import spaces
import random

class OfficeWorldActions:
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

class OfficeWorldState:
    def __init__(self, agent_loc, has_coffee, has_mail):
        self.agent_loc = agent_loc
        self.has_coffee = has_coffee
        self.has_mail = has_mail
        self._cached_hash = None
        self.__hash__()

    def __str__(self):
        string = "("
        string += "("+str(self.agent_loc[0])+","+str(self.agent_loc[1])+"),"
        string += str(self.has_coffee)+","
        string += str(self.has_mail)+")"
        return string

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, OfficeWorldState):
            return False
        else:
            return self._cached_hash == other._cached_hash

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        self._cached_hash = hash((self.agent_loc, self.has_coffee, self.has_mail))
        return self._cached_hash


class OfficeworldEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, step_max, gridsize):
        super(OfficeworldEnv, self).__init__()
        if gridsize == (18,18):
            self.load_18x18_map()
        elif gridsize == (27,27):
            self.load_27x27_map()
        elif gridsize == (36,36):
            self.load_36x36_map()
        elif gridsize == (45,45):
            self.load_45x45_map()
        elif gridsize == (54,54):
            self.load_54x54_map()

        num_actions = 4
        num_states = ((self.grid_size[0] * self.grid_size[1]) * 4) # 4 possible combinations of has_coffee, has_mail
        self.stochastic_prob = 0.8

        self.step_max = step_max
        self.steps = 0
        self.done = False
        self.success = False
        self.num_episodes = 0
        self.total_reward = 0

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)
        self.id_to_action = {0: "RIGHT", 1: "LEFT", 2: "UP", 3:"DOWN"}
        self.action_probs = {0:[2,3], 1:[2,3], 2:[0,1], 3:[0,1]}

    def actual_action_due_to_stochasticity(self, action):
        actual_action = action
        if random.uniform(0,1) > self.stochastic_prob:
            if random.uniform(0,1) > 0.5: 
                actual_action = self.action_probs[action][0]
            else: 
                actual_action = self.action_probs[action][1]
        return actual_action

    def out_of_bounds(self, agent_loc):
        if agent_loc[0] >= self.height or agent_loc[0] < 0:
            return True
        if agent_loc[1] >= self.width or agent_loc[1] < 0:
            return True
        return False

    def step(self, action):
        agent_row, agent_col, has_coffee, has_mail = self.decode(self.state_id)
        reward = 0  # default reward when there is no pickup/dropoff
        self.done = False
        agent_loc = (agent_row, agent_col)
        max_row = self.height - 1
        max_col = self.width - 1

        action = self.actual_action_due_to_stochasticity(action)
        if action == OfficeWorldActions.RIGHT:
            new_row = agent_row
            new_col = agent_col + 1
        elif action == OfficeWorldActions.LEFT:
            new_row = agent_row
            new_col = agent_col - 1
        elif action == OfficeWorldActions.UP:
            new_row = agent_row - 1
            new_col = agent_col
        elif action == OfficeWorldActions.DOWN:
            new_row = agent_row + 1
            new_col = agent_col
        if self.out_of_bounds((new_row,new_col)):
            new_row, new_col = agent_row, agent_col

        new_has_coffee, new_has_mail = has_coffee, has_mail

        if has_coffee and has_mail and (new_row, new_col) == self._office_loc:
            self.done = True
            self.success = True
            reward = 1000
        else:
            self.done = False
            self.success = False
            reward = 0
            if not has_coffee and (new_row, new_col) in self._coffee_locs:
                new_has_coffee = 1
            elif not has_mail and (new_row, new_col) in self._mail_locs:
                new_has_mail = 1

        self.state = OfficeWorldState((new_row, new_col), new_has_coffee, new_has_mail)
        self.state_id = self.encode(self.state)
        
        self.steps += 1
        if self.steps == self.step_max:
            self.done = True
        if self.done:
            # self.render()
            self.num_episodes += 1
        self.total_reward += reward
        # self.render()
        # print(new_agent_loc, self.state_id)

        info = {}
        info["done"] = self.done
        info["succ"] = self.success
        info["reward"] = self.total_reward
        info["steps"] = self.steps
        info["num_episodes"] = self.num_episodes
        return self.state_id, reward, self.done, info

    def encode(self, state):
        current_loc = state.agent_loc
        has_coffee = state.has_coffee
        has_mail = state.has_mail

        x_index = current_loc[0] * self.width * 4
        y_index = current_loc[1] * 4
        if has_coffee and has_mail:
            o = 3
        elif has_coffee and not has_mail: 
            o = 2 
        elif not has_coffee and has_mail:
            o = 1
        else:
            o = 0
        index = x_index + y_index + o
        return index

    def decode(self, state_id):
        z_index = state_id % 4
        inter_index = (state_id - z_index) // 4
        y_index = inter_index % self.width
        x_index = inter_index // self.width
        if z_index == 3:
            has_coffee, has_mail = True, True
        elif z_index == 2:
            has_coffee, has_mail = True, False
        elif z_index == 1:
            has_coffee, has_mail = False, True
        elif z_index == 0:
            has_coffee, has_mail = False, False
        return x_index, y_index, has_coffee, has_mail

    def reset(self):
        self.steps = 0
        self.total_reward = 0
        self.done = False
        self.success = False 

        self.has_coffee = 0
        self.has_mail = 0
        self.state = OfficeWorldState(self._start_loc, self.has_coffee, self.has_mail)
        self.state_id = self.encode(self.state)
        return self.state_id

    def render(self, mode='human'):
        print(self.state.__str__())

    def close (self):
        pass

    def load_54x54_map(self):
        # Creating the map
        self.grid_size = (54,54)
        self.height = self.grid_size[0]
        self.width = self.grid_size[1]

        self.has_coffee = 0
        self.has_mail = 0

        # Adding the agent
        self._start_loc = (2,1)
        self._coffee_locs = [(12,18)]
        self._mail_locs = [(22,19)]
        self._office_loc = (30,33)
        self._rooms = {'a': (1,1), 'b': (52,1), 'c':(52,52), 'd':(1,52)}

        # Adding walls
        self.forbidden_transitions = set()
        for x in range(54):
           for y in [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51]:
               self.forbidden_transitions.add((x,y,OfficeWorldActions.DOWN))
               self.forbidden_transitions.add((x,y+2,OfficeWorldActions.UP))
        for y in range(54):
           for x in [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51]:
               self.forbidden_transitions.add((x,y,OfficeWorldActions.LEFT))
               self.forbidden_transitions.add((x+2,y,OfficeWorldActions.RIGHT))
        # adding 'doors'
        for y in [1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52]:
           for x in [2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50]:
               self.forbidden_transitions.remove((x,y,OfficeWorldActions.RIGHT))
               self.forbidden_transitions.remove((x+1,y,OfficeWorldActions.LEFT))
        for x in [1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52]:
            for y in [2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50]:
                self.forbidden_transitions.remove((x,y,OfficeWorldActions.UP))
                self.forbidden_transitions.remove((x,y+1,OfficeWorldActions.DOWN))  

    def load_45x45_map(self):
        # Creating the map
        self.grid_size = (45,45)
        self.height = self.grid_size[0]
        self.width = self.grid_size[1]

        self.has_coffee = 0
        self.has_mail = 0

        # Adding the agent
        self._start_loc = (2,1)
        self._coffee_locs = [(8,14)]
        self._mail_locs = [(18,15)]
        self._office_loc = (26,29)
        self._rooms = {'a': (1,1), 'b': (43,1), 'c':(43,43), 'd':(1,43)}

        # Adding walls
        self.forbidden_transitions = set()
        for x in range(45):
           for y in [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42]:
               self.forbidden_transitions.add((x,y,OfficeWorldActions.DOWN))
               self.forbidden_transitions.add((x,y+2,OfficeWorldActions.UP))
        for y in range(45):
           for x in [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42]:
               self.forbidden_transitions.add((x,y,OfficeWorldActions.LEFT))
               self.forbidden_transitions.add((x+2,y,OfficeWorldActions.RIGHT))
        # adding 'doors'
        for y in [1,4,7,10,13,16,19,22,25,28,31,34,37,40,43]:
           for x in [2,5,8,11,14,17,20,23,26,29,32,35,38,41]:
               self.forbidden_transitions.remove((x,y,OfficeWorldActions.RIGHT))
               self.forbidden_transitions.remove((x+1,y,OfficeWorldActions.LEFT))
        for x in [1,4,7,10,13,16,19,22,25,28,31,34,37,40,43]:
            for y in [2,5,8,11,14,17,20,23,26,29,32,35,38,41]:
                self.forbidden_transitions.remove((x,y,OfficeWorldActions.UP))
                self.forbidden_transitions.remove((x,y+1,OfficeWorldActions.DOWN))  

    def load_36x36_map(self):
        # Creating the map
        self.grid_size = (36,36)
        self.height = self.grid_size[0]
        self.width = self.grid_size[1]

        self.has_coffee = 0
        self.has_mail = 0

        # Adding the agent
        self._start_loc = (2,1)
        self._coffee_locs = [(8,14)]
        self._mail_locs = [(11,8)]
        self._office_loc = (17,20)
        self._rooms = {'a': (1,1), 'b': (34,1), 'c':(34,34), 'd':(1,34)}

        # Adding walls
        self.forbidden_transitions = set()
        for x in range(36):
           for y in [0,3,6,9,12,15,18,21,24,27,30,33]:
               self.forbidden_transitions.add((x,y,OfficeWorldActions.DOWN))
               self.forbidden_transitions.add((x,y+2,OfficeWorldActions.UP))
        for y in range(36):
           for x in [0,3,6,9,12,15,18,21,24,27,30,33]:
               self.forbidden_transitions.add((x,y,OfficeWorldActions.LEFT))
               self.forbidden_transitions.add((x+2,y,OfficeWorldActions.RIGHT))
        # adding 'doors'
        for y in [1,4,7,10,13,16,19,22,25,28,31,34]:
           for x in [2,5,8,11,14,17,20,23,26,29,32]:
               self.forbidden_transitions.remove((x,y,OfficeWorldActions.RIGHT))
               self.forbidden_transitions.remove((x+1,y,OfficeWorldActions.LEFT))
        for x in [1,4,7,10,13,16,19,22,25,28,31,34]:
            for y in [2,5,8,11,14,17,20,23,26,29,32]:
                self.forbidden_transitions.remove((x,y,OfficeWorldActions.UP))
                self.forbidden_transitions.remove((x,y+1,OfficeWorldActions.DOWN))  

    def load_27x27_map(self):
        # Creating the map
        self.grid_size = (27,27)
        self.height = self.grid_size[0]
        self.width = self.grid_size[1]

        self.has_coffee = 0
        self.has_mail = 0

        # Adding the agent
        self._start_loc = (2,1)
        self._coffee_locs = [(6,12)]
        self._mail_locs = [(10,8)]
        self._office_loc = (15,18)
        self._rooms = {'a': (1,1), 'b': (25,1), 'c':(25,25), 'd':(1,25)}

        # Adding walls
        self.forbidden_transitions = set()
        for x in range(27):
           for y in [0,3,6,9,12,15,18,21,24]:
               self.forbidden_transitions.add((x,y,OfficeWorldActions.DOWN))
               self.forbidden_transitions.add((x,y+2,OfficeWorldActions.UP))
        for y in range(27):
           for x in [0,3,6,9,12,15,18,21,24]:
               self.forbidden_transitions.add((x,y,OfficeWorldActions.LEFT))
               self.forbidden_transitions.add((x+2,y,OfficeWorldActions.RIGHT))
        # adding 'doors'
        for y in [1,4,7,10,13,16,19,22,25]:
           for x in [2,5,8,11,14,17,20,23]:
               self.forbidden_transitions.remove((x,y,OfficeWorldActions.RIGHT))
               self.forbidden_transitions.remove((x+1,y,OfficeWorldActions.LEFT))
        for x in [1,4,7,10,13,16,19,22,25]:
            for y in [2,5,8,11,14,17,20,23]:
                self.forbidden_transitions.remove((x,y,OfficeWorldActions.UP))
                self.forbidden_transitions.remove((x,y+1,OfficeWorldActions.DOWN))  

    def load_18x18_map(self):
        # Creating the map
        self.grid_size = (18,18)
        self.height = self.grid_size[0]
        self.width = self.grid_size[1]

        self.has_coffee = 0
        self.has_mail = 0

        # Adding the agent
        self._start_loc = (2,1)
        self._coffee_locs = [(6,12)]
        self._mail_locs = [(8,6)]
        self._office_loc = (12,15)
        self._rooms = {'a': (1,1), 'b': (16,1), 'c':(16,16), 'd':(1,16)}

        # Adding walls
        self.forbidden_transitions = set()
        for x in range(18):
           for y in [0,3,6,9,12,15]:
               self.forbidden_transitions.add((x,y,OfficeWorldActions.DOWN))
               self.forbidden_transitions.add((x,y+2,OfficeWorldActions.UP))
        for y in range(18):
           for x in [0,3,6,9,12,15]:
               self.forbidden_transitions.add((x,y,OfficeWorldActions.LEFT))
               self.forbidden_transitions.add((x+2,y,OfficeWorldActions.RIGHT))
        # adding 'doors'
        for y in [1,4,7,10,13,16]:
           for x in [2,5,8,11,14]:
               self.forbidden_transitions.remove((x,y,OfficeWorldActions.RIGHT))
               self.forbidden_transitions.remove((x+1,y,OfficeWorldActions.LEFT))
        for x in [1,4,7,10,13,16]:
            for y in [2,5,8,11,14]:
                self.forbidden_transitions.remove((x,y,OfficeWorldActions.UP))
                self.forbidden_transitions.remove((x,y+1,OfficeWorldActions.DOWN))  