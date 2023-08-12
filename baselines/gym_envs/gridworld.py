import gym
import numpy as np
import random
from gym import spaces


class GridWorldActions:
    EAST = 0
    WEST = 1
    NORTH = 2
    SOUTH = 3

class GridWorldState:
    def __init__(self, agent_loc):
        self.agent_loc = agent_loc
        self._cached_hash = None
        self.__hash__()

    def __str__(self):
        string = ""
        string += "("+str(self.agent_loc[0])+","+str(self.agent_loc[1])+")"
        return string

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, GridWorldState):
            return False
        else:
            return self._cached_hash == other._cached_hash

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        self._cached_hash = hash(self.agent_loc)
        return self._cached_hash


class GridworldEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, file_map, start_loc, dest_loc, step_max):
        super(GridworldEnv, self).__init__()

        self._load_map(file_map)
        self.start_loc = start_loc
        self.dest_loc = dest_loc
        num_actions = 4
        num_states = self.height * self.width
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
        self.id_to_action = {0: "East", 1: "West", 2: "North", 3:"South"}
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
        agent_row, agent_col = self.decode(self.state_id)
        reward = -1  # default reward
        self.done = False
        agent_loc = (agent_row, agent_col)
        max_row = self.height - 1
        max_col = self.width - 1

        action = self.actual_action_due_to_stochasticity(action)
        if action == GridWorldActions.EAST:
            new_row = agent_row
            new_col = agent_col + 1
        elif action == GridWorldActions.WEST:
            new_row = agent_row
            new_col = agent_col - 1
        elif action == GridWorldActions.NORTH:
            new_row = agent_row - 1
            new_col = agent_col
        elif action == GridWorldActions.SOUTH:
            new_row = agent_row + 1
            new_col = agent_col

        if (new_row, new_col) == self.dest_loc:
            self.done = True
            self.success = True
            reward = 500
            # print("Reached goal!!")
        elif self.out_of_bounds((new_row,new_col)):
            new_row, new_col = agent_row, agent_col
            self.done = False
            self.success = False
            reward = -2
        elif (new_row, new_col) in self.pitfalls:
            self.done = True
            self.success = False
            reward = -1000
        elif (agent_row,agent_col,action) in self.forbidden_transitions:
            new_row, new_col = agent_row, agent_col
            self.done = False
            self.success = False
            reward = -2

        agent_loc = (new_row, new_col)
        self.state = GridWorldState(agent_loc)
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
        # print(self.id_to_action[action])
        # self.render()
        info = {}
        info["done"] = self.done
        info["succ"] = self.success
        info["reward"] = self.total_reward
        info["steps"] = self.steps
        info["num_episodes"] = self.num_episodes
        return self.state_id, reward, self.done, info

    def encode(self, state):
        agent_row = state.agent_loc[0]
        agent_col = state.agent_loc[1]
        x_index = agent_row * self.width
        y_index = agent_col
        index = x_index + y_index
        return index

    def decode(self, i):
        out = []
        y_index = i % self.width
        out.append(y_index)
        x_index = i // self.width
        out.append(x_index)
        decoded_state = list(reversed(out))
        return decoded_state[0], decoded_state[1]

    def reset(self):
        self.steps = 0
        self.total_reward = 0
        self.done = False
        self.success = False 
        self.state = GridWorldState(self.start_loc)
        self.state_id = self.encode(self.state)
        # self.render()
        return self.state_id

    def render(self, mode='human'):
        print("Epi_num:{},Steps:{},{}".format(self.num_episodes,self.steps,self.state.__str__()))
        # for i in range(self.height):
        #     string = ""
        #     for j in range(self.width):
        #         if (i,j) == self.state.agent_loc:
        #             string += "A"
        #         elif (i-1,j,GridWorldActions.SOUTH) in self.forbidden_transitions and (i+1,j,GridWorldActions.NORTH) in self.forbidden_transitions and \
        #             (i,j-1,GridWorldActions.EAST) in self.forbidden_transitions and (i,j+1,GridWorldActions.WEST) in self.forbidden_transitions:
        #             string += 'X'
        #         elif (i,j) in self.pitfalls:
        #             string += "O"
        #         else:
        #             string += "-"
        #     string += "\n"
        #     print(string)

    def close (self):
        pass

    def _load_map(self,file_map):
        """
            This method adds the following attributes to the game:
                - self.objects: dict of features
                - self.forbidden_transitions: set of forbidden transitions (i,j,a)
                - self.agent: is the agent!
                - self.map_height: number of rows in every room
                - self.map_width: number of columns in every room
            The inputs:
                - file_map: path to the map file
        """
        self.pitfalls = set()
        self.forbidden_transitions = set()
        with open(file_map) as f:
            map = [line.rstrip()
                for line in f.readlines()
                if line.rstrip() # skip empty lines
                # if not "-" in line # skip beginning and end
            ]
        # loading the map
        for i,line in enumerate(map):
            for j,c in enumerate(line):
                e = line[j]
                if e == "X":
                    self.forbidden_transitions.add((i,j+1,GridWorldActions.WEST))
                    self.forbidden_transitions.add((i,j-1,GridWorldActions.EAST))
                    self.forbidden_transitions.add((i-1,j,GridWorldActions.SOUTH))
                    self.forbidden_transitions.add((i+1,j,GridWorldActions.NORTH))
                elif e == "O":
                    self.pitfalls.add((i,j))
        self.height, self.width = i+1, j+1 # last i and j used

        for i in range(self.height):
            self.forbidden_transitions.add((i,0,GridWorldActions.WEST))
            self.forbidden_transitions.add((i,self.width-1,GridWorldActions.EAST))
        for j in range(self.width):
            self.forbidden_transitions.add((0,j,GridWorldActions.NORTH))
            self.forbidden_transitions.add((self.height-1,j,GridWorldActions.SOUTH))
