"""
    inspired from https://gym.openai.com/envs/Taxi-v3/
"""

if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../')

from worlds.game_objects import *
from automata_learning.Traces import Traces
import random, math, os
import numpy as np
import random
from PIL import Image

"""
Auxiliary class with the configuration parameters that the Game class needs
"""
class GridWorldParams:
    def __init__(self, file_map):
        self.file_map = file_map

class GridWorld:

    def __init__(self, params):
        self.params = params
        agent = (0,0)
        self.dest = (63,63)
        self._load_map(params.file_map, agent)
        # self.get_map(params.file_map)
        self.env_game_over = False
        self._stoch_prob = 0.8
        self.bad_action = False

    def execute_action(self, a):
        """
            We execute 'action' in the game
        """
        action = Actions(a)
        agent = self.agent

        # MDP
        # p = 0.9
        p = self._stoch_prob # desactivate slip
        slip_p = [p,(1-p)/2,(1-p)/2]
        check = random.random()

        # print(check)

        if (check<=slip_p[0]):
            a_ = a

        elif (check>slip_p[0]) & (check<=(slip_p[0]+slip_p[1])):
            if a == 0:
                a_ = 3
            elif a == 2:
                a_ = 1
            elif a == 3:
                a_ = 2
            elif a == 1:
                a_ = 0

        else:
            if a == 0:
                a_ = 1
            elif a == 2:
                a_ = 3
            elif a == 3:
                a_ = 0
            elif a == 1:
                a_ = 2

        action_ = Actions(a_)
        self.a_ = a_

        # Getting new position after executing action
        ni,nj = agent.i, agent.j
        action_ = Actions(a_)
        self.a_ = a_
        if (ni,nj,action_) not in self.forbidden_transitions:
            if action_ == Actions.up   : ni-=1
            if action_ == Actions.down : ni+=1
            if action_ == Actions.left : nj-=1
            if action_ == Actions.right: nj+=1
            self.bad_action = False
        else:
            self.bad_action = True
        current_loc = self.objects.get((ni,nj), "")

        map = {Actions.up:"Up",Actions.down:"down",Actions.left:"left",Actions.right:"right"}
        # print(agent.i,agent.j,map[action_])
        agent.change_position(ni,nj)
        # print(agent.i,agent.j)
        # self.show_map()
        if (agent.i, agent.j) in self.objects:
            self.env_game_over = True

    def get_state(self):
        return None # we are only using "simple reward machines" for the taxi domain



    def get_actions(self):
        """
            Returns the list with the actions that the agent can perform
        """
        return self.actions

    def get_last_action(self):
        """
            Returns agent's last action
        """
        return self.a_

    def get_true_propositions(self):
        """
            Returns the string with the propositions that are True in this state
        """
        if self.bad_action:
            return "e"
        if (self.agent.i,self.agent.j) == self.dest:
            return "g"
        current_loc = self.objects.get((self.agent.i,self.agent.j), "")
        if current_loc: 
            loc_i = "a".index(current_loc.lower())
            return Traces.letters[loc_i]
        else: 
            return ""

        # ret = self.objects.get((self.agent.i,self.agent.j), "").lower()
        # ret += "efgh"["abcd".index(self.destination.lower())]
        # if self.passenger is not None: # at location
        #     ret += "ijkl"["abcd".index(self.passenger.lower())]
        # else: # in taxi
        #     ret += "m"
        # return ret

    # The following methods return different feature representations of the map ------------
    def get_features(self):
        N,M = self.map_height, self.map_width
        ret = np.zeros((N,M), dtype=np.float64)
        ret[self.agent.i,self.agent.j] = 1
        return ret.ravel() # from 2D to 1D (use a.flatten() is you want to copy the array)
        # if self.params.use_tabular_representation:
        #     return self._get_features_one_hot_representation()
        # return self._get_features_manhattan_distance()
        # return self._get_features_one_hot_representation()

    # The following methods create a string representation of the current state ---------

    def _load_map(self,file_map,agent):
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
        # contains all the actions that the agent can perform
        actions = [
            Actions.down.value,  # move south
            Actions.up.value,    # move north
            Actions.left.value,  # move east
            Actions.right.value, # move west
        ]
        self.actions = actions

        self.objects = {}
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
                    self.forbidden_transitions.add((i,j+1,Actions.left))
                    self.forbidden_transitions.add((i,j-1,Actions.right))
                    self.forbidden_transitions.add((i-1,j,Actions.down))
                    self.forbidden_transitions.add((i+1,j,Actions.up))
                elif e == "O":
                    self.objects[(i,j)] = "a"
        self.map_height, self.map_width = i+1, j+1 # last i and j used

        for i in range(self.map_height):
            self.forbidden_transitions.add((i,0,Actions.left))
            self.forbidden_transitions.add((i,self.map_width-1,Actions.right))
        for j in range(self.map_width):
            self.forbidden_transitions.add((0,j,Actions.up))
            self.forbidden_transitions.add((self.map_height-1,j,Actions.down))
        # while True:
        #     i, j = random.randrange(self.map_height), random.randrange(self.map_width)
        #     if (i,j) not in self.objects.keys(): break # prevent the taxi spawning on a location
        #     # break
        # self.agent = Agent(i,j,actions)
        self.agent = Agent(agent[0],agent[1],actions)



    # def show_map(self):
    #     """
    #         Prints the current map
    #     """
    #     print(self.__str__())

    # def __str__(self):
    #     r = "+" + "-"*(self.map_width*2-1) + "+\n"
    #     for i in range(self.map_height):
    #         s = "|"
    #         for j in range(self.map_width):
    #             if self.agent.idem_position(i,j):
    #                 # s += str(self.agent)
    #                 s += "A"
    #             else:
    #                 s += str(self.objects.get((i,j), " "))
    #             if (i,j,Actions.right) in self.forbidden_transitions:
    #                 s += "|"
    #             else:
    #                 s += ":"
    #         r += s + "\n"
    #     r += "+" + "-"*(self.map_width*2-1) + "+"
    #     return r

    # The following methods create the map ----------------------------------------------
    # def _load_map(self,file_map,agent):
    #     """
    #         This method adds the following attributes to the game:
    #             - self.objects: dict of features
    #             - self.forbidden_transitions: set of forbidden transitions (i,j,a)
    #             - self.agent: is the agent!
    #             - self.map_height: number of rows in every room
    #             - self.map_width: number of columns in every room
    #         The inputs:
    #             - file_map: path to the map file
    #     """
    #     # contains all the actions that the agent can perform
    #     actions = [
    #         Actions.down.value,  # move south
    #         Actions.up.value,    # move north
    #         Actions.left.value,  # move east
    #         Actions.right.value, # move west
    #     ]
    #     self.actions = actions

    #     self.objects = {}
    #     self.forbidden_transitions = set()
    #     with open(file_map) as f:
    #         map = [line.rstrip()
    #             for line in f.readlines()
    #             if line.rstrip() # skip empty lines
    #             if not "-" in line # skip beginning and end
    #         ]
    #     # loading the map
    #     for i,line in enumerate(map):
    #         for j,c in enumerate(range(1,len(line),2)):
    #             e = line[c]
    #             if e not in " ":
    #                 self.objects[(i,j)] = e
    #             if line[c-1] == "|": self.forbidden_transitions.add((i,j,Actions.left))
    #             if line[c+1] == "|": self.forbidden_transitions.add((i,j,Actions.right))
    #             # adding forbidden transitions if two walls side by side to make it an obstacle
    #             # i.e. | | is treated as an obstacle
    #             if line[c-1] == "|" and line[c+1] == "|":
    #                 self.forbidden_transitions.add((i-1,j,Actions.down))
    #                 self.forbidden_transitions.add((i+1,j,Actions.up))
    #             if i == 0:           self.forbidden_transitions.add((i,j,Actions.up))
    #             if i == len(map)-1:  self.forbidden_transitions.add((i,j,Actions.down))
    #     self.map_height, self.map_width = i+1, j+1 # last i and j used

    #     # while True:
    #     #     i, j = random.randrange(self.map_height), random.randrange(self.map_width)
    #     #     if (i,j) not in self.objects.keys(): break # prevent the taxi spawning on a location
    #     #     # break
    #     # self.agent = Agent(i,j,actions)
    #     self.agent = Agent(agent[0],agent[1],actions)