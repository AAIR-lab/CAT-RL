import numpy as np
import random 

class Office_Domain:
    def __init__(self, map_name):
        self._visited = []
        self._action_space = ['up','down','left','right']
        self._action_size = len(self._action_space)

        if map_name == "office_18x18_map1":
            self.load_18x18_map()   
        elif map_name == "office_27x27_map1":
            self.load_27x27_map()              
        if map_name == "office_36x36_map1":
            self.load_36x36_map() 
        elif map_name == "office_45x45_map1":
            self.load_45x45_map()      
        elif map_name == "office_54x54_map1":
            self.load_54x54_map()   
            
        self._coffee_locs = self._init_coffee_locs.copy()
        self._mail_locs = self._init_mail_locs.copy()
        self._has_coffee = 0
        self._has_mail = 0
        self._state_size = ((self._dimension[0] * self._dimension[1]) * 4) # 4 possible combinations of has_coffee, has_mail
        self._current_loc = self._start
        self._action_size = len(self._action_space)
        self._action_probs = {0:[2,3], 1:[2,3], 2:[0,1], 3:[0,1]}
        self._stoch_prob = 0.8
        self._visit_map = np.zeros_like(self._maze)
        self._n_state_variables = 4
        self._state_ranges = [
            (0,self._dimension[0]), # y variable
            (0,self._dimension[1]),
            (0,2),
            (0,2)
        ]
        self._vars_split_allowed = [1 for i in range(len(self._state_ranges))]
        
    def load_54x54_map(self):
        # Creating the map
        self._dimension = (54,54)
        self._maze = np.zeros((self._dimension[0], self._dimension[1]))

        # Adding the agent
        self._start = (2,1)
        self._init_coffee_locs = [(12,18)]
        self._init_mail_locs = [(22,19)]
        self._office_loc = (30,33)
        self._rooms = {'a': (1,1), 'b': (52,1), 'c':(52,52), 'd':(1,52)}

        # Adding walls
        self.forbidden_transitions = set()
        for x in range(54):
           for y in [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51]:
               self.forbidden_transitions.add((x,y,self._action_space[1]))
               self.forbidden_transitions.add((x,y+2,self._action_space[0]))
        for y in range(54):
           for x in [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51]:
               self.forbidden_transitions.add((x,y,self._action_space[2]))
               self.forbidden_transitions.add((x+2,y,self._action_space[3]))
        # adding 'doors'
        for y in [1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52]:
           for x in [2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50]:
               self.forbidden_transitions.remove((x,y,self._action_space[3]))
               self.forbidden_transitions.remove((x+1,y,self._action_space[2]))
        for x in [1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52]:
            for y in [2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50]:
                self.forbidden_transitions.remove((x,y,self._action_space[0]))
                self.forbidden_transitions.remove((x,y+1,self._action_space[1]))  

    def load_45x45_map(self):
        # Creating the map
        self._dimension = (45,45)
        self._maze = np.zeros((self._dimension[0], self._dimension[1]))
        self._dimension = self._maze.shape

        # Adding the agent
        self._start = (2,1)
        self._init_coffee_locs = [(8,14)]
        self._init_mail_locs = [(18,15)]
        self._office_loc = (26,29)
        self._rooms = {'a': (1,1), 'b': (43,1), 'c':(43,43), 'd':(1,43)}

        # Adding walls
        self.forbidden_transitions = set()
        for x in range(45):
           for y in [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42]:
               self.forbidden_transitions.add((x,y,self._action_space[1]))
               self.forbidden_transitions.add((x,y+2,self._action_space[0]))
        for y in range(45):
           for x in [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42]:
               self.forbidden_transitions.add((x,y,self._action_space[2]))
               self.forbidden_transitions.add((x+2,y,self._action_space[3]))
        # adding 'doors'
        for y in [1,4,7,10,13,16,19,22,25,28,31,34,37,40,43]:
           for x in [2,5,8,11,14,17,20,23,26,29,32,35,38,41]:
               self.forbidden_transitions.remove((x,y,self._action_space[3]))
               self.forbidden_transitions.remove((x+1,y,self._action_space[2]))
        for x in [1,4,7,10,13,16,19,22,25,28,31,34,37,40,43]:
            for y in [2,5,8,11,14,17,20,23,26,29,32,35,38,41]:
                self.forbidden_transitions.remove((x,y,self._action_space[0]))
                self.forbidden_transitions.remove((x,y+1,self._action_space[1]))  
         #for x in [1,4,7,10]:
         #   self.forbidden_transitions.remove((x,5,self._action_space[0]))
         #   self.forbidden_transitions.remove((x,6,self._action_space[1]))
         #for x in [1,10]:
         #   self.forbidden_transitions.remove((x,2,self._action_space[0]))
         #   self.forbidden_transitions.remove((x,3,self._action_space[1]))   
         
    def load_36x36_map(self):
        # Creating the map
        self._dimension = (36,36)
        self._maze = np.zeros((self._dimension[0], self._dimension[1]))
        self._dimension = self._maze.shape

        # Adding the agent
        self._start = (2,1)
        self._init_coffee_locs = [(8,14)]
        self._init_mail_locs = [(11,8)]
        self._office_loc = (17,20)
        self._rooms = {'a': (1,1), 'b': (34,1), 'c':(34,34), 'd':(1,34)}

        # Adding walls
        self.forbidden_transitions = set()
        for x in range(36):
           for y in [0,3,6,9,12,15,18,21,24,27,30,33]:
               self.forbidden_transitions.add((x,y,self._action_space[1]))
               self.forbidden_transitions.add((x,y+2,self._action_space[0]))
        for y in range(36):
           for x in [0,3,6,9,12,15,18,21,24,27,30,33]:
               self.forbidden_transitions.add((x,y,self._action_space[2]))
               self.forbidden_transitions.add((x+2,y,self._action_space[3]))
        # adding 'doors'
        for y in [1,4,7,10,13,16,19,22,25,28,31,34]:
           for x in [2,5,8,11,14,17,20,23,26,29,32]:
               self.forbidden_transitions.remove((x,y,self._action_space[3]))
               self.forbidden_transitions.remove((x+1,y,self._action_space[2]))
        for x in [1,4,7,10,13,16,19,22,25,28,31,34]:
            for y in [2,5,8,11,14,17,20,23,26,29,32]:
                self.forbidden_transitions.remove((x,y,self._action_space[0]))
                self.forbidden_transitions.remove((x,y+1,self._action_space[1]))  
         #for x in [1,4,7,10]:
         #   self.forbidden_transitions.remove((x,5,self._action_space[0]))
         #   self.forbidden_transitions.remove((x,6,self._action_space[1]))
         #for x in [1,10]:
         #   self.forbidden_transitions.remove((x,2,self._action_space[0]))
         #   self.forbidden_transitions.remove((x,3,self._action_space[1])) 
             
    def load_27x27_map(self):
        # Creating the map
        self._dimension = (27,27)
        self._maze = np.zeros((self._dimension[0], self._dimension[1]))

        # Adding the agent
        self._start = (2,1)
        self._init_coffee_locs = [(6,12)]
        self._init_mail_locs = [(10,8)]
        self._office_loc = (15,18)
        self._rooms = {'a': (1,1), 'b': (25,1), 'c':(25,25), 'd':(1,25)}

        # Adding walls
        self.forbidden_transitions = set()
        for x in range(27):
           for y in [0,3,6,9,12,15,18,21,24]:
               self.forbidden_transitions.add((x,y,self._action_space[1]))
               self.forbidden_transitions.add((x,y+2,self._action_space[0]))
        for y in range(27):
           for x in [0,3,6,9,12,15,18,21,24]:
               self.forbidden_transitions.add((x,y,self._action_space[2]))
               self.forbidden_transitions.add((x+2,y,self._action_space[3]))
        # adding 'doors'
        for y in [1,4,7,10,13,16,19,22,25]:
           for x in [2,5,8,11,14,17,20,23]:
               self.forbidden_transitions.remove((x,y,self._action_space[3]))
               self.forbidden_transitions.remove((x+1,y,self._action_space[2]))
        for x in [1,4,7,10,13,16,19,22,25]:
            for y in [2,5,8,11,14,17,20,23]:
                self.forbidden_transitions.remove((x,y,self._action_space[0]))
                self.forbidden_transitions.remove((x,y+1,self._action_space[1]))   
         
    def load_18x18_map(self):
        # Creating the map
        self._dimension = (18,18)
        self._maze = np.zeros((self._dimension[0], self._dimension[1]))

        # Adding the agent
        self._start = (2,1)
        self._init_coffee_locs = [(6,12)]
        self._init_mail_locs = [(8,6)]
        self._office_loc = (12,15)
        self._rooms = {'a': (1,1), 'b': (16,1), 'c':(16,16), 'd':(1,16)}

        # Adding walls
        self.forbidden_transitions = set()
        for x in range(18):
           for y in [0,3,6,9,12,15]:
               self.forbidden_transitions.add((x,y,self._action_space[1]))
               self.forbidden_transitions.add((x,y+2,self._action_space[0]))
        for y in range(18):
           for x in [0,3,6,9,12,15]:
               self.forbidden_transitions.add((x,y,self._action_space[2]))
               self.forbidden_transitions.add((x+2,y,self._action_space[3]))
        # adding 'doors'
        for y in [1,4,7,10,13,16]:
           for x in [2,5,8,11,14]:
               self.forbidden_transitions.remove((x,y,self._action_space[3]))
               self.forbidden_transitions.remove((x+1,y,self._action_space[2]))
        for x in [1,4,7,10,13,16]:
            for y in [2,5,8,11,14]:
                self.forbidden_transitions.remove((x,y,self._action_space[0]))
                self.forbidden_transitions.remove((x,y+1,self._action_space[1]))   

    def reset_visited (self):
        self._visit_map = np.zeros_like(self._maze)

    def update_visited (self, state):
        flag = True
        for i in self._visited:
            if state == i: flag = False
        if flag: self._visited.append(state)

    # state to state_index
    def state_to_index (self, state):
        current_loc = [state[0],state[1]]
        has_coffee = state[2]
        has_mail = state[3]
        state_coffee_mail = has_coffee + (has_mail*2)

        x_index = current_loc[0] * self._dimension[1] * 4
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
        if (a,b,action) not in self.forbidden_transitions:
            if action == 'up':
                a -= 1
            elif action == 'down':
                a += 1
            elif action == 'left':
                b -= 1
            elif action == 'right':
                b += 1
        next_loc = tuple([a,b])

        if self.in_bound(next_loc):
            self._current_loc = next_loc
        else:
            next_loc = self._current_loc

        if self._has_coffee and self._has_mail and next_loc == self._office_loc:
            reward = 1000
            flag = True
            flag_succ = True
            state = [next_loc[0],next_loc[1], self._has_coffee, self._has_mail]
            # return state, reward, flag, flag_succ, flag_pitfall
            return state, reward, flag, flag_succ
        else:
            reward = 0
            # reward = -1
            flag = False
            flag_succ = False
            if not self._has_coffee and next_loc in self._coffee_locs:
                self._has_coffee = 1
                # reward = 10
            elif not self._has_mail and next_loc in self._mail_locs:
                self._has_mail = 1
                # reward = 10
            state = [next_loc[0],next_loc[1], self._has_coffee, self._has_mail]
            # return state, reward, flag, flag_succ, flag_pitfall
            return state, reward, flag, flag_succ

    def reset (self):
        self._current_loc = self._start
        self._has_coffee = 0
        self._has_mail = 0
        state = [self._current_loc[0],self._current_loc[1], self._has_coffee, self._has_mail]
        return state

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
