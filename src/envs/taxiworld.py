import numpy as np
import random 
import math
import src.envs.map_maker as map_maker

class Taxi_Domain:
    def __init__(self, env_name, start, passenger_n):
        self._visited = []
        self._action_space = ['up','down','left','right', 'pickup', 'dropoff']
        self._action_size = len(self._action_space)
        self._maze = map_maker.get_map(env_name)
        self._dimension = self._maze.shape
        self._locations = {0:[]}
        self._passenger_n = passenger_n
        for y in range(self._dimension[0]):
            for x in range(self._dimension[1]):
                if self._maze[y,x] >= 2:
                    self._locations[self._maze[y,x]-1] = [y,x]
        pick_up_loc, drop_off_loc = self.choose_pickup_locations (self._passenger_n)
        self._state_size = int(self._dimension[0] * self._dimension[1] * math.pow(len (self._locations), passenger_n) * (len(self._locations)-1))
        self._start = start
        self._current_loc = self._start
        self._action_size = len(self._action_space)
        self._action_probs = {0:[2,3], 1:[2,3], 2:[0,1], 3:[0,1], 4:[4,4], 5:[5,5]}
        self._stoch_prob = 0.8
        self._visit_map = np.zeros_like(self._maze)
        self._passenger_loc = pick_up_loc
        self._drop_loc = drop_off_loc
        self._n_state_variables = 3 + len(self._passenger_loc) 
        self._state_ranges, self._vars_split_allowed = self.get_state_ranges (passenger_n)
        self._extra_info = [self._locations, drop_off_loc]
        self._taxi_capacity = 1

    def get_state_ranges (self, passenger_n):
        ranges = [ (0,self._dimension[0]), (0,self._dimension[1])]  # taxi location
        for i in range (passenger_n): ranges = ranges + [(0,5)]     # passenger ids
        ranges = ranges + [(1,5)]                                   # goal location id
        vars_split_allowed = [1 for i in range(len(ranges))]
        return ranges, vars_split_allowed

    def choose_pickup_locations (self, passenger_n):
        pickup_locs = []
        while (len (pickup_locs)<passenger_n):
            temp_loc = random.randint(1, len(self._locations)-1)
            if temp_loc not in pickup_locs: pickup_locs.append(temp_loc)
        drop_off_loc = pickup_locs[0]
        while drop_off_loc in pickup_locs:
            drop_off_loc = random.randint(1, len(self._locations)-1)
        return pickup_locs, drop_off_loc


    def reset_visited (self):
        self._visit_map = np.zeros_like(self._maze)

    # state to state_index
    def state_to_index (self, state):
        taxi_loc = (state[0],state[1])
        p_loc = state[2]
        drop_loc = state[3]
        grid_size = self._dimension[0]*self._dimension[1]
        output_index = (taxi_loc[0]*self._dimension[0] + taxi_loc[1] ) + ((p_loc-1)* grid_size) + ((len(self._locations)) * grid_size * (drop_loc-2))
        return output_index


    def action_stochastic (self, action_index):
        if random.uniform (0,1) > self._stoch_prob:
            if random.uniform (0,1) > 0.5 : 
                action_index_stoch = self._action_probs[action_index][0]
            else: action_index_stoch = self._action_probs[action_index][1]
        else: action_index_stoch = action_index
        return action_index_stoch

    def there_is_passenger (self, current_location):
        for l in self._passenger_loc:
            if current_location == self._locations [l] and l != self._drop_loc: return True
        return False

    def update_passenger_location (self, current_location, action):
        if action == 'pickup':
            for l in self._passenger_loc:
                if current_location == self._locations [l]:
                    self._passenger_loc.remove(l)
                    self._passenger_loc.append(0)

        if action == 'dropoff':
            temp = []
            for l in self._passenger_loc:
                if l == 0: temp.append (self._drop_loc)
                else: temp.append (l)
            self._passenger_loc = temp
                
    def step (self, action_index_input):
        r_move = -1
        r_wrong_pickup, r_wrong_dropoff = -100, -100
        r_correct_dropoff, r_correct_pickup = 500, 0
        [a,b] = self._current_loc
        reward  = None # the episode's reward (-100 for pitfall, 0 for reaching the goal, and -1 otherwise)
        flag = False # termination flag is true if the agent falls in a pitfall or reaches to the goal
        flag_succ = False
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
        elif action == 'pickup':
            next_loc = self._current_loc
        elif action == 'dropoff':
            next_loc = self._current_loc

        if self.in_bound (next_loc) == False:
            reward = r_move
            next_loc = self._current_loc
            flag = False
            next_passenger_location = self._passenger_loc
            state = [next_loc[0], next_loc[1]] + self._passenger_loc + [self._drop_loc]
            return state, reward, flag, flag_succ
        else:
            if self._maze [self._current_loc[0]] [self._current_loc[1]] == -1:
                flag_pitfall = True 
                reward = -1000
                flag = True
                next_passenger_location = self._passenger_loc
                state = [next_loc[0], next_loc[1]] + self._passenger_loc + [self._drop_loc]
                return state, reward, flag, flag_succ
            elif self._maze [next_loc[0]] [next_loc[1]] == 1:
                next_loc = self._current_loc
                reward = r_move
                flag = False
                next_passenger_location = self._passenger_loc
                state = [next_loc[0], next_loc[1]] + self._passenger_loc + [self._drop_loc]
                return state, reward, flag, flag_succ
            elif next_loc == self._current_loc:
                if action == 'pickup':
                    if self.there_is_passenger (self._current_loc) and self.taxi_is_not_full():
                        reward = r_correct_pickup
                        flag = False
                        self.update_passenger_location (self._current_loc, action)
                        state = [next_loc[0], next_loc[1]] + self._passenger_loc + [self._drop_loc]
                        return state, reward, flag, flag_succ
                    else:
                        reward = r_wrong_dropoff
                        flag = False
                        state = [next_loc[0], next_loc[1]] + self._passenger_loc + [self._drop_loc]
                        return state, reward, flag, flag_succ
                elif action == 'dropoff':
                    if self.any_passenger_in_taxi() and self._current_loc == self._locations[self._drop_loc]:
                        reward = r_correct_dropoff
                        self.update_passenger_location (self._current_loc, action)
                        state = [next_loc[0], next_loc[1]] + self._passenger_loc + [self._drop_loc]
                        if self.all_passengers_at_destination(): flag, flag_succ = True, True
                        return state, reward, flag, flag_succ
                    else:
                        reward = r_wrong_dropoff
                        flag = False
                        state = [next_loc[0], next_loc[1]] + self._passenger_loc + [self._drop_loc]
                        return state, reward, flag, flag_succ

            else:
                self._current_loc = next_loc
                reward = r_move
                state = state = [next_loc[0], next_loc[1]] + self._passenger_loc + [self._drop_loc]
                return state, reward, flag, flag_succ

    def taxi_is_not_full(self):
        count = 0
        for l in self._passenger_loc:
            if l == 0:
                count += 1
        if count >= self._taxi_capacity: return False
        else: return True

    def any_passenger_in_taxi(self):
        for p in self._passenger_loc:
            if p == 0: return True
        return False

    def all_passengers_at_destination (self):
        for p in self._passenger_loc: 
            if p != self._drop_loc: return False
        return True

    def reset (self):
        self._passenger_loc, self._drop_loc = self.choose_pickup_locations (self._passenger_n)
        self._current_loc = self._start
        state = [self._current_loc[0], self._current_loc[1]] + self._passenger_loc + [self._drop_loc]
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



    def update_visited (self, state):
        flag = True
        for i in self._visited:
            if state == i: flag = False
        if flag: self._visited.append(state)