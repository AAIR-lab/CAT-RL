import gym
import numpy as np
from gym import spaces
import random

MAP = [
    "+---------+",
    "|R: : : :G|",
    "| : : : : |",
    "| : : : : |",
    "| : : : : |",
    "|Y: : : :B|",
    "+---------+",
]

class TaxiWorldActions:
    EAST = 0
    WEST = 1
    NORTH = 2
    SOUTH = 3
    PICK = 4
    DROP = 5

class TaxiWorldState:
    def __init__(self, taxi_loc, pass_idxs, dest_idx):
        self.taxi_loc = taxi_loc
        self.pass_idxs = pass_idxs # [0, 1, 2, 3]
        self.dest_idx = dest_idx   # 0, 1, 2, 3
        self._cached_hash = None
        self.__hash__()

    def __str__(self):
        string = "("
        string += "("+str(self.taxi_loc[0])+","+str(self.taxi_loc[1])+"),"
        string += str(self.pass_idxs)+","
        string += str(self.dest_idx)+")"
        return string

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, TaxiWorldState):
            return False
        else:
            return self._cached_hash == other._cached_hash

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        self._cached_hash = hash((self.taxi_loc, self.pass_idxs, self.dest_idx))
        return self._cached_hash

class TaxiworldEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size, start_loc, dest_loc, step_max):
        super(TaxiworldEnv, self).__init__()

        self.start_loc = start_loc
        self.dest_loc = dest_loc
        self.num_rows = grid_size[0]
        self.num_columns = grid_size[1]
        num_actions = 6
        num_states = 100000
        self.num_passengers = 1
        self.stochastic_prob = 0.8

        self.step_max = step_max
        self.steps = 0
        self.done = False
        self.success = False
        self.num_episodes = 0
        self.total_reward = 0

        self.idx_to_loc = {0: (0,0), 1: (0,self.num_columns-1), 2: (self.num_rows-1,0), 3: (self.num_rows-1,self.num_columns-1)}
        self.locs_to_symbol = {(0,0): 'R', (0,self.num_columns-1):'G', (self.num_rows-1,0):'Y', (self.num_rows-1,self.num_columns-1): 'B'}

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)
        self.id_to_action = {0: "East", 1: "West", 2: "North", 3:"South", 4:"Pickup", 5:"Drop"}
        self.action_probs = {0:[2,3], 1:[2,3], 2:[0,1], 3:[0,1], 4:[4,4], 5:[5,5]}

    def actual_action_due_to_stochasticity(self, action):
        actual_action = action
        if random.uniform(0,1) > self.stochastic_prob:
            if random.uniform(0,1) > 0.5: 
                actual_action = self.action_probs[action][0]
            else: 
                actual_action = self.action_probs[action][1]
        return actual_action

    def out_of_bounds(self, taxi_loc):
        if taxi_loc[0] >= self.num_rows or taxi_loc[0] < 0:
            return True
        if taxi_loc[1] >= self.num_columns or taxi_loc[1] < 0:
            return True
        return False

    def has_delivered_all_passengers(self, pass_idxs):
        for id in pass_idxs: 
            if id != self.state.dest_idx: 
                return False
        return True

    def there_is_passenger(self, taxi_loc, pass_idxs):
        for pass_idx in pass_idxs:
            if self.idx_to_loc[pass_idx] == taxi_loc:
                return True
        return False

    def taxi_is_empty(self, pass_idxs):
        if 4 not in pass_idxs:
            return True
        return False

    def is_taxi_at_loc_and_has_passenger(self, taxi_loc, pass_idxs):
        for pass_idx in pass_idxs:
            if pass_idx == 4 and taxi_loc in self.idx_to_loc:
                return True
        return False

    def is_taxi_at_dest_and_has_passenger(self, taxi_loc, pass_idxs, dest_idx):
        for pass_idx in pass_idxs:
            if pass_idx == 4 and taxi_loc == self.idx_to_loc[dest_idx]:
                return True
        return False

    def step(self, action):
        taxi_row, taxi_col, pass_idxs = self.decode(self.state_id)

        reward = -1  # default reward when there is no pickup/dropoff
        self.done = False
        taxi_loc = (taxi_row, taxi_col)
        max_row = self.num_rows - 1
        max_col = self.num_columns - 1
        new_pass_idxs = pass_idxs

        action = self.actual_action_due_to_stochasticity(action)
        if action == TaxiWorldActions.EAST:
            new_row = taxi_row
            new_col = taxi_col + 1
        elif action == TaxiWorldActions.WEST:
            new_row = taxi_row
            new_col = taxi_col - 1
        elif action == TaxiWorldActions.NORTH:
            new_row = taxi_row - 1
            new_col = taxi_col
        elif action == TaxiWorldActions.SOUTH:
            new_row = taxi_row + 1
            new_col = taxi_col
        elif action == TaxiWorldActions.PICK:
            new_row, new_col = taxi_row, taxi_col
        elif action == TaxiWorldActions.DROP:
            new_row, new_col = taxi_row, taxi_col

        # print("action:",self.id_to_action[action])
        if self.out_of_bounds((new_row,new_col)):
            new_row, new_col = taxi_row, taxi_col
            self.done = False
            self.success = False
            reward = -1
        elif action == TaxiWorldActions.PICK:
            new_pass_idxs = list()
            self.done = False
            self.success = False
            if self.taxi_is_empty(pass_idxs) and self.there_is_passenger(taxi_loc, pass_idxs):
                for pass_idx in pass_idxs:
                    if taxi_loc == self.idx_to_loc[pass_idx] and self.taxi_is_empty(pass_idxs):
                        # pick up when passenger at taxi's loc and taxi not full
                        new_pass_idx = 4
                        new_pass_idxs.append(new_pass_idx)
                    else:  
                        # passenger not at location or taxi is full
                        new_pass_idxs.append(pass_idx)
                reward = 0
            else:
                # illegal pickup
                new_pass_idxs = pass_idxs
                reward = -100
        elif action == TaxiWorldActions.DROP:
            new_pass_idxs = list()
            if self.is_taxi_at_dest_and_has_passenger(taxi_loc, pass_idxs, self.dest_idx):
                # dropoff when taxi at destination and has passenger
                for pass_idx in pass_idxs:
                    if taxi_loc == self.idx_to_loc[self.dest_idx] and pass_idx == 4:
                        new_pass_idx = self.dest_idx
                        reward = 500 
                        new_pass_idxs.append(self.dest_idx)
                        if self.has_delivered_all_passengers(new_pass_idxs): 
                            self.done = True
                            self.success = True      
                    else:
                        new_pass_idxs.append(pass_idx)
            else:  
                # illegal dropoff at wrong location
                new_pass_idxs = pass_idxs
                reward = -100
                self.done = False
                self.success = False

        self.state = TaxiWorldState((new_row, new_col), tuple(new_pass_idxs), self.dest_idx)
        self.state_id = self.encode(self.state)
        
        self.steps += 1
        if self.steps == self.step_max:
            self.done = True
        if self.done:
            # self.render()
            self.num_episodes += 1
        self.total_reward += reward
        # self.render()

        info = {}
        info["done"] = self.done
        info["succ"] = self.success
        info["reward"] = self.total_reward
        info["steps"] = self.steps
        info["num_episodes"] = self.num_episodes
        return self.state_id, reward, self.done, info

    def encode(self, state):
        taxi_row = state.taxi_loc[0]
        taxi_col = state.taxi_loc[1]
        pass_loc = state.pass_idxs[0]
        x_index = taxi_row * self.num_columns * 5
        y_index = taxi_col * 5 
        p_index = pass_loc
        index = x_index + y_index + p_index
        return int(index)

    def decode(self, i):
        out = []
        z_index = i % 5
        out.append(z_index)
        inter_index = (i - z_index) // 5
        y_index = inter_index % self.num_columns
        out.append(y_index)
        x_index = inter_index // self.num_columns
        out.append(x_index)
        decoded_state = list(reversed(out))
        decoded_state[2] = [decoded_state[2]]
        return decoded_state[0], decoded_state[1], decoded_state[2]

    def reset(self):
        self.steps = 0
        self.total_reward = 0
        self.done = False
        self.success = False 

        taxi_row = np.random.choice(list(range(self.num_rows)))
        taxi_col = np.random.choice(list(range(self.num_columns)))
        while (taxi_row, taxi_col) in self.idx_to_loc:
            taxi_row = np.random.choice(list(range(self.num_rows)))
            taxi_col = np.random.choice(list(range(self.num_columns)))
        # pass_idxs = list()
        # while len(pass_idxs) != self.num_passengers:
        #     pass_idxs.append(np.random.choice([0,1,2]))
        pass_idxs = [0]
        self.dest_idx = 3
        self.state = TaxiWorldState((taxi_row, taxi_col), tuple(pass_idxs), self.dest_idx)
        self.state_id = self.encode(self.state)
        # self.render()
        return self.state_id

    def render(self, mode='human'):
        print(self.state.__str__())

    def close (self):
        pass
