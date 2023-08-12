import numpy as np
import random 
import src.hyper_param as hyper_param

def argmax_rand_tie_breaker(data):
    max_value = np.max(data)
    max_indices = []
    for i in range(len(data)):
        if data[i] == max_value:
            max_indices.append(i)
    index = random.randint(0,len(max_indices)-1)
    res = max_indices[index]
    return res

class qlearning:
    def __init__(self, env, epsilon = 1, gamma = hyper_param.gamma, alpha = hyper_param.alpha, state_size = 64, action_size = 4, eps_min = hyper_param.epsilon_min, decay = hyper_param.decay):
        self._epsilon = epsilon
        self._epsilon_min = eps_min
        self._gamma = gamma
        self._alpha = alpha
        self._qtable = np.ones((state_size, action_size)) * 100
        #2d array, rows are states and cols are q-value action pairs
        self._action_size = action_size
        self._state_size = state_size
        self._decay = decay
        self._env = env
        self._index_track = {}
        self._acc_reward_data = {}
        

    def batch_train(self, batch):
        for b in batch:
            self.train(b[0], b[1], b[2], b[3], b[4])

    def train(self, state, new_state, action_index, next_action_index, reward):
        state_index = self._env.state_to_index (state)
        new_state_index = self._env.state_to_index (new_state)
        self._qtable [state_index, action_index] = self._qtable [state_index, action_index] + self._alpha * (reward 
                    + self._gamma * np.max(self._qtable[new_state_index, :])  
                    - self._qtable[state_index, action_index]) 

    def policy (self,state):
        state_index = self._env.state_to_index(state)
        if random.uniform (0,1) < self._epsilon:
            action_index = random.randint (0, self._action_size-1)
        else:
            action_index = argmax_rand_tie_breaker(self._qtable[state_index,:])
            #action_index = np.argmax(self._qtable[state_index,:])
        return action_index


    def policy_greedy (self,state):
        state_index = self._env.state_to_index(state)
        action_index = argmax_rand_tie_breaker(self._qtable[state_index,:])
        return action_index


    def decay (self):
        if self._epsilon > self._epsilon_min: self._epsilon =  self._epsilon * self._decay

    def get_max_q (self, state):
        state_index = self._env.state_to_index(state)
        max_value = max(self._qtable[state_index,:])
        return max_value
 
    def pull_qvalues_single (self, indices):
        pulled = []
        for j in range(self._action_size):
            pulled.append(0)
        for i in indices:
            for j in range(self._action_size):
                pulled[j] += self._qtable[i][j]
        for j in range (len(pulled)):
            pulled[j] = pulled[j] / len(indices)
        return pulled

    def max_pooling (self, indices):
        pulled = []
        for j in range(self._action_size):
            pulled.append(-np.inf)
        for i in indices:
            for j in range(self._action_size):
                if self._qtable[i][j] > pulled[j]: pulled[j] = self._qtable[i][j]
        return pulled

class qlearning_abs:
    def __init__(self, epsilon = 1, gamma = hyper_param.gamma, alpha = hyper_param.alpha, action_size = 4, eps_min = hyper_param.epsilon_min, decay = hyper_param.decay):
        self._epsilon = epsilon
        self._epsilon_min = eps_min
        self._gamma = gamma
        self._alpha = alpha
        self._qtable = {}
        self._action_size = action_size
        self._initial_value = 500
        self._decay = decay
        self._abstract = None
        self._acc_reward_data = {}

    def get_init_qvalues (self):  
        values = np.ones ((1,self._action_size))
        for i in range (len(values)):
            values[0][i] = self._initial_value
        return values[0]

    def batch_train (self, batch):
        for b in batch:
            self.train(b[0], b[1],b[2],b[3],b[4])

    def train(self, state, new_state, action_index, next_action_index, reward):
        self._qtable [state][0][action_index] = self._qtable [state][0][action_index] + self._alpha * (
                    reward + self._gamma *np.max(self._qtable[new_state])  
                    - self._qtable[state][0][action_index]) 


    def policy (self,state_abs):
        if random.uniform (0,1) < self._epsilon:
            action_index = random.randint (0, self._action_size-1)
        else:
            action_index = argmax_rand_tie_breaker(self._qtable[state_abs][0])
            #action_index = np.argmax(self._qtable[state_abs])
        return action_index


    def policy_fixed (self,state_abs):
        action_index = argmax_rand_tie_breaker(self._qtable[state_abs][0])
        return action_index

    def policy_rand (self,state_abs):
        action_index = random.randint (0, self._action_size-1)
        return action_index

    def update_qtable (self, state_abs):
        if state_abs not in self._qtable:
            pulled = self._abstract.bootstrap(state_abs)
            self._qtable [state_abs] = np.ones ((1,self._action_size))
            for i in range (self._action_size):
                self._qtable [state_abs][0][i] = pulled [i]

    def update_eval (self, state_abs):
        if state_abs not in self._eval:
            self._eval [state_abs] = []
            for i in range(self._action_size):
                self._eval [state_abs].append([])

    def intialize_eval (self):
        self._eval = {}
        for state in self._qtable:
            self._eval [state] = []
            for i in range (self._action_size):
                 self._eval [state].append([])
    
    def log_values (self, state, new_state, action_index, reward):
        self._eval[state][action_index].append (reward + self._gamma * np.max(self._qtable[new_state]) - self._qtable[state][0][action_index]) 


    def decay (self):
        if self._epsilon > self._epsilon_min: self._epsilon =  self._epsilon * self._decay

    def thrive_epsilon (self):
        coef = 1.03
        if self._epsilon* coef > 1: self._epsilon = 1
        else: self._epsilon = self._epsilon * coef


# class tdlambda:
#     def __init__(self, env, epsilon = 1, gamma = hyper_param.gamma, alpha = hyper_param.alpha, state_size = 64, action_size = 4, lam = hyper_param.lam, eps_min = hyper_param.epsilon_min, decay = hyper_param.decay):
#         self._epsilon = epsilon
#         self._epsilon_min = eps_min
#         self._gamma = gamma
#         self._alpha = alpha
#         self._qtable = np.ones((state_size, action_size)) * 1000
#         self._etable = np.zeros((state_size, action_size)) * 1000
#         #2d array, rows are states and cols are q-value action pairs
#         self._action_size = action_size
#         self._state_size = state_size
#         self._decay = decay
#         self._lam = lam
#         self._env = env
#         self._abstract = None

#     def batch_train (self, batch):
#         for b in batch:
#             self.train(b[0], b[1],b[2],b[3],b[4])

#     def train(self, state, new_state, action_index, next_action_index, reward):
#         a_prime = next_action_index
#         a_star = self.get_max_q_index(new_state)
#         q_sprime_astar = self._qtable[self._env.state_to_index(new_state), a_star]
#         q_s_a = self._qtable[self._env.state_to_index(state), action_index]
#         delta = reward + (self._gamma * q_sprime_astar) - q_s_a
#         self._etable[self._env.state_to_index(state), action_index] = self._etable[self._env.state_to_index(state), action_index] + 1
#         for x in range(len(self._qtable)):
#             for y in range(len(self._qtable[x])):
#                 self._qtable[x, y] = self._qtable[x,y] + (self._alpha * delta * self._etable[x,y])
#                 if a_prime == a_star:
#                     self._etable[x,y] = self._gamma * self._lam * self._etable[x,y]
#                 else:
#                     self._etable[x,y] = 0


#     def policy (self,state):
#         state_index = self._env.state_to_index(state)
#         if random.uniform (0,1) < self._epsilon:
#             action_index = random.randint (0, self._action_size-1)
#         else:
#             action_index = argmax_rand_tie_breaker(self._qtable[state_index,:])
#             #action_index = np.argmax(self._qtable[state_index,:])
#         return action_index


#     def decay (self):
#         if self._epsilon > self._epsilon_min: self._epsilon =  self._epsilon * self._decay

#     def get_max_q (self, state):
#         state_index = self._env.state_to_index(state)
#         max_value = max(self._qtable[state_index,:])
#         return max_value
    
#     def get_max_q_index(self, state):
#         state_index = self._env.state_to_index(state)
#         max_value = np.argmax(self._qtable[state_index,:])
#         return max_value

#     def pull_qvalues_single (self, indices):
#         pulled = []
#         for j in range(self._action_size):
#             pulled.append(0)
#         for i in indices:
#             for j in range(self._action_size):
#                 pulled[j] += self._qtable[i][j]
#         for j in range (len(pulled)):
#             pulled[j] = pulled[j] / len(indices)
#         return pulled


# class tdlambda_abs:
#     def __init__(self, epsilon = 1, gamma = hyper_param.gamma, alpha = hyper_param.alpha, state_size = 64, action_size = 4, lam = hyper_param.lam, eps_min = hyper_param.epsilon_min, decay = hyper_param.decay):
#         self._epsilon = epsilon
#         self._epsilon_min = eps_min
#         self._gamma = gamma
#         self._alpha = alpha
#         self._qtable = {}
#         self._etable = {}        
#         self._action_size = action_size
#         self._state_size = state_size
#         self._decay = decay
#         self._initial_value = 500
#         self._lam = lam
#         self._acc_reward_data = {}
        

#     def batch_train (self, batch):
#         for b in batch:
#             self.train(b[0], b[1],b[2],b[3],b[4])

#     def train(self, state, new_state, action_index, next_action_index, reward):
#         a_prime = next_action_index
#         a_star = self.policy(new_state)
#         q_sprime_astar = self._qtable[new_state][0][a_star]
#         q_s_a = self._qtable[state][0] [action_index]
#         delta = reward + (self._gamma * q_sprime_astar) - q_s_a
#         self._etable[state][0][action_index] = self._etable[state][0][action_index] + 1
#         for st in self._qtable.keys():
#             for ac in range(len(self._qtable[st][0])):
#                 self._qtable[st][0][ac] = self._qtable[st][0][ac] + (self._alpha * delta * self._etable[st][0][ac])
#                 if a_prime == a_star:
#                     self._etable[st][0][ac] = self._gamma * self._lam * self._etable[st][0][ac]
#                 else:
#                     self._etable[st][0][ac] = 0


#     def policy (self,state_abs):
#         if random.uniform (0,1) < self._epsilon:
#             action_index = random.randint (0, self._action_size-1)
#         else:
#             action_index = argmax_rand_tie_breaker(self._qtable[state_abs][0])
#             #action_index = np.argmax(self._qtable[state_abs])
#         return action_index

#     def update_qtable (self, state_abs):
#         if state_abs not in self._qtable:
#             pulled = self._abstract.bootstrap(state_abs)
#             self._qtable [state_abs] = np.ones ((1,self._action_size))
#             self._etable [state_abs] = np.zeros ((1,self._action_size))
#             for i in range (self._action_size):
#                 self._qtable [state_abs][0][i] = pulled [i]
                

#     def update_eval (self, state_abs):
#         if state_abs not in self._eval:
#             self._eval [state_abs] = []
#             for i in range(self._action_size):
#                 self._eval [state_abs].append([])

#     def intialize_eval (self):
#         self._eval = {}
#         for state in self._qtable:
#             self._eval [state] = []
#             for i in range (self._action_size):
#                  self._eval [state].append([])
    
#     def log_values (self, state, new_state, action_index, reward):
#         self._eval[state][action_index].append (reward + self._gamma * np.max(self._qtable[new_state]) - self._qtable[state][0][action_index]) 


#     def decay (self):
#         if self._epsilon > self._epsilon_min: self._epsilon =  self._epsilon * self._decay

#     def thrive_epsilon (self):
#         coef = 1.03
#         if self._epsilon* coef > 1: self._epsilon = 1
#         else: self._epsilon = self._epsilon * coef