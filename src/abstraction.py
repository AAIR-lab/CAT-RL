import numpy as np
from copy import copy, deepcopy
import math
import itertools
from sklearn.cluster import KMeans

from src.abs_tree import *
from src.results import Results 
from src.learning import *

class Abstraction:
    def __init__(self, env, n_state_variables, state_variable_ranges, n_action_size, agent_con, agent, k_cap = hyper_param.k_cap, boot_type = 'from_init'):
        self._env = env
        if hyper_param.bootstrap == 'from_concrete': 
            self._maze_abstract = np.chararray(self._env._maze.shape, itemsize = 100 ,unicode=True)
            self._maze_abstract[:] = ''
        self._n_states = n_state_variables
        self._state_ranges = state_variable_ranges
        self._split = []
        self._n_abstract_states = 0
        self._init_abs_level = 1
        self.initialize_abstraction()
        self._n_action_size = n_action_size
        self._agent_concrete = agent_con
        self._agent = agent
        self._root_name = self.split_to_abs_state(self._state_ranges)
        self._tree = Abstraction_Tree (root_split = self._split, root_abs_state = self._root_name)
        self.initialize_tree()
        self.update_n_abstract_state ()
        if hyper_param.bootstrap == 'from_concrete': self.update_abstract_maze ()
        self._gen_pool = []
        self._current_counter_examples = []
        self._k = k_cap
        self._unstable_cap = []
        self._bootstrap_mode = boot_type #'from_concrete' #'from_ancestor' #  #'from_init'      
        self._e_frequency = 400
        self._e_duration = 100
        self._e_flag =  False # evaluation phase flag
        self._re_flag = False # refinement phase flag
        self._e_counter = 0
        self._f_counter = 0

    def split_to_abs_state (self, split):
        abs_state = []
        for sv in split:
            abs_state.append( str(sv[0]) + "," + str(sv[1]))
        return tuple(abs_state)

    def initialize_abstraction(self):
        for i in range (self._n_states):
            midpoint = (self._state_ranges[i][1] - self._state_ranges[i][0])/2 + self._state_ranges[i][0]
            if not midpoint - int(midpoint) == 0: midpoint = math.ceil(midpoint)
            min_value = self._state_ranges[i][0]
            max_value = self._state_ranges[i][1]
            self._split.append ([min_value, int(midpoint), max_value])
      
    def split_to_all_state_values (self, split):
        state_values = []
        for sp in split:
            temp = []
            for i in range (len(sp)-1):
                value = str (sp[i]) + ',' + str(sp[i + 1])
                temp.append(value)
            state_values.append(temp)
        return state_values

    def initialize_tree (self):
        root = self._tree._root
        if self._init_abs_level == 1:
            new_state_values = self.split_to_all_state_values(root._split)
            new_leaf_nodes = list(itertools.product(*new_state_values))
            for s in new_leaf_nodes:
                new_node = self._tree.add_node (split = [], abs_state = s)
                new_node._parent = self._tree._root
                self._tree._root._child.append(new_node)
                self._tree._leaves[s] = new_node
            del self._tree._leaves[self._root_name] # the no longer is a leaf node
        elif self._init_abs_level == 0:
            new_leaf_nodes = root 

    def state_to_split_indices (self, state, split):
        indices = []
        for i in range(len(state)): 
            s = state[i].split(",")
            s = [int(s[0]), int(s[1])]
            for j in range (len(split[i])-1):
                if s == split[i] [j:j+2]:
                    indices.append(j)
                    break
        return indices
    
    def state(self, state_con):
        try:
            state_abstract = self.state_recursive(state_con, self._tree._root) 
        except Exception as e:
            print(str(e))
            print("abstract state not in CAT")
        return state_abstract 

    # def state_recursive(self, state_con, start_node):
    #     found = False
    #     result = None
    #     abstract_state = self.con_state_to_abs (state_con, start_node._split)   
    #     flag = False
    #     for n in start_node._child:
    #         if abstract_state == n._state:
    #             flag = True
    #             temp_node = n
    #             if len(n._child) == 0:
    #                 found = True 
    #                 result = abstract_state
    #     if found: return result
    #     else:
    #         if not flag:
    #             print (state_con)
    #         return self.state_recursive(state_con, temp_node)

    def state_recursive(self, state_con, start_node):
        found = False
        result = None
        abstract_state = self.con_state_to_abs (state_con, start_node._split)   
        flag = False
        for n in start_node._child:
            if abstract_state == n._state:
                flag = True
                temp_node = n
                if len(n._child) == 0:
                    found = True 
                    result = abstract_state
        if found: return result
        else:
            if not flag:
                print (state_con)
            return self.state_recursive(state_con, temp_node)

    def con_state_to_abs (self, state_con, split):
        state = []
        for i in range(len(state_con)):
            for j in range (len(split[i]) -1):
                if state_con[i] >= split[i][j] and state_con[i] < split[i][j+1]:
                    state.append(str(split[i][j]) + ',' + str(split[i][j+1]) )
                    break
        state = tuple(state)
        if len(state) == len(state_con): return state
        else: return None  

    def split_abs_state_wrs (self, abs_state, wrt_variable_index):
        abs_state_1 = list(abs_state)
        abs_state_2 = list(abs_state)
        state_value = abs_state[wrt_variable_index]
        
        interval = state_value.split(",")
        for i in range(2): interval[i] = int(interval[i])
        midpoint = int((interval[1] - interval[0])/2) + interval[0] 
        interval1 = str(interval[0]) + "," + str(midpoint)
        interval2 = str(midpoint) + "," + str(interval[1])  
        abs_state_1[wrt_variable_index] = interval1
        abs_state_2[wrt_variable_index] = interval2
        return [(*abs_state_1, ), (*abs_state_2, )]

    def qtable_variation(self, abs_state, wrt_variable_index):
        unstable_state_expanded = self.split_abs_state_wrs(abs_state, wrt_variable_index)
        q_values = []
        for item in unstable_state_expanded: q_values.append (np.average(self.bootstrap(item)))
        variation = 100* (max(q_values) - min(q_values)) /(min(q_values))
        return variation

    def get_to_split_variables2(self, unstable_state):
        vars = []
        for k in range (len(unstable_state)):
            if self.is_refinable(unstable_state[k]):
                vars.append(self.qtable_variation(unstable_state, k))
            else: vars.append(0) 
        vector = []
        for i in range (self._n_states):
            vector.append(0)
        for i in range (self._n_states):
            if vars[i] > 0: vector[i] = 1
        return vector

    def get_to_split_variables(self, unstable_state):
        vector = []
        for i in range (self._n_states):
            vector.append(1)
        return vector

    def is_refinable (self, interval):
        interval = interval.split(",")
        if int(interval[1]) - int(interval[0]) > 1: return True
        else: return False

    def bootstrap(self, state):
        if self._bootstrap_mode == 'from_concrete':
            concrete_states = self.possile_concrete_state (state)
            concrete_qtable_indices = self.find_concrete_qtable_indices (concrete_states)
            pulled = self._agent_concrete.pull_qvalues_single (concrete_qtable_indices)
            #pulled = self._agent_concrete.max_pooling (concrete_qtable_indices)
        elif self._bootstrap_mode == 'from_ancestor':
            if state not in self._agent._qtable:
                pulled = []
                for i in range (self._n_action_size):
                    pulled.append(self._agent._initial_value)
            else:
                pulled = self._agent._qtable[state]
        elif self._bootstrap_mode == 'from_init':
            pulled = self._agent.get_init_qvalues()
        return pulled

    def possile_concrete_state (self, state):
        possible_values = []
        for i in range (len(state)):
            s = state[i].split(",")
            temp = []
            for j in range (int(s[0]), int(s[1])):
                temp.append(j)
            possible_values.append(temp)
        return list(itertools.product(*possible_values))

    def find_concrete_qtable_indices (self, concrete_states):
        indices = []
        for s in concrete_states:
            indices.append(self._env.state_to_index(s))
        return indices

    def update_n_abstract_state (self):
        self._n_abstract_states = len(self._tree._leaves)

    def update_abstraction (self, eval_log):
        eval = self.clean_eval(eval_log)
        if len(eval)>0:
            unstable_states = self.find_k_unstable_state(eval)
            #self.generalize_good_splits()
            for s in unstable_states:
                self.update_tree (s)
            self.update_n_abstract_state ()
            if hyper_param.bootstrap == 'from_concrete': self.update_abstract_maze ()

    def clean_eval(self, eval_in):
        eval = deepcopy(eval_in)
        indivisible_states = []
        for state in eval:
            valid = False
            node = self._tree.find_node (state)
            if node._parent is None: 
                split = self._state_ranges
                for i in range (len(state)):
                    lower = split [i][0]
                    upper = split [i][1]
                    if int(upper) - math.ceil(lower) > 1: 
                        valid = True
                        break
                if not valid: 
                    indivisible_states.append(state)
            else: 
                split = node._parent._split
                indices = self.state_to_split_indices (state, split)
                for i in range (len(state)):
                    index = indices[i]
                    lower = split [i][index]
                    upper = split [i][index+1]
                    if int(upper) - math.ceil(lower) > 1: 
                        valid = True
                        break
                if not valid: 
                    indivisible_states.append(state)

        for s in indivisible_states:
            del eval[s]
        return eval

    def find_k_unstable_state (self, eval_log):
        var_dict = {}
        unstable_states = []
        for state in eval_log:
            std_temp = []
            for i in range (self._n_action_size):
                variation = pow(np.std(eval_log[state][i]),2) / abs(np.average(eval_log[state][i]))
                if np.isnan(variation): variation = 0
                std_temp.append(variation)
            max_current = max(std_temp)
            var_dict [max_current] = state
               
        var_dict = dict(sorted(var_dict.items(),reverse=True))
        unstable_selected = list(var_dict.items())
        q = self.get_total_unstable_number(unstable_selected)
        self._unstable_cap.append(q)
        k = min (self._k, q)
        unstable_selected = unstable_selected[0:k]
        for item in unstable_selected:
            unstable_states.append(item[1])
        self._current_counter_examples = unstable_states 
        return unstable_states

    def get_total_unstable_number(self, variation_values):
        if len(variation_values) > 0:
            X = []
            if variation_values[-1][0] < 1: base = 1 
            else: base = variation_values[-1][0]
            for i in range (len(variation_values)):
                temp = []
                item = variation_values[i][0]
                if item < 1: item  = 1
                v = int(item/base)
                temp.append(math.log(v,2))
                X.append(temp)
            X = np.array(X)
            X.reshape(-1, 1)
            kmeans = KMeans(n_clusters=min(len(variation_values),3), n_init="auto").fit(X)
            res = kmeans.predict(X)
            ref = res[0]
            num = 0
            for i in range(len(res)):
                if res[i] == ref: num += 1
            return num
        else: return 0

    def update_tree (self, unstable_state):
        temp = []
        node = self._tree.find_node (unstable_state)
        if node._parent is None: 
            split = self._state_ranges
            for i in range(len(split)): split[i] = list(split[i])
        else: split = node._parent._split
        vector = self.get_to_split_variables(unstable_state)
        
        new_split, new_state_values = self.update_split(unstable_state, split, vector)
        node._split = new_split # the node now has a different split compared to its parent 
        del self._tree._leaves[unstable_state] # the no longer is a leaf node
        new_leaf_nodes = list(itertools.product(*new_state_values))
        for s in new_leaf_nodes:
            self._agent.update_qtable (s)
            #self._agent._qtable [s] = self._agent._qtable [unstable_state]
            new_node = self._tree.add_node (split = [], abs_state = s)
            new_node._parent = node
            node._child.append(new_node)
            self._tree._leaves[s] = new_node
            temp.append(s)
        if unstable_state in self._agent._qtable: del self._agent._qtable[unstable_state] # the no longer is a leaf node
    
    def update_split(self, unstable_state, split_in, to_split_vector):
        split = deepcopy(split_in)
        split_indices = self.state_to_split_indices(unstable_state, split)
        if unstable_state == self._root_name: 
            split_indices = []
            for i in range (self._n_states):
                split_indices.append(0)
        new_state_values = []
        for i in range(len(split_indices)):
            index = split_indices[i]
            # if we need to split the state variable
            if to_split_vector[i] == 1:
                if split[i][index+1] - split[i][index] > 1: # if the sepcific range is dividable
                    new_split_point = (split[i][index+1] - split[i][index])/2 + split[i][index]
                    if not new_split_point - int(new_split_point) == 0: new_split_point = math.ceil(new_split_point)
                    split[i].append(int(new_split_point))
                    split[i].sort()
                    new_state_values.append([str(split[i][index]) + "," + str(int(new_split_point)), str(int(new_split_point)) + "," + str(split[i][index + 2]) ])
                else: new_state_values.append([str(split[i][index]) + "," + str(split[i][index + 1])])
            else:
                new_state_values.append([str(split[i][index]) + "," + str(split[i][index + 1])])
                split[i] = split_in[i]
        return split, new_state_values

          
    def update_abstract_maze (self):
        max_y = self._maze_abstract.shape[0] 
        max_x = self._maze_abstract.shape[1]
        for i in range (max_y):
            for j in range (max_x):
                temp = [i,j]
                for k in range (2, self._n_states):
                    temp.append(self._state_ranges[k][0])
                abs_state = self.state(temp)
                self._maze_abstract[i][j] = abs_state[0] + "_" + abs_state[1]

    def get_all_mazes (self):
        max_y = self._maze_abstract.shape[0] 
        max_x = self._maze_abstract.shape[1]
        mazes = []
        other_values_temp = []
        for k in range (2, self._n_states):
            temp = []
            current = self._state_ranges[k][0]
            high = self._state_ranges[k][1]
            while (current < high):
                temp.append(current)
                current += 1
            other_values_temp.append(temp)
        other_values = list(itertools.product(*other_values_temp))
        for value in other_values:
            if self.value_is_valid(value):
                temp_maze = deepcopy (self._maze_abstract)
                for i in range (max_y):
                    for j in range (max_x):
                        temp = [i,j]
                        for v in value: temp.append(v)
                        abs_state = self.state(temp)
                        self._agent.update_qtable(abs_state)
                        temp_maze[i][j] = str(abs_state)
                mazes.append ([temp_maze, value, self._env._locations])
        return mazes

    def value_is_valid (self, values):
        for i in range (len(values)):
            for j in range (len(values)):
                if values[i] == values[j] and i != j:
                    if values[i] != 0: return False
        return True

    def plot_all_heatmaps(self, directory, abstraction_colors, best_actions):
        # if self._n_states == 2: 
        #     self.plot_heat_map()
        # else:
        all_data = self.get_all_mazes()
        main_maze = self._env._maze
        qtable = self.revise_qtable()
        for data in all_data:
            maze_abs = data[0]
            file_name = directory + str(data[1]) + ".png"
            all_locs = data[2]
            goal = all_locs[data[1][-1]]
            p_locs = []
            for i in range (len(data[1])-1):
                point = all_locs[data[1][i]]
                p_locs.append(point)
            abstraction_colors, best_actions = Results.get_qtable_heatmap(main_maze, maze_abs, 40, qtable, file_name, goal, p_locs, abstraction_colors, best_actions)
        # print (data[2])
        return abstraction_colors, best_actions

    def revise_qtable(self):
        new_table = {}
        for key in self._agent._qtable:
            new_table[str(key)] = self._agent._qtable[key]
        return new_table

    def plot_heat_map(self):
        abstraction_colors, best_actions = Results.get_qtable_heatmap(self._env._maze, self._maze_abstract, 40, self.revise_qtable(), hyper_param.map_name + ".png", [], [], {}, {})
