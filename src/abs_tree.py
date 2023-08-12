import os
import networkx as nx
import ast

class Abstraction_Tree():

    class Abs_node():
        def __init__(self, split, abs_state):
            self._split = split
            self._state = abs_state
            self._parent = None
            self._child = []
           
    def find_node (self, abs_state):
        found = None
        for leaf in self._leaves:
            if leaf == abs_state: 
                found = self._leaves[leaf]
                break
        return found

    def __init__(self, root_split, root_abs_state):
        self._leaves = {}
        self._root = self.add_node(root_split, root_abs_state)
        
    def add_node (self, split, abs_state):
        node  = self.Abs_node(split, abs_state)
        self._leaves[abs_state] = node
        return node
    
    def get_networkx_cat(self):
        graph = nx.DiGraph()
        stack = [self._root]
        while stack:
            temp = stack.pop()
            for child in temp._child:
                stack.append(child)
                node1 = temp._state
                node2 = child._state 
                graph.add_edge(node1, node2)
        return graph
    
    def relabel(self,old_node,gran): 
        new_label = []
        for s in old_node: 
            t = list(ast.literal_eval(s))
            for i in range(len(t)):
                t[i] = round(t[i] * gran,2)
            new_label.append(str(tuple(t)))
        return tuple(new_label)
    
    def rescale_and_augment_best_actions(self, graph, gran, best_actions):
        mapping = dict()
        for node in graph.nodes:
            if node=="root": 
                mapping[node] = node
            elif node in best_actions:
                mapping[node] = str(self.relabel(node,0.001)) + " -> "+ str(best_actions[node])
            else:
                mapping[node] = str(self.relabel(node,0.001))
        return mapping
        
    def plot_cat(self, directory, index, best_actions):
        graph = self.get_networkx_cat()

        for node in graph.nodes:
            if node=="root": 
                graph.add_node(node)
            if self.find_node(node):
                graph.add_node(node, shape="box")

        mapping = self.rescale_and_augment_best_actions(graph, 0.001, best_actions)
        graph = nx.relabel_nodes(graph,mapping)

        basepath = os.getcwd()
        if not os.path.exists(basepath+"/plots/"+directory+"/"):
            os.makedirs(basepath+"/plots/"+directory+"/")
        nx.nx_pydot.write_dot(graph, basepath+"/plots/"+directory+"/cat_"+str(index)+".dot")

    

            








