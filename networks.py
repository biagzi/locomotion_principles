import random
import numpy as np
import networkx as nx
from networkx import DiGraph
from copy import deepcopy
from collections import OrderedDict

from tools.utils import neg_abs, neg_square, sqrt_abs, neg_sqrt_abs, normalize, sigmoid, vox_xyz_from_id, add_patch


class OrderedGraph(DiGraph):
    """Create a graph object that tracks the order nodes and their neighbors are added."""
    node_dict_factory = OrderedDict  #R: OrderedDict is a dictionary that keeps the same order as the order you added the key and value
    adjlist_dict_factory = OrderedDict


class Network(object):
    """Base class for networks."""

    input_node_names = []

    def __init__(self, output_node_names):
        self.output_node_names = output_node_names
        #R: .graph is a DiGraph
        self.graph = OrderedGraph()  # preserving order is necessary for checkpointing
        self.freeze = False
        self.allow_neutral_mutations = False
        self.num_consecutive_mutations = 1
        self.switch = False
        self.direct_encoding = False
        self.direct_global_feature = False

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def set_input_node_states(self, *args, **kwargs):
        raise NotImplementedError

    def mutate(self, *args, **kwargs):
        raise NotImplementedError


class CPPN(Network):
    """A Compositional Pattern Producing Network"""

    #The coordinates of each voxel are specified by x,y,z
    # and radial distances from the center of the lattice workspace d. here, 8-by-8-by-7
    # bias: b term set equal to 1 is taken as input to the network.
    input_node_names = ['x', 'y', 'z', 'd', 'b']

    #functions of CPPN 
    activation_functions = [np.sin, np.abs, neg_abs, np.square, neg_square, sqrt_abs, neg_sqrt_abs]

    def __init__(self, output_node_names):
        Network.__init__(self, output_node_names)
        self.set_minimal_graph()
        self.mutate()

    def set_minimal_graph(self):
        """Create a simple graph with each input attached to each output"""
        for name in self.input_node_names:                  #R: Cria os nodes de input
            self.graph.add_node(name, type="input", function=None)

        for name in self.output_node_names:                 #R: Cria os nodes de output
            self.graph.add_node(name, type="output", function=sigmoid)

        for input_node in nx.nodes(self.graph):             #R: Para cada node de input, conecta com um node de output com peso 0
            if self.graph.node[input_node]["type"] == "input":
                for output_node in nx.nodes(self.graph):
                    if self.graph.node[output_node]["type"] == "output":
                        self.graph.add_edge(input_node, output_node, weight=0.0)

    def set_input_node_states(self, orig_size_xyz,bias=1): #R: Creates a input vector to each initial node (x,y,z,d,b)
        input_x = np.zeros(orig_size_xyz)
        input_y = np.zeros(orig_size_xyz)
        input_z = np.zeros(orig_size_xyz)
        for x in range(orig_size_xyz[0]):
            for y in range(orig_size_xyz[1]):
                for z in range(orig_size_xyz[2]): 
                    input_x[x, y, z] = x
                    input_y[x, y, z] = y
                    input_z[x, y, z] = z

        input_x = normalize(input_x)
        input_y = normalize(input_y)
        input_z = normalize(input_z)
        input_d = normalize(np.power(np.power(input_x, 2) + np.power(input_y, 2) + np.power(input_z, 2), 0.5))
        if bias == 1:
            input_b = np.ones(orig_size_xyz)
        else:
            input_b = np.ones(orig_size_xyz)*bias

        for name in self.graph.nodes():  #R: nodes in Di.Graph.nodes is like a dict [node,nodedata]
            if name == "x":
                self.graph.node[name]["state"] = input_x
                self.graph.node[name]["evaluated"] = True
            if name == "y":
                self.graph.node[name]["state"] = input_y
                self.graph.node[name]["evaluated"] = True
            if name == "z":
                self.graph.node[name]["state"] = input_z
                self.graph.node[name]["evaluated"] = True
            if name == "d":
                self.graph.node[name]["state"] = input_d
                self.graph.node[name]["evaluated"] = True
            if name == "b":
                self.graph.node[name]["state"] = input_b
                self.graph.node[name]["evaluated"] = True

    def mutate(self, num_random_node_adds=10, num_random_node_removals=0, num_random_link_adds=10,
               num_random_link_removals=5, num_random_activation_functions=100, num_random_weight_changes=100):

        variation_degree = None
        variation_type = None

        for _ in range(num_random_node_adds):
            variation_degree = self.add_node()
            variation_type = "add_node"

        for _ in range(num_random_node_removals):
            variation_degree = self.remove_node()
            variation_type = "remove_node"

        for _ in range(num_random_link_adds):
            variation_degree = self.add_link()
            variation_type = "add_link"

        for _ in range(num_random_link_removals):
            variation_degree = self.remove_link()
            variation_type = "remove_link"

        for _ in range(num_random_activation_functions):
            variation_degree = self.mutate_function()
            variation_type = "mutate_function"

        for _ in range(num_random_weight_changes):
            variation_degree = self.mutate_weight()
            variation_type = "mutate_weight"

        self.prune_network()
        return variation_type, variation_degree

    ###############################################
    #   Mutation functions
    ###############################################

    def add_node(self):
        #choose two random nodes (between which a link could exist)
        if len(self.graph.edges()) == 0:
            return "NoEdges"
        this_edge = random.choice(self.graph.edges())
        node1 = this_edge[0]
        node2 = this_edge[1]

        # create a new node hanging from the previous output node
        new_node_index = self.get_max_hidden_node_index()
        self.graph.add_node(new_node_index, type="hidden", function=random.choice(self.activation_functions))
        # random activation function here to solve the problem with admissible mutations in the first generations
        self.graph.add_edge(new_node_index, node2, weight=1.0)

        # if this edge already existed here, remove it
        # but use it's weight to minimize disruption when connecting to the previous input node
        if (node1, node2) in nx.edges(self.graph):
            weight = self.graph.edge[node1][node2]["weight"]
            self.graph.remove_edge(node1, node2)
            self. graph.add_edge(node1, new_node_index, weight=weight)
        else:
            self.graph.add_edge(node1, new_node_index, weight=1.0)
            # weight 0.0 would minimize disruption of new edge
            # but weight 1.0 should help in finding admissible mutations in the first generations
        return ""

    def remove_node(self):
        hidden_nodes = list(set(self.graph.nodes()) - set(self.input_node_names) - set(self.output_node_names))
        if len(hidden_nodes) == 0:
            return "NoHiddenNodes"
        this_node = random.choice(hidden_nodes)

        # if there are edge paths going through this node, keep them connected to minimize disruption
        incoming_edges = self.graph.in_edges(nbunch=[this_node])
        outgoing_edges = self.graph.out_edges(nbunch=[this_node])

        for incoming_edge in incoming_edges:
            for outgoing_edge in outgoing_edges:
                w = self.graph.edge[incoming_edge[0]][this_node]["weight"] * \
                    self.graph.edge[this_node][outgoing_edge[1]]["weight"]
                self.graph.add_edge(incoming_edge[0], outgoing_edge[1], weight=w)

        self.graph.remove_node(this_node)
        return ""

    def add_link(self):
        done = False
        attempt = 0
        while not done:
            done = True

            # choose two random nodes (between which a link could exist, *but doesn't*)
            node1 = random.choice(self.graph.nodes())
            node2 = random.choice(self.graph.nodes())
            while (not self.new_edge_is_valid(node1, node2)) and attempt < 999:
                node1 = random.choice(self.graph.nodes())
                node2 = random.choice(self.graph.nodes())
                attempt += 1
            if attempt > 999:  # no valid edges to add found in 1000 attempts
                done = True

            # create a link between them
            if random.random() > 0.5:
                self.graph.add_edge(node1, node2, weight=0.1)
            else:
                self.graph.add_edge(node1, node2, weight=-0.1)

            # If the link creates a cyclic graph, erase it and try again
            if self.has_cycles():
                self.graph.remove_edge(node1, node2)
                done = False
                attempt += 1
            if attempt > 999:
                done = True
        return ""

    def remove_link(self):
        if len(self.graph.edges()) == 0:
            return "NoEdges"
        this_link = random.choice(self.graph.edges())
        self.graph.remove_edge(this_link[0], this_link[1])
        return ""

    def mutate_function(self):
        this_node = random.choice(self.graph.nodes())
        while this_node in self.input_node_names:
            this_node = random.choice(self.graph.nodes())
        old_function = self.graph.node[this_node]["function"]
        while self.graph.node[this_node]["function"] == old_function:
            self.graph.node[this_node]["function"] = random.choice(self.activation_functions)
        return old_function.__name__ + "-to-" + self.graph.node[this_node]["function"].__name__

    def mutate_weight(self, mutation_std=0.5):
        if len(self.graph.edges()) == 0:
            return "NoEdges"
        this_edge = random.choice(self.graph.edges())
        node1 = this_edge[0]
        node2 = this_edge[1]
        old_weight = self.graph[node1][node2]["weight"]
        new_weight = old_weight
        while old_weight == new_weight:
            new_weight = random.gauss(old_weight, mutation_std)
            new_weight = max(-1.0, min(new_weight, 1.0))
        self.graph[node1][node2]["weight"] = new_weight
        return float(new_weight - old_weight)

    ###############################################
    #   Helper functions for mutation
    ###############################################

    def prune_network(self):
        """Remove erroneous nodes and edges post mutation."""
        done = False
        while not done:
            done = True
            for node in self.graph.nodes():
                if len(self.graph.in_edges(nbunch=[node])) == 0 and \
                                node not in self.input_node_names and \
                                node not in self.output_node_names:
                    self.graph.remove_node(node)
                    done = False

            for node in self.graph.nodes():
                if len(self.graph.out_edges(nbunch=[node])) == 0 and \
                                node not in self.input_node_names and \
                                node not in self.output_node_names:
                    self.graph.remove_node(node)
                    done = False

    def has_cycles(self):
        """Return True if the graph contains simple cycles (elementary circuits).

        A simple cycle is a closed path where no node appears twice, except that the first and last node are the same.

        """
        return sum(1 for _ in nx.simple_cycles(self.graph)) != 0

    def get_max_hidden_node_index(self):
        max_index = 0
        for input_node in nx.nodes(self.graph):
            if self.graph.node[input_node]["type"] == "hidden" and int(input_node) >= max_index:
                max_index = input_node + 1
        return max_index

    def new_edge_is_valid(self, node1, node2):
        if node1 == node2:
            return False
        if self.graph.node[node1]['type'] == "output":
            return False
        if self.graph.node[node2]['type'] == "input":
            return False
        if (node2, node1) in nx.edges(self.graph):
            return False
        if (node1, node2) in nx.edges(self.graph):
            return False
        return True


class DirectEncoding(Network):
    def __init__(self, output_node_name, orig_size_xyz, lower_bound=-1, upper_bound=1, func=None, symmetric=True,
                 p=None, scale=None, start_val=None, mutate_start_val=False, allow_neutral_mutations=False,
                 sub_vox_dict=None, frozen_vox=None, patch_mode=False, vox_options=None):

        Network.__init__(self, [output_node_name])

        self.vox_options = vox_options

        self.direct_encoding = True
        self.allow_neutral_mutations = allow_neutral_mutations
        self.size = orig_size_xyz
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if p is None:
            p = 1/np.product(self.size, dtype='f')
        self.p = p
        self.scale = scale
        self.func = func
        self.symmetric = symmetric
        self.start_value = start_val

        self.patch_mode = patch_mode

        if sub_vox_dict is None:
            self.sub_vox_dict = dict()
        else:
            self.sub_vox_dict = sub_vox_dict

        self.mutable_vox = np.ones(shape=orig_size_xyz, dtype=bool)

        if frozen_vox is not None:
            for idx in frozen_vox:
                x, y, z = vox_xyz_from_id(idx, self.size)
                self.mutable_vox[x, y, z] = False

        #R: The values are random valuem from -1 to 1 in the size of the lattice: (x,y,z)
        if start_val is None:
            self.values = np.random.uniform(lower_bound, upper_bound, size=orig_size_xyz)
        else:
            self.values = np.ones(shape=orig_size_xyz) * start_val
            if mutate_start_val:
                self.mutate()

        if vox_options is not None:
            self.values = np.random.choice(vox_options, orig_size_xyz)

        self.enforce_symmetry()

        self.regulate_sub_voxels()

        if self.func is not None:
            self.values = self.func(self.values)

        self.values = np.clip(self.values, self.lower_bound, self.upper_bound)

    def set_input_node_states(self, *args, **kwargs):
        pass

    def mutate(self, rate=None):

        if self.patch_mode:
            self.values, sub_vox_dict = add_patch(self.values)

            for parent, child in sub_vox_dict.items():
                self.sub_vox_dict[parent] = [child]

            return "patched", 1

        else:
            if rate is None:
                rate = self.p #p = 1/np.product(self.size, dtype='f')

            scale = self.scale
            if self.scale is None:
                scale = np.abs(1/self.values)
                # scale = np.clip(self.values**0.5, self.start_value**0.5, self.upper_bound)
                # this was for meta mutations

            #Selects the places where the mutation occurs
            selection = np.random.random(self.size) < rate
            selection = np.logical_and(selection, self.mutable_vox)

            if self.vox_options is not None:
                change = np.random.choice(self.vox_options, self.size)
                self.values[selection] = change[selection]
            else:
                change = np.random.normal(scale=scale, size=self.size) #R: if the "standart" case, 
                self.values[selection] += change[selection]

            self.values = np.clip(self.values, self.lower_bound, self.upper_bound)

            self.enforce_symmetry()

            self.regulate_sub_voxels()

            if self.func is not None:
                self.values = self.func(self.values)

            return "gaussian", self.scale

    #R: enforce_symmetry in x axis (x matrices are all equal)
    def enforce_symmetry(self):
        if self.symmetric: #R: Just if the parammeter symmetric is True
            reversed_array = self.values[::-1, :, :] #[::-1] means 'nothing for the first argument, nothing for the second, and jump by -1'
            self.values[:int(self.size[0]/2.0), :, :] = reversed_array[:int(self.size[0]/2.0), :, :]

    def regulate_sub_voxels(self):
        if len(self.sub_vox_dict) == 0:
            return

        self.mutable_vox = np.zeros(self.size, dtype=bool)

        for parent, children in self.sub_vox_dict.items():
            px, py, pz = vox_xyz_from_id(parent, self.size)
            self.mutable_vox[px, py, pz] = True
            group_val = self.values[px, py, pz] / float(len(children))
            self.values[px, py, pz] = group_val
            for child in children:
                cx, cy, cz = vox_xyz_from_id(child, self.size)
                self.values[cx, cy, cz] = group_val

class DirectGlobalFeature(Network):
    def __init__(self, output_node_names, feature_possibilities = [5e5,5e6,5e7]):

        Network.__init__(self, [output_node_names])

        self.direct_global_feature = True
        self.feature_possibilities = feature_possibilities
        self.feature = np.random.choice(self.feature_possibilities)


    def set_input_node_states(self, *args, **kwargs):
        pass

    def mutate(self):
        new_feature = self.feature
        while new_feature == self.feature:
            new_feature = np.random.choice(self.feature_possibilities)

        self.feature = new_feature

        return "Different_global_stiffnes",self.feature


        
