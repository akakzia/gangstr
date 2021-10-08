from collections import defaultdict
import random
import numpy as np
import pickle as pkl

from graph.semantic_graph import SemanticGraph
import networkit as nk

class Teacher():
    def __init__(self,args):
        self.oracle_graph = SemanticGraph.load_oracle(args.n_blocks)
        self.strategy = args.strategy
        self.args = args
        self.agent_frontier = {} # store configuration through networkit node_id from agent_graph
        self.agent_stepping_stones = {}
        self.agent_terminal = {}
        with open('data/classes_and_configs.pkl', 'rb') as f:
            self.config_to_class, _ = pkl.load(f)

        self.init_stats()

    def is_in_frontier(self,config,agent_graph:SemanticGraph):
        """
        Compute the set of nodes which belong to the frontier :
            --> nodes that exist in the oracle graph
            --> nodes that are terminal
            --> nodes that are stepping stones
        """
        if not self.oracle_graph.hasNode(config):
            return False

        if agent_graph.getNodeId(config) in agent_graph.frontier:
            return True
    
        neighbours = self.oracle_graph.iterNeighbors(config)
        for neighbour in neighbours:
            # if not agent_graph.hasNode(neighbour):
            if not agent_graph.hasEdge(config, neighbour):
                return True
        return False

    def is_stepping_stone(self, config, agent_graph: SemanticGraph):
        """
        Compute the set of nodes that are stepping stones to the unknown
        """
        if not self.oracle_graph.hasNode(config):
            return False

        neighbours = self.oracle_graph.iterNeighbors(config)
        for neighbour in neighbours:
            # if not agent_graph.hasNode(neighbour):
            if not agent_graph.hasEdge(config, neighbour):
                return True
        return False

    def is_terminal(self, config, agent_graph: SemanticGraph):
        """
        Compute the set of nodes that are not stepping stones but are at the frontier
        """
        if not self.oracle_graph.hasNode(config):
            return False

        if agent_graph.getNodeId(config) in agent_graph.frontier:
            return True
        return False

    def compute_frontier(self, agent_graph: SemanticGraph):
        self.agent_frontier = set()
        self.agent_stepping_stones = set()
        self.agent_terminal = set()
        for node in agent_graph.configs:
            if self.is_stepping_stone(node, agent_graph):
                self.agent_frontier.add(agent_graph.getNodeId(node))
                self.agent_stepping_stones.add(agent_graph.getNodeId(node))
                continue
            elif self.is_terminal(node, agent_graph):
                self.agent_frontier.add(agent_graph.getNodeId(node))
                self.agent_terminal.add(agent_graph.getNodeId(node))
        
    def sample_in_frontier(self,current_node,agent_graph,k):
        reachable = agent_graph.get_reachables_node_ids(current_node)
        # reachable_frontier = [agent_graph.getConfig(node_id)
        #                       for node_id in reachable
        #                       if node_id in self.agent_frontier]
        reachable_stepping_stones = [agent_graph.getConfig(node_id)
                                     for node_id in reachable
                                     if node_id in self.agent_stepping_stones]
        reachable_terminal = [agent_graph.getConfig(node_id)
                                                  for node_id in reachable
                                                  if node_id in self.agent_terminal]

        reachable_frontier = reachable_terminal + reachable_stepping_stones
        try:
            if self.strategy in [0, 3]:
                # If strategy is Frontier
                # If strategy is Beyond (first sample in frontier without giving it to the agent)
                goals = random.choices(reachable_frontier, k=k)
            elif self.strategy == 1:
                # If strategy is Frontier and Stop
                goals = random.choices(reachable_terminal, k=k)
            elif self.strategy == 2:
                # If strategy is Frontier and Beyond
                goals = random.choices(reachable_stepping_stones, k=k)
            else:
                raise NotImplementedError
            for g in goals:
                try:
                    c = self.config_to_class[str(np.array(g).reshape(1, -1))]
                    self.stats[c+1] += 1
                except KeyError:
                    pass
            return goals
        except IndexError:
            return []

    def sample_from_frontier(self,node,agent_graph,k):
        to_explore = []
        to_exploit = []
        for neighbour in self.oracle_graph.iterNeighbors(node):
            if not agent_graph.hasEdge(node,neighbour):
                to_explore.append(neighbour)
            else:
                to_exploit.append(neighbour)
        # If there are goals outside to explore
        if to_explore:
            goals = random.choices(to_explore,k=k) # sample with replacement
            for g in goals:
                try:
                    c = self.config_to_class[str(np.array(g).reshape(1, -1))]
                    self.stats[c+1] += 1
                except KeyError:
                    pass
            return goals
        # If there are goals inside to consolidate and the probability of exploring inside is not exclusive
        elif to_exploit:
            return random.choices(to_exploit,k=k)
        else : 
            return []

    def sample_goal_uniform(self,nb_goal):
        return random.choices(self.oracle_graph.configs.inverse,k=nb_goal) # sample with replacement

    def init_stats(self):
        self.stats = dict()
        for i in range(11):
            self.stats[i+1] = 0
