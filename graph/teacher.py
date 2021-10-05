from collections import defaultdict
import random
import numpy as np
import pickle as pkl

from graph.semantic_graph import SemanticGraph
import networkit as nk

class Teacher():
    def __init__(self,args):
        self.oracle_graph = SemanticGraph.load_oracle(args.n_blocks)
        self.args = args
        self.agent_frontier = {} # store configuration through networkit node_id from agent_graph
        self.agent_stepping_stones = {}
        self.agent_frontier_not_stepping_stones = {}
        with open('data/classes_and_configs.pkl', 'rb') as f:
            self.config_to_class, _ = pkl.load(f)

        self.init_stats()

    def is_in_frontier(self,config,agent_graph:SemanticGraph):
        '''
        Compute the ensemble of nodes wich a part of the frontier : 
            --> nodes that exist in the oracle graph
            --> nodes that are not intermediate node in the path [coplanar -> any node] of the agent graph
            --> nodes that have unknown explorable childs 
        '''
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
        '''
        Compute the set of nodes that are stepping stones to the unknown
        '''
        if not self.oracle_graph.hasNode(config):
            return False

        neighbours = self.oracle_graph.iterNeighbors(config)
        for neighbour in neighbours:
            # if not agent_graph.hasNode(neighbour):
            if not agent_graph.hasEdge(config, neighbour):
                return True
        return False

    def is_frontier_not_stepping_stone(self, config, agent_graph: SemanticGraph):
        '''
        Compute the set of nodes that are not stepping stones but are at the frontier
        '''
        if not self.oracle_graph.hasNode(config):
            return False

        if agent_graph.getNodeId(config) in agent_graph.frontier:
            return True
        return False


    def computeFrontier(self,agent_graph:SemanticGraph):
        self.agent_frontier = set()
        self.agent_stepping_stones = set()
        self.agent_frontier_not_stepping_stones = set()
        for node in agent_graph.configs:
            # if self.is_in_frontier(node,agent_graph):
            #     self.agent_frontier.add( agent_graph.getNodeId(node))
            if self.is_stepping_stone(node,agent_graph):
                self.agent_frontier.add(agent_graph.getNodeId(node))
                self.agent_stepping_stones.add(agent_graph.getNodeId(node))
                continue
            elif self.is_frontier_not_stepping_stone(node, agent_graph):
                self.agent_frontier.add(agent_graph.getNodeId(node))
                self.agent_frontier_not_stepping_stones.add(agent_graph.getNodeId(node))

    def sample_in_frontier(self,current_node,agent_graph,k, ablation=5):
        reachables = agent_graph.get_reachables_node_ids(current_node)
        reachable_frontier = [agent_graph.getConfig(node_id) 
                              for node_id in reachables 
                              if node_id in self.agent_frontier]
        reachable_stepping_stones = [agent_graph.getConfig(node_id)
                              for node_id in reachables
                              if node_id in self.agent_stepping_stones]
        reachable_frontier_not_stepping_stones = [agent_graph.getConfig(node_id)
                                                  for node_id in reachables
                                                  if node_id in self.agent_frontier_not_stepping_stones]
        if ablation != 2:
            if reachable_frontier:
                goals = random.choices(reachable_frontier, k=k)  # sample with replacement
                for g in goals:
                    try:
                        c = self.config_to_class[str(np.array(g).reshape(1, -1))]
                        self.stats[c+1] += 1
                    except KeyError:
                        pass
                return goals
            else:
                return []
        else:
            if reachable_stepping_stones:
                goals = random.choices(reachable_stepping_stones, k=k)  # sample with replacement
                for g in goals:
                    try:
                        c = self.config_to_class[str(np.array(g).reshape(1, -1))]
                        self.stats[c + 1] += 1
                    except KeyError:
                        pass
                return goals
            else:
                return []

    def sample_from_frontier(self,node,agent_graph,k, ablation=5):
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
