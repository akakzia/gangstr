from collections import defaultdict
import math
import numpy as np
from bidict import bidict
import networkit as nk
from graph.SemanticOperation import config_permutations
from graph.semantic_graph import SemanticGraph

class UnorderedSemanticGraph(SemanticGraph):
    '''
    all edges identical to one permutation share the same success rate
    '''
    def __init__(self, configs: bidict, graph: nk.graph, nb_blocks, GANGSTR=True,edges_infos=None,args=None):
        super().__init__(configs, graph, nb_blocks, GANGSTR=GANGSTR, edges_infos=edges_infos, args=args)
        self.ordered_edge_to_unordered_edge = dict()
        self.unordered_edge_to_ordered_edge = defaultdict(set)

    def create_node(self,config):
        if config not in self.configs:
            for c in set(config_permutations(config,self.semantic_operation)):
                super().create_node(c)
    
    def create_edge_stats(self,edge,start_sr):
        c1,c2 = edge

        if self.args.one_object_edge and not self.semantic_operation.one_object_edge(edge):
            return

        if not self.hasEdge(c1,c2):
            unordered_id = len(self.unordered_edge_to_ordered_edge)
            self.edges_infos[unordered_id] = {'SR':start_sr,'Count':1}

            clamped_sr = max(np.finfo(float).eps, min(start_sr, 1-np.finfo(float).eps))
            additive_sr = -math.log(clamped_sr)
            
            for c_perm_1,c_perm_2 in zip(config_permutations(c1,self.semantic_operation),
                                        config_permutations(c2,self.semantic_operation)):
                n1,n2 = self.configs[c_perm_1],self.configs[c_perm_2]
                if not self.nk_graph.hasEdge(n1,n2):
                    self.nk_graph.addEdge(n1,n2)
                    self.nk_graph.setWeight(n1,n2,additive_sr)

                    self.ordered_edge_to_unordered_edge[(n1,n2)] = unordered_id
                    self.unordered_edge_to_ordered_edge[unordered_id].add((n1,n2))
        else : 
            raise Exception(f'Already existing edge {c1}->{c2}')

    def edge_config_to_edge_id(self,edge_config):
        c1,c2 = edge_config
        n1,n2 = (self.configs[c1],self.configs[c2])
        unordered_edge_id = self.ordered_edge_to_unordered_edge[(n1,n2)]
        return unordered_edge_id

    def update_graph_edge_weight(self,edge):

        new_mean_sr = self.edges_infos[edge]['SR']
        clamped_sr = max(np.finfo(float).eps, min(new_mean_sr, 1-np.finfo(float).eps))

        for n1,n2 in self.unordered_edge_to_ordered_edge[edge]:
            self.nk_graph.setWeight(n1,n2,-math.log(clamped_sr))

    def log(self,logger):
        logger.record_tabular('agent_nodes_ordered',self.nk_graph.numberOfNodes())
        logger.record_tabular('agent_edges_ordered',self.nk_graph.numberOfEdges())
        logger.record_tabular('agent_edges_unordered',len(self.edges_infos))

    def get_unique_unordered_paths(self,paths,scores):
        '''
            Receives an array of paths, paths are list of nodes. 
            Convert each path into a list of unordered edges, 
            If multiple paths are identical unordered-edes-wise, only return the id for one of them (at random among best scores).
        '''
        # create paths of unordered edges from paths of nodes
        unordered_edge_paths = np.ones((len(paths),max(map(len,paths))-1))*-1 # init with unused weight id
        for i,path in enumerate(paths):
            for j in range(0,len(path)-1):
                e = (path[j], path[j+1])
                unordered_edge_paths[i,j] = self.ordered_edge_to_unordered_edge[e]
        unique_unordered_paths,inverse_inds = np.unique(unordered_edge_paths,axis=0,return_inverse = True) 

        # sample among unordered edges paths with highest scores
        unique_ordered_paths_ids = np.zeros(unique_unordered_paths.shape[0],dtype=int)
        for i,e in enumerate(unique_unordered_paths):
            e_neigh_ids = np.where(inverse_inds == i)[0]
            e_scores = scores[e_neigh_ids]
            highest_scores = e_neigh_ids[np.argwhere(e_scores == np.amax(e_scores)).flatten()]
            choosen_id = np.random.choice(highest_scores)
            unique_ordered_paths_ids[i] = choosen_id

        return unique_ordered_paths_ids
