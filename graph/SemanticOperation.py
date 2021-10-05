
from collections import defaultdict
from functools import reduce
from bidict import bidict
import networkx as nx
from itertools import combinations,permutations

import numpy as np
from utils import get_graph_structure

class SemanticOperation():
    '''
    Class used to transform configuration with above and close operations. 
    '''
    def __init__(self,nb_blocks,GANGSTR):
        self.nb_blocks = nb_blocks
        edges, incoming_edges, predicate_ids = get_graph_structure(nb_blocks)
        self.edge_to_close_ids = { edge:predicate_ids[edge_id][0] for edge_id,edge in enumerate(edges)  }
        self.edge_to_above_ids = { edge:predicate_ids[edge_id][1] for edge_id,edge in enumerate(edges)  }

        self.pred_id_to_edge = {pred_id:edge for edge,pred_id in self.edge_to_close_ids.items()}
        self.pred_id_to_edge.update({pred_id:edge for edge,pred_id in self.edge_to_above_ids.items()})
        self.pred_id_to_pred_name = {pred_id:'close' for pred_id in self.edge_to_close_ids.values()}
        self.pred_id_to_pred_name.update({pred_id:'above' for pred_id in self.edge_to_above_ids.values()})

        self.predicates= {'close':self.close,'above': self.above}

        # define how truth values are replaced in semantic configurations : 
        if GANGSTR:
            self.semantic = bidict({True: 1., False:-1.})
        else : 
            self.semantic = bidict({True: 1, False:0})

    def is_close(self,config,a,b):
        close_id = self.edge_to_close_ids[(a,b)]
        return config[close_id]

    def is_above(self,config,a,b):
        above_id = self.edge_to_close_ids[(a,b)]
        return config[above_id]

    def close(self,config,a,b,pred_val):
        '''Return a copy of config with a close from b with value True/False'''
        close_id = self.edge_to_close_ids[(a,b)]
        new_config = config[:close_id] + (self.semantic[pred_val],) + config[close_id+1:]
        return new_config

    def above(self,config,a,b,pred_val):
        '''Return a copy of config with a above b with value True/False'''
        above_id = self.edge_to_above_ids[(a,b)]
        new_config = config[:above_id] + (self.semantic[pred_val],) + config[above_id+1:]
        return new_config

    def close_and_above(self,config,a,b,pred_val):
        '''Return a copy of config with a above b and a close from b with value True/False'''
        config = self.close(config,a,b,pred_val)
        return self.above(config,a,b,pred_val)
    
    def empty(self):
        ''' Return the empty configuration where everything is far appart'''
        return (self.semantic[False],) * ((3*self.nb_blocks * (self.nb_blocks-1))//2)
    
    def to_GANGSTR(config):
        return tuple(1 if c > 0 else -1 
                        for c in config)
    def to_BOOLEAN(config):
        return tuple(1 if c > 0 else 0 
                        for c in config)

    def to_nx_graph(self,config):
        ''' Convert a configuration in an unordered semantic multi-graph'''
        multi_di_graph = nx.MultiDiGraph()

        for pred_id,pred_val in enumerate(config):
            pred_name = self.pred_id_to_pred_name[pred_id]
            obj_a,obj_b = self.pred_id_to_edge[pred_id]
            if self.semantic.inverse[pred_val]:
                multi_di_graph.add_edge(obj_a,obj_b,label=pred_name)
                if pred_name == 'close':
                    multi_di_graph.add_edge(obj_b,obj_a,label=pred_name)
        return multi_di_graph

    def to_nx_graph_hash(self,config):
        nx_graph = self.to_nx_graph(config)
        return nx.weisfeiler_lehman_graph_hash(nx_graph)

    def one_object_edge(self,edge):
        c1,c2 = edge
        c1,c2 = np.array(c1),np.array(c2)
        config_delta_ids = np.arange(len(c1))[c1!=c2] 
        if len(config_delta_ids)>0:
            config_delta_objects = map(lambda id : set(self.pred_id_to_edge[id]),config_delta_ids)
            unique_objects = reduce(set.intersection,config_delta_objects)
            return len(unique_objects)>0
        return False


def all_stack_trajectories(stack_size,GANGSTR= True):
    '''Return a dictionnary of cube-stack associated with semantic-trajectories :
    Keys are all possible stack permutation. (ex : (0,1,2) , (1,2,0) ... )
    Values contains all intermediate configurations from [0,0..] to the config describing the key.'''
    # Generate all possible stack list
    all_cubes = range(stack_size)
    all_stack = list(permutations(all_cubes))

    sem_op = SemanticOperation(stack_size,GANGSTR)
    config_to_path = {}
    for stack in all_stack:
        cur_config = sem_op.empty() # start with the empy config [0,0, ... ]
        config_path = [] 
        # construct intermediate stack config by adding blocks one by one : 
        for top,bottom in zip(stack,stack[1:]) : 
            cur_config = sem_op.close_and_above(cur_config,bottom,top,1)
            config_path.append(cur_config)
        config_to_path[stack] = config_path
    return config_to_path

def config_permutations(config,semantic_operator):
    perms = permutations(range(semantic_operator.nb_blocks))
    all_config_perms = []
    for perm in perms : 
        new_config = semantic_operator.empty()
        for pred_id,pred_val in enumerate(config):
            if pred_val == semantic_operator.semantic[True]:
                pred_name = semantic_operator.pred_id_to_pred_name[pred_id]
                obj_a,obj_b = semantic_operator.pred_id_to_edge[pred_id]
                obj_a,obj_b = perm[obj_a],perm[obj_b]
                new_config = semantic_operator.predicates[pred_name](new_config,obj_a,obj_b,True)
        all_config_perms.append(new_config)
    return all_config_perms

def config_to_unique_str(config,semantic_operator):
    preds = []
    for pred_id in range(len(config)):
        if config[pred_id] == semantic_operator.semantic[True]:
            pred_name = semantic_operator.pred_id_to_pred_name[pred_id]
            obj_a,obj_b = semantic_operator.pred_id_to_edge[pred_id]
            preds+=[f'{pred_name}({obj_a},{obj_b})']
    if len(preds)==0 : 
        return "coplanar"
    else :
        return ' and '.join(preds)

def config_to_name(config,semantic_operator):
    counts = defaultdict(int)
    for pred_id in range(len(config)):
        if config[pred_id] == semantic_operator.semantic[True]:
            pred_name = semantic_operator.pred_id_to_pred_name[pred_id]
            counts[pred_name] +=1
    if len(counts) == 0:
        return 'coplanar'
    else :  
        return '_'.join([f'{pred_name}{pred_nb}' for pred_name,pred_nb in counts.items() if pred_nb>0])

            



