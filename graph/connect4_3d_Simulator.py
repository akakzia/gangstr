
from collections import defaultdict
import numpy as np
from itertools import product
import networkit as nk
import networkx as nx
import copy
import queue
from bidict import bidict
from graph.SemanticOperation import SemanticOperation

'''
Simplified environment used to construct best path semantic graph.
'''
class UnstableException(Exception):
    pass

class Block:
    CLOSE_DIST = np.sqrt(1+1.4**2)
    ABOVE_DIST = np.sqrt(1.9)
    @property
    def h(self):
        return self.pos[0]
    @h.setter
    def h(self, new_h):
        self.pos = (new_h,self.pos[1],self.pos[2])

    def __init__(self,pos,block_id,size=(1,0.9,0.9)):
        self.pos = pos
        self.id = block_id
        self.size=size
    
    def overlap_area(a:'Block',b:'Block'):
        a_h,a_x,a_y = a.pos
        b_h,b_x,b_y= b.pos 
        if a_h == b_h:
            left = max(a_x, b_x)
            right = min(a_x + a.size[1], b_x + b.size[1])
            bottom = max(a_y, b_y)
            top = min(a_y + a.size[2], b_y + b.size[2])
            height = top - bottom 
            width = right - left
            if width>0 and height>0:
                return height*width
        return 0

    def overlap(a:'Block',b:'Block'):
        a_h,a_x,a_y = a.pos
        b_h,b_x,b_y= b.pos 
        if a_h == b_h :
            return (a.pos[1] <= (b.pos[1] + b.size[1]) and 
                    (a.pos[1] + a.size[1]) >= b.pos[1] and 
                    a.pos[2] <= (b.pos[2] + b.size[2]) and 
                    (a.pos[2] + a.size[2]) >= b.pos[2])
        return False

    def is_above(a:'Block',b:'Block'):
        a_h,a_x,a_z = a.pos
        b_h,b_x,b_z= b.pos
        distance = (a_h-b_h)**2 + (a_x-b_x)**2 + (a_z-b_z)**2
        return ((a_h-1== b_h) and distance  < Block.ABOVE_DIST**2)

    def is_close(a:'Block',b:'Block'):
        a_h,a_x,a_z = a.pos
        b_h,b_x,b_z= b.pos
        distance = (a_h-b_h)**2 + (a_x-b_x)**2 + (a_z-b_z)**2
        return distance  < Block.CLOSE_DIST**2

    def __str__(self):
        return f'[id:{self.id},pos:({self.pos})]'

class EnvSimulatorSimplified(): 
    EMPTY =-1

    def __init__(self,size):
        '''
            size : height *n*n
        '''
        self.blocks_by_level = defaultdict(list)
        self.blocks_by_id = dict()
        self.predicates= {'close':Block.is_close,'above': Block.is_above}
        self.size = size

    def remove(self,block_id):
        block = self.blocks_by_id[block_id]
        del self.blocks_by_id[block.id]
        self.blocks_by_level[block.h].remove(block)
    
    def overlap_any(self,block:Block):
        for other_b in self.blocks_by_level[block.h]:
            if other_b.id != block.id and Block.overlap(block,other_b):
                return True
        return False
        
    def total_overlap(self,block:Block):
        ''' Return total horizontal surface of overlapping blocks'''
        total = 0
        for other_b in self.blocks_by_level[block.h]:
            if other_b.id != block.id :
                total+=Block.overlap_area(block,other_b)
        return total

    def add(self,cell_id,block_id):
        x,z = cell_id
        if not(0<=x<self.size[1]) or not(0<=z<self.size[2]): 
            raise Exception('Insertion out of bounds')
        block = Block((self.size[0]-1,x,z),block_id)
        while(block.h >=0):
            pedestal_area = self.total_overlap(block)
            if pedestal_area>0.5:
                break
            elif 0 < pedestal_area <= 0.5:
                raise UnstableException(pedestal_area)
            else : 
                block.h -= 1
        block.h+=1
        if block.h >= self.size[0]:
            raise Exception('Column is full')
        else : 
            self.blocks_by_level[block.h].append(block)
            self.blocks_by_id[block_id] = block

    def to_semantic_config(self,semantic_operator):
        config = semantic_operator.empty()
        for pred_id in range(len(config)):
            pred_name = semantic_operator.pred_id_to_pred_name[pred_id]
            obj_a,obj_b = semantic_operator.pred_id_to_edge[pred_id]
            if (obj_a in self.blocks_by_id and obj_b in self.blocks_by_id and 
                self.predicates[pred_name](self.blocks_by_id[obj_a],self.blocks_by_id[obj_b])):
                config = semantic_operator.predicates[pred_name](config,obj_a,obj_b,True)
        return config

    def to_nx_graph(self,semantic_operator):
        '''
            Return a graph of the current grid.
            Blocks with all predicates false aren't present in the graph. 
        '''
        multi_di_graph = nx.MultiDiGraph()
        for pred_id in range(len(semantic_operator.empty())):
            pred_name = semantic_operator.pred_id_to_pred_name[pred_id]
            obj_a,obj_b = semantic_operator.pred_id_to_edge[pred_id]
            if (obj_a in self.blocks_by_id and obj_b in self.blocks_by_id and 
                self.predicates[pred_name](self.blocks_by_id[obj_a],self.blocks_by_id[obj_b])):
                multi_di_graph.add_edge(obj_a,obj_b,label=pred_name)
                if pred_name == 'close':
                    multi_di_graph.add_edge(obj_b,obj_a,label=pred_name)

        return multi_di_graph

    def to_nx_graph_hash(self,semantic_operator):
        nx_graph = self.to_nx_graph(semantic_operator)
        return nx.weisfeiler_lehman_graph_hash(nx_graph)
                
    def create(size,object_ids,cell_ids) -> 'EnvSimulatorSimplified':
        connect_four = EnvSimulatorSimplified(size)
        for cube_id,cell_id in zip(object_ids,cell_ids): 
            connect_four.add(cell_id,cube_id)
        return connect_four

    def get_top_objects_ids(self):
        top_objects = []
        for obj_id,obj in self.blocks_by_id.items():
            on_top = True
            for other_id,other in self.blocks_by_id.items():
                if obj_id!=other_id and Block.is_above(other,obj):
                    on_top = False
                    break
            if on_top:
                top_objects.append(obj_id)
        return top_objects
        
    def get_neighbours(self):
        neighbours = []
        all_cells = list(product(np.arange(0,self.size[1],0.5),
                            np.arange(0,self.size[2],0.5)))
        for obj_id in self.get_top_objects_ids():
            for cell_id in all_cells:
                new_grid = copy.deepcopy(self)
                new_grid.remove(obj_id)
                try : 
                    new_grid.add(cell_id,obj_id)
                except UnstableException:
                    continue
                neighbours.append(new_grid)
        return neighbours
            
def generate_ordered_tree_from_start(nb_block,start_grid,GANGSTR=True):
    semantic_operator = SemanticOperation(nb_block,GANGSTR)
    to_explore = queue.Queue()
    to_explore.put((0,start_grid))
    explored = bidict({start_grid.to_semantic_config(semantic_operator):0})
    nk_graph = nk.Graph(1,weighted=True, directed=True)
    while not to_explore.empty() :
        parent_id,grid = to_explore.get()
        childs = grid.get_neighbours()
        for child_grid in childs : 
            child_config = child_grid.to_semantic_config(semantic_operator)
            if child_config not in explored:
                child_id = nk_graph.addNode()
                to_explore.put((child_id,child_grid))
                nk_graph.addEdge(parent_id,child_id)
                explored[child_config]= child_id
                if len(explored)%1000 == 0 :
                    print(len(explored))
        
    return explored,nk_graph

def generate_unordered_tree_from_start(nb_block,start_grid:EnvSimulatorSimplified,GANGSTR=True):
    semantic_operator = SemanticOperation(nb_block,GANGSTR)
    to_explore = queue.Queue()
    to_explore.put((0,start_grid))
    explored_graph_hash = bidict({start_grid.to_nx_graph_hash(semantic_operator):0})
    explored_sem = bidict({start_grid.to_semantic_config(semantic_operator):0})
    nk_graph = nk.Graph(1,weighted=False, directed=False)
    while not to_explore.empty() :
        parent_id,grid = to_explore.get()
        childs = grid.get_neighbours()
        for child_grid in childs : 
            child_graph_hash = child_grid.to_nx_graph_hash(semantic_operator)
            if child_graph_hash not in explored_graph_hash:
                child_id = nk_graph.addNode()
                to_explore.put((child_id,child_grid))
                nk_graph.addEdge(parent_id,child_id)
                explored_graph_hash[child_graph_hash]= child_id
                explored_sem[child_grid.to_semantic_config(semantic_operator)] = child_id
                if len(explored_graph_hash)%1000 == 0 :
                    print(len(explored_graph_hash))
        
    return nk_graph,explored_graph_hash,explored_sem
