

from collections import defaultdict

from numpy.core.shape_base import block
from graph.SemanticOperation import SemanticOperation, all_stack_trajectories, config_to_unique_str,config_permutations
from graph.semantic_graph import SemanticGraph
from graph.UnorderedSemanticGraph import UnorderedSemanticGraph
from graph.agent_network import AgentNetwork
from utils import get_eval_goals
import numpy as np
import math 
import networkit as nk
from types import SimpleNamespace
from bidict import bidict

def start_test(name):
    print()
    print('_'*10,name,'_'*10)
    print('_'*20 + '_'*(len(name)+2))


def number_of_one_object_change_edges(semantic_graph: SemanticGraph):
    sem_op = SemanticOperation(semantic_graph.nb_blocks,True)
    n_object_move_edges = 0
    all_edges = list(semantic_graph.nk_graph.iterEdges())
    for (n1,n2) in all_edges:
        c1,c2 = semantic_graph.getConfig(n1),semantic_graph.getConfig(n2)
        n_object_move_edges+= sem_op.one_object_edge((c1,c2))
    print(f'number of one objects moves :{n_object_move_edges}/{len(all_edges)}',)

def check_one_object_edge(block_size):
    sem_op = SemanticOperation(block_size,True)
    semantic_graph = SemanticGraph.load_oracle(block_size)
    number_of_one_object_change_edges(semantic_graph)
    end = sem_op.close_and_above(sem_op.empty(),0,1,True)
    end = sem_op.close_and_above(end,2,3,True)
    # print(sem_op.one_object_edge((sem_op.empty(),end)))
    print('')

def create_simple_graph(block_size):
    sem_op = SemanticOperation(block_size,True)
    args = SimpleNamespace()
    args.n_blocks = block_size
    args.rollout_exploration_k = 4
    args.one_object_edge = False
    configs = bidict()
    nk_graph = nk.Graph(0,weighted=True, directed=True)
    semantic_graph = UnorderedSemanticGraph(configs,nk_graph,args.n_blocks,True,args=args)
    start = semantic_graph.empty()
    close1 = sem_op.close(start,3,4,True)
    stack2 = sem_op.close_and_above(start,1,0,True)
    stack3 = sem_op.close_and_above(stack2,2,1,True)

    semantic_graph.create_node(start)
    semantic_graph.create_node(close1)
    semantic_graph.create_node(stack2)
    semantic_graph.create_node(stack3)
    semantic_graph.create_edge_stats((start,close1),0.5)
    semantic_graph.create_edge_stats((start,stack2),0.5)
    semantic_graph.create_edge_stats((close1,stack2),0.5)
    semantic_graph.create_edge_stats((stack2,stack3),0.5)
    
    return semantic_graph,start,close1,stack2,stack3


def check_sample_path(block_size):
        # init and construct simple graph : 
    semantic_graph,start,close1,stack2,stack3 = create_simple_graph(block_size)
        # check path sampling : 
    paths,scores = semantic_graph.k_shortest_path(start,stack3,20,unordered_bias=False)
    assert scores[0] == 0.25 and len(paths[0]) == 3,'error best path'
    assert len(scores) == 4 ,'missing paths'
    assert all ([len(path)==4 for path in paths[1:]] ) and all ([score==0.125 for score in scores[1:]] ),'error secondary paths'
    
    paths,scores = semantic_graph.k_shortest_path(start,stack3,20,unordered_bias=True)
    assert scores[0] == 0.25 and len(paths[0]) == 3,'error best path'
    assert len(scores) == 2 ,'missing paths'
    assert scores[1] == 0.125 and len(paths[1]) == 4,'error secondary path'

    print(f"start:{semantic_graph.getNodeId(start)},close1:{semantic_graph.getNodeId(close1)},stack2:{semantic_graph.getNodeId(stack2)},stack3:{semantic_graph.getNodeId(stack3)} ")
    # print(list(zip(paths,scores)))

    paths,scores = semantic_graph.k_shortest_path(start,stack3,20,use_weights=False,unordered_bias=True)
    assert scores[0] == 2 and len(paths[0]) == 3,'error best path'
    assert len(scores) == 2 ,'missing paths'
    assert scores[1] == 3 and len(paths[1]) == 4,'error secondary path'

    print(f"start:{semantic_graph.getNodeId(start)},close1:{semantic_graph.getNodeId(close1)},stack2:{semantic_graph.getNodeId(stack2)},stack3:{semantic_graph.getNodeId(stack3)} ")
    # print(list(zip(paths,scores)))

def check_sample_neighbors(block_size):

    semantic_graph,start,close1,stack2,stack3 = create_simple_graph(block_size)
    agent_network = AgentNetwork(semantic_graph,None,semantic_graph.args)
    reversed_dijktra = agent_network.semantic_graph.get_sssp_to_goal(stack2)
    res = agent_network.sample_neighbour_based_on_SR_to_goal(start,reversed_dijktra,stack2)
    print(res)



def check_semantic_hash(block_size):
    unordered_sem_graph = SemanticGraph.load(SemanticGraph.ORACLE_PATH,
                                            f"{SemanticGraph.ORACLE_NAME}{block_size}_unordered"
                                            )
    sem_op = unordered_sem_graph.semantic_operation
    all_hash = set()
    for config in unordered_sem_graph.configs:
        all_perm = set(config_permutations(config,sem_op))
        t = [sem_op.to_nx_graph_hash(c) for c in all_perm]
        graph_hash = set([sem_op.to_nx_graph_hash(c) for c in all_perm])
        assert len(graph_hash) == 1
        all_hash.add(graph_hash.pop())
    assert len(all_hash) == len(unordered_sem_graph.configs),f'error in : {len(all_hash)}!={len(unordered_sem_graph.configs)}'

    print('semantic hash diversity ok for ',block_size)
 
def graph_overview(nb_block):
    start_test(f'overview for {nb_block}')
    semantic_graph = SemanticGraph.load_oracle(nb_block)
    nk.overview(semantic_graph.nk_graph)
    print(f"frontier size : {len(semantic_graph.get_frontier_nodes())}")
        

def check_generate_goals(nb_block,unordered_edge = False):
    ''' Check frontier goal sampling'''
    GANGSTR = True
    semantic_operator = SemanticOperation(nb_block,GANGSTR)
    configs = bidict()
    args = SimpleNamespace()
    nk_graph = nk.Graph(0,weighted=True, directed=True)

    args.n_blocks = nb_block
    if unordered_edge:
        semantic_graph = UnorderedSemanticGraph(configs,nb_block,True,args=args)
    else : 
        semantic_graph = SemanticGraph(configs,nk_graph,args.n_blocks,True,args=args)
    agentNetwork = AgentNetwork(semantic_graph,'',args)
    print([agentNetwork.semantic_graph.configs.inverse[n] for n in  agentNetwork.teacher.agent_frontier])

    agentNetwork.semantic_graph.create_node(semantic_operator.empty())
    agentNetwork.teacher.computeFrontier(agentNetwork.semantic_graph)
    print([agentNetwork.semantic_graph.configs.inverse[n] for n in  agentNetwork.teacher.agent_frontier])

    stack2 = semantic_operator.close_and_above(semantic_operator.empty(),1,0,True)
    agentNetwork.semantic_graph.create_node(stack2)
    agentNetwork.teacher.computeFrontier(agentNetwork.semantic_graph)
    print([agentNetwork.semantic_graph.configs.inverse[c] for c in  agentNetwork.teacher.agent_frontier])

    stack3 = semantic_operator.close_and_above(stack2,2,1,True)
    agentNetwork.semantic_graph.create_node(stack3)
    agentNetwork.teacher.computeFrontier(agentNetwork.semantic_graph)
    print([agentNetwork.semantic_graph.configs.inverse[c] for c in  agentNetwork.teacher.agent_frontier])

def check_stack_number(block_size):
    start_test(f"check_stack_number {block_size}")
    # load graph 
    semantic_graph = SemanticGraph.load_oracle(block_size)
    semantic_operator = SemanticOperation(block_size,True)
    # nk.overview(semantic_graph.nk_graph)
    stacks = all_stack_trajectories(block_size)
    all_good = True 
    for s,config_path in stacks.items():
        result,path_length = check_goal(semantic_graph,semantic_operator,config_path[-1],len(s)-1,s,True)
        if result != 'success': 
            all_good = False
    if all_good:
        print(f'Succeed all stacks of {block_size}')

def test_eval_goals(block_size):
    start_test(f"test_eval_goals {block_size}")
    # load graph 
    semantic_graph = SemanticGraph.load_oracle(block_size)
    semantic_operator = SemanticOperation(block_size,True)
    if block_size == 3:
        instructions_to_length = {'close_1':1, 'close_2':2, 'close_3':2, 'stack_2':1, 'pyramid_3':2, 'stack_3':2}
    elif block_size == 5:
        instructions_to_length = {'close_1':1, 'close_2':2, 'close_3':3, 'stack_2':1, 'stack_3':2, '2stacks_2_2':2, '2stacks_2_3':3, 'pyramid_3':2,
                        'mixed_2_3':4, 'trapeze_2_3':3, 'stack_4':3, 'stack_5':4}
    all_good = True 
    for instruction,true_length in instructions_to_length.items():
        error_count = defaultdict(int)
        goals = np.unique(get_eval_goals(instruction,block_size,math.factorial(block_size)*10),axis=0)
        for goal in goals:
            goal = tuple(goal)
            # if instruction == 'trapeze_2_3':
            #     print(config_to_unique_str(goal,semantic_operator))
            result,path_length = check_goal(semantic_graph,semantic_operator,goal,true_length,instruction)
            error_count[result]+=1
        if result=='missing_goal' : 
            print("missing_goal for ",instruction , f"({error_count['missing_goal']} failed /{len(goals)})")
            all_good = False
        elif result=='no_path' : 
            print("no_path ",instruction, f"({error_count['no_path']} failed /{len(goals)})")
            all_good = False
        elif result=='wrong_path_size' :
            all_good = False
            print(f"wrong_path_size, should be {true_length}  ",instruction, f"({error_count['wrong_path_size']} failed /{len(goals)})")
    if all_good:
        print(f'Succeed tests {block_size}')


def check_goal(semantic_graph,semantic_operator,goal,true_length,node_name,verbose=False):
    result = 'success'
    path_length = None
    if goal not in semantic_graph.configs:
        result = 'missing_goal'
    else : 
        path,sr,dist = semantic_graph.sample_shortest_path(semantic_operator.empty(),goal)
        path_length = len(path)-1
        if path == []:
            result = 'no_path'
        elif path_length!=true_length: 
            result = f'wrong_path_size'
    if verbose and result!='success':
            print(f'{result} : ',node_name)
    return result,path_length


if __name__ == '__main__':
    block_sizes = [5]
    for block_size in block_sizes:
        check_sample_path(block_size)
        check_sample_neighbors(block_size)
        # check_one_object_edge(block_size)
        # check_semantic_hash(block_size)
        graph_overview(block_size)
        check_stack_number(block_size)        
        test_eval_goals(block_size)
        # check_generate_goals(block_size)
