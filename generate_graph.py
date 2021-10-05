import os
import time
from graph.SemanticOperation import SemanticOperation
from graph.connect4_3d_Simulator import EnvSimulatorSimplified,generate_unordered_tree_from_start
from graph.semantic_graph import SemanticGraph,augment_with_all_permutation
import networkit as nk
import matplotlib.pyplot as plt
import time
from graph.SemanticOperation import SemanticOperation
import networkit as nk

PATH = 'data/'

def generate_unordered_network(nb_blocks,GANGSTR):
    '''
        Generate nk network with configs as nodes and save it in file.
        (Blocks are only ordered one way).
    '''
    # start = time.time()
    semantic_operation = SemanticOperation(nb_blocks,GANGSTR)

    # choose inital grid with blocks far appart.
    remote_places = [(0,0),(0,3.5),(0,7),(0,10.5),(0,14),(0,17.5)]
    grid_size = (nb_blocks,2,18)
    start_grid = EnvSimulatorSimplified.create(grid_size,list(range(nb_blocks)),remote_places[:nb_blocks])
    assert start_grid.to_semantic_config(semantic_operation) == semantic_operation.empty()
    
    nk_graph,explored_graph_hash,explored_sem = generate_unordered_tree_from_start(nb_blocks,start_grid)
    sem_graph = SemanticGraph(explored_sem,nk_graph,nb_blocks,GANGSTR)
    sem_graph.save(SemanticGraph.ORACLE_PATH,f"{SemanticGraph.ORACLE_NAME}{nb_blocks}_unordered")
    
    # nk.overview(nk_graph)
    # elapsed = time.time()-start
    # print(f"elapsed : {nb_blocks} unordered",elapsed)

    # nk.viztasks.drawGraph(nk_graph,with_labels =True)
    # plt.savefig(f'network{nb_blocks}.png')
    # plt.close()

    # for n in nk_graph.iterNodes():
    #     print(f"{n} => {explored_sem.inverse[n]}")

def generate_ordered_network(nb_blocks,GANGSTR):
    '''
        Load unordered nk network from file and generate all other permutations.
        Rsulting graph is savedc in file.
    '''
    # start = time.time()
        
    unordered_sem_graph = SemanticGraph.load(SemanticGraph.ORACLE_PATH,
                                            f"{SemanticGraph.ORACLE_NAME}{nb_blocks}_unordered")
    nk_graph,explored_sem = augment_with_all_permutation(unordered_sem_graph.nk_graph,
                                                        unordered_sem_graph.configs,
                                                        nb_blocks,GANGSTR=GANGSTR)
    ordered_sem_graph = SemanticGraph(explored_sem,nk_graph,nb_blocks,GANGSTR)
    ordered_sem_graph.save(SemanticGraph.ORACLE_PATH,f"{SemanticGraph.ORACLE_NAME}{nb_blocks}")
    # nk.overview(nk_graph)
    # elapsed = time.time()-start
    # print(f"elapsed : {nb_blocks} ordered",elapsed)
    # if nb_blocks ==3:
    #     nk.viztasks.drawGraph(nk_graph,with_labels =True)
    #     plt.savefig(f'network{nb_blocks}_ordered.png')

def generate_expert_graph(n_blocks,GANGSTR):
    if not os.path.isdir('data'):
        not os.mkdir('data')
    generate_unordered_network(n_blocks,GANGSTR)
    generate_ordered_network(n_blocks,GANGSTR)

if __name__=='__main__':
    GANGSTR= True
    for nb_blocks in [3,5]:
        generate_expert_graph(nb_blocks,GANGSTR)
