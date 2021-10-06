import random
import numpy as np
from graph.semantic_graph import SemanticGraph
from mpi4py import MPI
from graph.teacher import Teacher
import pickle

class AgentNetwork():
    
    def __init__(self,semantic_graph :SemanticGraph,exp_path,args):
        self.teacher = Teacher(args)
        self.semantic_graph = semantic_graph
        self.args = args
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.exp_path = exp_path

        self.init_stats()
        
    def update(self,episodes):
        all_episodes = MPI.COMM_WORLD.allgather(episodes)
        all_episode_list = [e for eps in all_episodes 
                                for e in eps] # flatten the list of episodes gathered by all actors
        # update agent graph : 
        for e in all_episode_list:
            # Verify if last achieved goal is stable during last 10 steps
            condition = True
            i = -1
            while condition and i > -10:
                condition = (str(e['ag'][-i]) == str(e['ag'][-i-1]))
                i -= 1
            if condition:
                start_config = tuple(e['ag'][0])
                achieved_goal = tuple(e['ag'][-1])
                goal = tuple(e['g'][-1])
                success = e['success'][-1]

                # update agent count stats
                try:
                    c = self.teacher.config_to_class[str(np.array(achieved_goal).reshape(1, -1))]
                    if self.stats[c+1] < 10000: # avoid float limit instabilities
                        self.stats[c+1] += 1
                except KeyError:
                    pass

                self.semantic_graph.create_node(start_config)
                self.semantic_graph.create_node(achieved_goal)

                if self.semantic_graph.getNodeId(goal)!=None:
                    self.update_or_create_edge(start_config,goal,success)

                # hindsight edge creation :
                if self.args.local_hindsight_edges: # consider add all unique transition inside an episode.
                    unique_configurations = [tuple(e['ag'][0])]
                    for i in range(1,len(e['ag'])):
                        if not (e['ag'][i] == e['ag'][i-1]).all():
                            config = tuple(e['ag'][i])
                            unique_configurations.append(config)
                            self.semantic_graph.create_node(config)
                    hindsight_edges = list(zip(unique_configurations[:-1],unique_configurations[1:]))
                    for ag,ag_next in hindsight_edges:
                        if not self.semantic_graph.hasEdge(ag,ag_next):
                            self.semantic_graph.create_edge_stats((ag,ag_next),self.args.edge_prior)
                else :
                    if (achieved_goal != goal and start_config != achieved_goal
                        and not self.semantic_graph.hasEdge(start_config,achieved_goal)):
                            self.semantic_graph.create_edge_stats((start_config,achieved_goal),self.args.edge_prior)

        # update frontier :  
        self.semantic_graph.update()
        self.teacher.computeFrontier(self.semantic_graph)
    
    def update_or_create_edge(self,start,end,success):
        if (start!=end):
            if not self.semantic_graph.hasEdge(start,end):
                self.semantic_graph.create_edge_stats((start,end),self.args.edge_prior)
            self.semantic_graph.update_edge_stats((start,end),success)

    def get_path(self,start,goal,algorithm='dijkstra'):
        if self.args.expert_graph_start: 
            return self.teacher.oracle_graph.sample_shortest_path(start,goal,algorithm=algorithm)
        else : 
            return self.semantic_graph.sample_shortest_path(start,goal,algorithm=algorithm)

    def get_path_from_coplanar(self,target):
        if self.args.expert_graph_start : 
            return self.teacher.oracle_graph.get_path_from_coplanar(target)
        else : 
            return self.semantic_graph.get_path_from_coplanar(target)

    def sample_goal_uniform(self,nb_goal,use_oracle=True):
        if use_oracle:
            return self.teacher.sample_goal_uniform(nb_goal)
        else :
            return random.choices(self.semantic_graph.configs.inverse,k=nb_goal)

    def sample_goal_in_frontier(self,current_node,k):
        return self.teacher.sample_in_frontier(current_node,self.semantic_graph,k)
    
    def sample_from_frontier(self,frontier_node,k):
        return self.teacher.sample_from_frontier(frontier_node,self.semantic_graph,k)

    def sample_rand_neighbour(self,source,excluding = []):
        neighbours = list(filter( lambda x : x not in excluding, self.semantic_graph.iterNeighbors(source)))
        if neighbours:
            return random.choice(neighbours)
        else : 
            return None

    def sample_neighbour_based_on_SR_to_goal(self,source,reversed_dijkstra,goal, excluding = []):

        neighbors = [ n for n  in self.semantic_graph.iterNeighbors(source) if n not in excluding]

        if len(neighbors)>0:
            _,source_sr,_ = self.semantic_graph.sample_shortest_path_with_sssp(source,goal,reversed_dijkstra,reversed=True)

            source_to_neighbors_sr,neighbors_to_goal_sr,_ = self.semantic_graph.get_neighbors_to_goal_sr(source,neighbors,goal,reversed_dijkstra)
            source_to_neighbour_to_goal_sr = source_to_neighbors_sr*neighbors_to_goal_sr
            
            # remove neighbors with SR lower than current node : 
            inds = neighbors_to_goal_sr>source_sr
            neighbors = np.array(neighbors)[inds]
            source_to_neighbour_to_goal_sr = source_to_neighbour_to_goal_sr[inds]

            # filter neighbors :  
            # Among multiple neighbors belonging to the same unordered edge, only keep one by sampling among highest SR neighbor_to_goal
            edges = [self.semantic_graph.edge_config_to_edge_id((source,tuple(neigh)))for neigh in neighbors]
            edges,inv_ids =  np.unique(np.array(edges),return_inverse = True)
            filtered_ids = np.empty_like(edges)
            for i,e in enumerate(edges):
                e_neigh_ids = np.where(inv_ids == i)[0]
                e_sr = neighbors_to_goal_sr[e_neigh_ids] 
                highest_neighbors_ids = e_neigh_ids[np.argwhere(e_sr == np.amax(e_sr)).flatten()]
                choosen_neighbor_id = np.random.choice(highest_neighbors_ids)
                filtered_ids[i] = choosen_neighbor_id
            neighbors = neighbors[filtered_ids]
            source_to_neighbour_to_goal_sr = source_to_neighbour_to_goal_sr[filtered_ids]
            
            # only keep k_ largest probs : 
            if len(source_to_neighbour_to_goal_sr) > self.args.rollout_exploration_k:
                inds = np.argpartition(source_to_neighbour_to_goal_sr, -self.args.rollout_exploration_k)[-self.args.rollout_exploration_k:]
                neighbors = np.array(neighbors)[inds]
                source_to_neighbour_to_goal_sr = source_to_neighbour_to_goal_sr[inds]
            sr_sum = np.sum(source_to_neighbour_to_goal_sr)
            if sr_sum == 0 :
                return None
            else : 
                probs = source_to_neighbour_to_goal_sr/sr_sum
                neighbour_id = np.random.choice(range(len(neighbors)),p = probs)
                return tuple(neighbors[neighbour_id])
        else : 
            return None

    def log(self,logger):
        self.semantic_graph.log(logger)
        # TODO : , à change selon qu'on soit unordered ou pas. 
        logger.record_tabular('frontier_len',len(self.teacher.agent_frontier))

    def save(self,model_path, epoch):
        self.semantic_graph.save(model_path+'/',f'{epoch}')
        with open(f"{model_path}/frontier_{epoch}.config", 'wb') as f:
            pickle.dump(self.teacher.agent_frontier,f,protocol=pickle.HIGHEST_PROTOCOL)
            
    def load(model_path,epoch,args) ->'AgentNetwork':
        semantic_graph = SemanticGraph.load(model_path,f'{epoch}')
        with open(f"{model_path}frontier_{epoch}.config", 'rb') as f:
            frontier = pickle.load(f)
        agent_network = AgentNetwork(semantic_graph,None,args)
        agent_network.teacher.agent_frontier = frontier
        return agent_network

    def sync(self):
        self.teacher.agent_frontier = MPI.COMM_WORLD.bcast(self.teacher.agent_frontier, root=0)
        if self.rank == 0:
            self.semantic_graph.save(self.exp_path+'/','temp')

        MPI.COMM_WORLD.Barrier()
        if self.rank!=0:
            self.semantic_graph = SemanticGraph.load(self.exp_path+'/','temp')

    def init_stats(self):
        self.stats = dict()
        for i in range(11):
            self.stats[i+1] = 0