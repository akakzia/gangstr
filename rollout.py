import random
from graph.agent_network import AgentNetwork
import numpy as np
from graph.SemanticOperation import SemanticOperation, config_to_name,config_to_unique_str
import time

class RolloutWorker:
    def __init__(self, env, policy, goal_sampler, args):
        self.env = env
        self.policy = policy
        self.env_params = args.env_params
        self.goal_sampler = goal_sampler
        self.goal_dim = args.env_params['goal']
        self.args = args
        self.last_obs = None
        self.reset(False)

        self.exploration_noise_prob = args.exploration_noise_prob

    @property
    def current_config(self):
        return tuple(self.last_obs['achieved_goal'])

    def reset(self,biased_init):
        self.long_term_goal = None
        self.config_path = None
        self.current_goal_id = None
        self.last_episode = None
        self.last_obs = self.env.unwrapped.reset_goal(goal=np.array([None]), biased_init=biased_init)
        self.dijkstra_to_goal = None
        self.state ='GoToFrontier'
    
    def test_rollout(self,goals,agent_network:AgentNetwork,episode_duration, animated=False):
        end_episodes = []
        for goal in goals : 
            self.reset(False)
            _,last_episode = self.guided_rollout(goal,True, agent_network, episode_duration, animated=animated)
            end_episodes.append(last_episode)
        self.reset(False)
        return end_episodes

    def guided_rollout(self,goal,evaluation,agent_network:AgentNetwork,episode_duration,episode_budget=None, animated=False):
        episode = None
        episodes = []
        goal = tuple(goal)
        sem_op = SemanticOperation(5,True)

        if animated : 
            print('goal : ',config_to_unique_str(goal,sem_op))

        if self.current_goal_id == None:
            self.plan(agent_network,goal,evaluation)

        # if len(self.config_path) > self.args.max_path_len:
        #     self.config_path = [self.config_path[0]] + self.config_path[-self.args.max_path_len+1:]

        while True:
            current_goal,goal_dist = self.get_next_goal(agent_network,goal,evaluation) 
            
            if animated:
                print(f'\t{self.current_goal_id} : \n{config_to_unique_str(self.current_config,sem_op)} ->  {config_to_unique_str(current_goal,sem_op)}'  )
                
            episode = self.generate_one_rollout(current_goal, evaluation, episode_duration, animated=False)
            episodes.append(episode)
            self.current_goal_id+=1
            
            success = episodes[-1]['success'][-1]

            if animated:
                print(f'success ',success  )

            if self.current_goal_id == len(self.config_path):
                break
            if episode_budget != None and len(episodes) >= episode_budget:
                break
            if success == False:
                break 

        return episodes,self.last_episode

    def generate_one_rollout(self, goal,evaluation, episode_duration, animated=False):

        g = np.array(goal)
        self.env.unwrapped.target_goal = np.array(goal)
        self.env.unwrapped.binary_goal = np.array(goal)
        obs = self.last_obs['observation']
        ag = self.last_obs['achieved_goal']

        ep_obs, ep_ag, ep_g, ep_actions, ep_success, ep_rewards = [], [], [], [], [], []
        # Start to collect samples
        for _ in range(episode_duration):
            # Run policy for one step
            no_noise = evaluation  # do not use exploration noise if running self-evaluations or offline evaluations
            # feed both the observation and mask to the policy module
            action = self.policy.act(obs.copy(), ag.copy(), g.copy(), no_noise)

            # feed the actions into the environment
            if animated:
                self.env.render()

            observation_new, r, _, _ = self.env.step(action)
            obs_new = observation_new['observation']
            ag_new = observation_new['achieved_goal']

            # Append rollouts
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            ep_g.append(g.copy())
            ep_actions.append(action.copy())
            ep_rewards.append(r)
            ep_success.append((ag_new == g).all())

            # Re-assign the observation
            obs = obs_new
            ag = ag_new

        ep_obs.append(obs.copy())
        ep_ag.append(ag.copy())

        # Gather everything
        episode = dict(obs=np.array(ep_obs).copy(),
                        act=np.array(ep_actions).copy(),
                        g=np.array(ep_g).copy(),
                        ag=np.array(ep_ag).copy(),
                        success=np.array(ep_success).copy(),
                        rewards=np.array(ep_rewards).copy(),
                        self_eval=evaluation)

        self.last_obs = observation_new
        self.last_episode = episode

        return episode

    def plan(self,agent_network,goal,evaluation):

        if evaluation : 
            self.current_goal_id = 1
            self.config_path,_,_ = agent_network.get_path(self.current_config,goal,algorithm = self.args.evaluation_algorithm)
            if not self.config_path:
                self.config_path = [self.current_config,goal]
        elif self.args.rollout_exploration =='sr_and_k_distance':
            self.current_goal_id = 1
            if  np.random.rand()< self.args.rollout_distance_ratio:
                k_best_paths,_ = agent_network.semantic_graph.k_shortest_path(self.current_config,goal,
                                                                                        self.args.rollout_exploration_k,
                                                                                        use_weights = False,
                                                                                        unordered_bias = self.args.unordered_edge)
                self.config_path = random.choices(k_best_paths,k=1)[0] if k_best_paths else None
            else:
                self.config_path,_,_ = agent_network.get_path(self.current_config,goal)
            if not self.config_path:
                self.config_path = [self.current_config,goal]
        elif self.args.rollout_exploration =='sr_and_best_distance':
            self.current_goal_id = 1
            if np.random.rand() < self.args.rollout_distance_ratio:
                self.config_path,_,_ = agent_network.get_path(self.current_config,goal,algorithm='dijkstra')
            else : 
                self.config_path,_,_ = agent_network.get_path(self.current_config,goal,algorithm='bfs')
            if not self.config_path:
                self.config_path = [self.current_config,goal]
        elif self.args.rollout_exploration=='sample_sr' :
            self.current_goal_id = 1
            self.dijkstra_to_goal = agent_network.semantic_graph.get_sssp_to_goal(goal)
        else : raise Exception('unknown exploration method',self.args.rollout_exploration) 
    
    def get_next_goal(self,agent_network,goal,evaluation):
        if evaluation : 
            current_goal = self.config_path[self.current_goal_id]
        elif self.args.rollout_exploration =='sr_and_k_distance':
            current_goal = self.config_path[self.current_goal_id]
        elif self.args.rollout_exploration =='sr_and_best_distance':
            self.plan(agent_network,goal,evaluation)
            current_goal = self.config_path[self.current_goal_id]
        elif self.args.rollout_exploration=='sample_sr' :
            current_goal = None
            if self.dijkstra_to_goal: 
                current_goal = agent_network.sample_neighbour_based_on_SR_to_goal(self.current_config,self.dijkstra_to_goal,goal=goal)
            if current_goal == None: # if no path to goal, try to reach directly 
                current_goal = goal
        else : raise Exception('unknown exploration method') 

        return current_goal,self.current_goal_id

        
class TeacherGuidedRolloutWorker(RolloutWorker):

    def train_rollout(self,agentNetwork:AgentNetwork,episode_duration,max_episodes=None,time_dict=None, animated=False,biased_init=False):
        all_episodes = []

        if np.random.uniform() < self.args.intervention_prob:
            # SP intervenes
            while len(all_episodes) < max_episodes:
                if self.state == 'GoToFrontier':
                    if self.long_term_goal == None :
                        t_i = time.time()
                        self.long_term_goal = next(iter(agentNetwork.sample_goal_in_frontier(self.current_config,1)),None) # first element or None
                        if time_dict:
                            time_dict['goal_sampler'] += time.time() - t_i
                        # if can't find frontier goal, explore directly
                        if self.long_term_goal == None or self.long_term_goal == self.current_config:
                            self.state = 'Explore'
                            continue
                    no_noise = np.random.uniform() > self.exploration_noise_prob
                    episodes,_ = self.guided_rollout(self.long_term_goal,no_noise, agentNetwork, episode_duration,
                                                episode_budget=max_episodes-len(all_episodes),animated=animated)
                    all_episodes += episodes

                    success = episodes[-1]['success'][-1]
                    if not success: # reset at the first failure
                        self.reset(biased_init)
                    elif success and self.current_config == self.long_term_goal:
                        self.state = 'Explore'

                elif self.state =='Explore':
                    t_i = time.time()
                    last_ag = tuple(self.last_obs['achieved_goal'])
                    explore_goal = next(iter(agentNetwork.sample_from_frontier(last_ag,1)),None) # first element or None
                    if time_dict !=None:
                        time_dict['goal_sampler'] += time.time() - t_i
                    if explore_goal:
                        episode = self.generate_one_rollout(explore_goal, False, episode_duration,animated=animated)
                        all_episodes.append(episode)
                        success = episode['success'][-1]
                    if explore_goal == None or  success == False:
                            self.reset(biased_init)
                            continue
                else :
                    raise Exception(f"unknown state : {self.state}")
        else:
            # No SP intervention
            self.reset(biased_init)
            while len(all_episodes) < max_episodes:
                # If no SP intervention
                t_i = time.time()
                if len(agentNetwork.semantic_graph.configs) > 0:
                    self.long_term_goal = agentNetwork.sample_goal_uniform(1, use_oracle=False)[0]
                else:
                    self.long_term_goal = tuple(np.random.choice([-1., 1.], size=(1, self.goal_dim))[0])

                if time_dict != None:
                    time_dict['goal_sampler'] += time.time() - t_i
                if (agentNetwork.semantic_graph.hasNode(self.long_term_goal)
                        and agentNetwork.semantic_graph.hasNode(self.current_config)
                        and self.long_term_goal != self.current_config):
                    new_episodes, _ = self.guided_rollout(self.long_term_goal, evaluation=False,
                                                          agent_network=agentNetwork, episode_duration=episode_duration,
                                                          episode_budget=max_episodes - len(all_episodes))
                else:
                    new_episodes = [self.generate_one_rollout(self.long_term_goal, 1, False, episode_duration, animated)]
                all_episodes += new_episodes
                self.reset(biased_init)
        return all_episodes