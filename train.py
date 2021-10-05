from typing import DefaultDict
from bidict import bidict
import numpy as np
from mpi4py import MPI
import gym
import os
from arguments import get_args
from rl_modules.rl_agent import RLAgent
import random
import torch
from rollout import TeacherGuidedRolloutWorker,NeighbourRolloutWorker,GANGSTR_RolloutWorker
from goal_sampler import GoalSampler
from utils import init_storage, get_eval_goals
import time
from mpi_utils import logger
import networkit as nk
from graph.semantic_graph import SemanticGraph
from graph.UnorderedSemanticGraph import UnorderedSemanticGraph
from graph.agent_network import AgentNetwork
from graph.SemanticOperation import SemanticOperation
from generate_graph import generate_expert_graph
import env


def get_env_params(env):
    obs = env.reset()

    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

def launch(args):

    rank = MPI.COMM_WORLD.Get_rank()

    t_total_init = time.time()

    # Make the environment
    args.env_name = 'FetchManipulate{}Objects-v0'.format(args.n_blocks)
    env = gym.make(args.env_name)

    # set random seeds for reproducibility
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # get saving paths
    logdir = None
    if rank == 0:
        logdir, model_path = init_storage(args)
        logger.configure(dir=logdir)
        logger.info(vars(args))
    logdir = MPI.COMM_WORLD.bcast(logdir, root=0)

    args.env_params = get_env_params(env)

    goal_sampler = GoalSampler(args)

    # Initialize RL Agent
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward, goal_sampler)
    else:
        raise NotImplementedError

    # Initialize Rollout Worker
    if args.rollout_goal_generator == 'teacher':
        rollout_worker = TeacherGuidedRolloutWorker(env, policy, goal_sampler,  args)
    elif args.rollout_goal_generator == 'neighbour':
        rollout_worker = NeighbourRolloutWorker(env, policy, goal_sampler,  args)
    elif args.rollout_goal_generator == 'known_unif':
        rollout_worker = GANGSTR_RolloutWorker(env, policy, goal_sampler,  args)

    # create graph if necessary
    if rank == 0 and not os.path.isdir('data'):
        generate_expert_graph(args.n_blocks,True)
    MPI.COMM_WORLD.Barrier()
    # initialize graph components : 
    sem_op = SemanticOperation(args.n_blocks,True)
    configs = bidict()
    nk_graph = nk.Graph(0,weighted=True, directed=True)
    if args.unordered_edge:
        semantic_graph = UnorderedSemanticGraph(configs,nk_graph,args.n_blocks,True,args=args)
    else : 
        semantic_graph = SemanticGraph(configs,nk_graph,args.n_blocks,True,args=args)
    agent_network = AgentNetwork(semantic_graph,logdir,args)
    agent_network.teacher.computeFrontier(agent_network.semantic_graph)

    # Main interaction loop
    episode_count = 0
    for epoch in range(args.n_epochs):
        t_init = time.time()

        # setup time_tracking
        time_dict = DefaultDict(int)

        # log current epoch
        if rank == 0: logger.info('\n\nEpoch #{}'.format(epoch))

        # Cycles loop
        for _ in range(args.n_cycles):

            # Environment interactions
            t_i = time.time()
            episodes = rollout_worker.train_rollout(agentNetwork= agent_network, 
                                                    max_episodes=args.num_rollouts_per_mpi,
                                                    episode_duration=args.episode_duration,
                                                    time_dict=time_dict)
            time_dict['rollout'] += time.time() - t_i

            # Goal Sampler updates
            t_i = time.time()
            # episodes = goal_sampler.update(episodes, episode_count)
            time_dict['gs_update'] += time.time() - t_i

            # Storing episodes
            t_i = time.time()
            policy.store(episodes)
            time_dict['store'] += time.time() - t_i

            # Agent Network Update : 
            t_i = time.time()
            agent_network.update(episodes)
            time_dict['update_graph'] += time.time() - t_i

            # Updating observation normalization
            t_i = time.time()
            for e in episodes:
                policy._update_normalizer(e)
            time_dict['norm_update'] += time.time() - t_i

            # Policy updates
            t_i = time.time()
            for _ in range(args.n_batches):
                policy.train()
            time_dict['policy_train'] += time.time() - t_i
            episode_count += len(episodes) * args.num_workers


        time_dict['epoch'] += time.time() -t_init
        time_dict['total'] = time.time() - t_total_init

        if args.evaluations:
            if rank==0: logger.info('\tRunning eval ..')
            # Performing evaluations
            t_i = time.time()
            eval_goals = []
            if args.n_blocks == 3:
                instructions = ['close_1', 'close_2', 'close_3', 'stack_2', 'pyramid_3', 'stack_3']
            elif args.n_blocks == 5:
                instructions = ['close_1', 'close_2', 'close_3', 'stack_2', 'stack_3', '2stacks_2_2', '2stacks_2_3', 'pyramid_3',
                                'mixed_2_3', 'stack_4', 'stack_5']
            else:
                raise NotImplementedError
            for instruction in instructions:
                eval_goal = get_eval_goals(instruction, n=args.n_blocks)
                eval_goals.append(eval_goal.squeeze(0))
            eval_goals = np.array(eval_goals)
            eval_masks = np.array(np.zeros((eval_goals.shape[0], args.n_blocks * (args.n_blocks - 1) * 3 // 2)))
            episodes = rollout_worker.test_rollout(eval_goals,agent_network,
                                                        episode_duration=args.episode_duration,
                                                        animated=False)

            results = np.array([e['success'][-1].astype(np.float32) for e in episodes])
            rewards = np.array([e['rewards'][-1] for e in episodes])
            all_results = MPI.COMM_WORLD.gather(results, root=0)
            all_rewards = MPI.COMM_WORLD.gather(rewards, root=0)
            time_dict['eval'] += time.time() - t_i

            # synchronize goals count per class in teacher
            synchronized_stats = sync(agent_network.teacher.stats)
            # synchronize agent's achieved goals classes
            # synchronized_agent_stats = sync(agent_network.stats)

            # Logs
            if rank == 0:
                assert len(all_results) == args.num_workers  # MPI test
                av_res = np.array(all_results).mean(axis=0)
                av_rewards = np.array(all_rewards).mean(axis=0)
                global_sr = np.mean(av_res)
                
                agent_network.log(logger)
                logger.record_tabular('replay_nb_edges', policy.buffer.get_nb_edges())
                log_and_save(goal_sampler, synchronized_stats, agent_network.stats, epoch, episode_count, av_res, av_rewards, global_sr, time_dict)

                # Saving policy models
                if epoch % args.save_freq == 0:
                    policy.save(model_path, epoch)
                    agent_network.save(model_path,epoch)
                if rank==0: logger.info('\tEpoch #{}: SR: {}'.format(epoch, global_sr))


def log_and_save( goal_sampler, teacher_stats, agent_stats, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
    goal_sampler.save(epoch, episode_count, av_res, av_rew, global_sr, time_dict, teacher_stats, agent_stats)
    for k, l in goal_sampler.stats.items():
        logger.record_tabular(k, l[-1])
    logger.dump_tabular()

def sync(x):
    """ x: dictionary of counts for every class_goal proposed by the teacher
        return the synchronized dictionary among all cpus """
    res = x.copy()
    for k in x.keys():
        res[k] = MPI.COMM_WORLD.allreduce(x[k], op=MPI.SUM)
    return res


if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # Get parameters
    args = get_args()

    launch(args)
