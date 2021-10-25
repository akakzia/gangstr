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
from rollout import HMERolloutWorker
from goal_sampler import GoalSampler
from utils import init_storage, get_eval_goals
import time
from mpi_utils import logger
import networkit as nk
from graph.semantic_graph import SemanticGraph
from graph.UnorderedSemanticGraph import UnorderedSemanticGraph
from graph.agent_network import AgentNetwork
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
    if rank == 0 and args.evaluations:
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
    rollout_worker = HMERolloutWorker(env, policy, goal_sampler,  args)

    # If non existent, create expert knowledge graph
    if rank == 0 and not os.path.isdir('data'):
        generate_expert_graph(args.n_blocks,True)

    MPI.COMM_WORLD.Barrier()
    # initialize graph components : 
    # sem_op = SemanticOperation(args.n_blocks,True)
    # configs = bidict()
    nk_graph = nk.Graph(0,weighted=True, directed=True)
    if args.unordered_edge:
        semantic_graph = UnorderedSemanticGraph(bidict(),nk_graph,args.n_blocks,True,args=args)
    else :
        semantic_graph = SemanticGraph(bidict(),nk_graph,args.n_blocks,True,args=args)
    agent_network = AgentNetwork(semantic_graph,logdir,args)
    agent_network.teacher.compute_frontier(agent_network.semantic_graph)

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
            episodes = rollout_worker.train_rollout(agent_network= agent_network,
                                                    time_dict=time_dict)

            time_dict['rollout'] += time.time() - t_i

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
            if args.n_blocks == 5:
                instructions = ['close_1', 'close_2', 'close_3', 'stack_2', 'stack_3', '2stacks_2_2', '2stacks_2_3', 'pyramid_3',
                                'mixed_2_3', 'stack_4', 'stack_5']
            else:
                raise NotImplementedError
            for instruction in instructions:
                eval_goal = get_eval_goals(instruction, n=args.n_blocks)
                eval_goals.append(eval_goal.squeeze(0))
            eval_goals = np.array(eval_goals)
            episodes = rollout_worker.test_rollout(eval_goals,agent_network,
                                                        episode_duration=args.episode_duration,
                                                        animated=False)
            results = np.array([e['success'][-1].astype(np.float32) for e in episodes])
            rewards = np.array([e['rewards'][-1] for e in episodes])
            all_results = MPI.COMM_WORLD.gather(results, root=0)

            time_dict['eval'] += time.time() - t_i

            # synchronize goals count per class in teacher
            synchronized_stats, sync_nb_ss, sync_nb_beyond = sync(agent_network.teacher.stats, agent_network.teacher.ss_interventions,
                                                                  agent_network.teacher.beyond_interventions)
            # internalized goal pairs
            nb_pairs_internalized = len(rollout_worker.stepping_stones_beyond_pairs_list)
            # nb_ss_internalized = len(rollout_worker.stepping_stones_list)
            nb_removed_in_internalization = len(rollout_worker.to_remove_internalization)
            nb_removed_in_individual = len(rollout_worker.to_remove_individual)

            # Logs
            if rank == 0:
                assert len(all_results) == args.num_workers# MPI test
                av_res = np.array(all_results).mean(axis=0)
                global_sr = np.mean(av_res)
                
                agent_network.log(logger)
                logger.record_tabular('replay_nb_edges', policy.buffer.get_nb_edges())
                log_and_save(goal_sampler, synchronized_stats, sync_nb_ss, sync_nb_beyond, agent_network.stats, epoch, episode_count,
                             av_res, global_sr, time_dict, nb_pairs_internalized, nb_removed_in_internalization,
                             nb_removed_in_individual)

                # Saving policy models
                if epoch % args.save_freq == 0:
                    policy.save(model_path, epoch)
                    agent_network.save(model_path,epoch)
                if rank==0: logger.info('\tEpoch #{}: SR: {}'.format(epoch, global_sr))


def log_and_save( goal_sampler, teacher_stats, proposed_ss, proposed_beyond, agent_stats, epoch, episode_count, av_res, global_sr,
                  time_dict, nb_internalized, nb_removed_inter, nb_removed_indiv):
    goal_sampler.save(epoch, episode_count, av_res, global_sr, time_dict, teacher_stats, agent_stats,
                      nb_internalized, nb_removed_inter, nb_removed_indiv, proposed_ss, proposed_beyond)
    for k, l in goal_sampler.stats.items():
        logger.record_tabular(k, l[-1])
    logger.dump_tabular()

def sync(x, a, b):
    """ x: dictionary of counts for every class_goal proposed by the teacher
        return the synchronized dictionary among all cpus """
    res = x.copy()
    for k in x.keys():
        res[k] = MPI.COMM_WORLD.allreduce(x[k], op=MPI.SUM)
    sync_a = MPI.COMM_WORLD.allreduce(a, op=MPI.SUM)
    sync_b = MPI.COMM_WORLD.allreduce(b, op=MPI.SUM)
    return res, sync_a, sync_b


if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # Get parameters
    args = get_args()

    launch(args)
