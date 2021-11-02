import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
from bidict import bidict
import numpy as np
from rollout import HMERolloutWorker
from goal_sampler import GoalSampler
import random
from mpi4py import MPI
from arguments import get_args
from utils import get_eval_goals
import networkit as nk
from graph.UnorderedSemanticGraph import UnorderedSemanticGraph
from graph.agent_network import AgentNetwork

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

if __name__ == '__main__':
    num_eval = 1
    path = './hipstr/'
    model_path = path + 'gangstr.pt'

    args = get_args()

    args.env_name = 'FetchManipulate{}Objects-v0'.format(args.n_blocks)

    # Make the environment
    env = gym.make(args.env_name)

    # set random seeds for reproduce
    args.seed = np.random.randint(1e6)
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    args.env_params = get_env_params(env)

    # create the sac agent to interact with the environment
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward)
        policy.load(model_path)
    else:
        raise NotImplementedError

    goal_sampler = GoalSampler(args, policy)

    # def rollout worker
    rollout_worker = HMERolloutWorker(env, policy, goal_sampler,  args)

    # load agent graph
    nk_graph = nk.Graph(0, weighted=True, directed=True)
    semantic_graph = UnorderedSemanticGraph(bidict(), nk_graph, args.n_blocks, True, args=args)
    agent_network = AgentNetwork(semantic_graph, None, args)
    agent_network = agent_network.load(path, 260, args)

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

    all_results = []
    for i in range(num_eval):
        episodes = rollout_worker.test_rollout(eval_goals, agent_network,
                                               episode_duration=args.episode_duration,
                                               animated=False)
        results = np.array([e['success'][-1].astype(np.float32) for e in episodes])
        all_results.append(results)

    results = np.array(all_results)
    print('Av Success Rate: {}'.format(results.mean()))