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
from utils import get_eval_goals, get_idxs_per_relation

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

def extract_random_subgoal(g):
    relations_ids = get_idxs_per_relation(5)
    attended_relation = random.choices(relations_ids)
    res = np.zeros(30)
    res[attended_relation] = g[attended_relation]
    return res

def extract_all_subgoals(g):
    res = []
    relations_ids = get_idxs_per_relation(5)
    for r in relations_ids:
        temp = np.zeros(30)
        temp[r] = g[r]
        res.append(temp)
    return res
if __name__ == '__main__':
    num_eval = 10
    path = '/home/ahmed/Documents/ICLR2022/gangstr/hipstr/'
    model_path = path + 'policy.pt'

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

    goal_sampler = GoalSampler(args)

    # create the sac agent to interact with the environment
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward, goal_sampler)
        policy.load(model_path)
    else:
        raise NotImplementedError

    # def rollout worker
    rollout_worker = HMERolloutWorker(env, policy, goal_sampler,  args)

    eval_goals = []
    instructions = ['stack_3']
    for instruction in instructions:
        eval_goal = get_eval_goals(instruction, n=args.n_blocks)
        eval_goals.append(eval_goal.squeeze(0))
    eval_goals = np.array(eval_goals)

    all_results = []
    # for i in range(num_eval):
    goals = extract_all_subgoals(eval_goals[0])
    for goal in goals:
        episodes = rollout_worker.generate_one_rollout(goal, True, 100, animated=True)
