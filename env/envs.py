from env import fetch_manipulate_env
from gym import utils

PREDICATE_THRESHOLD = 0.09  # The minimal threshold to consider two blocks close to each other
PROBA_COPLANAR = 0.8  # The probability of initializing all blocks without stacks
PROBA_STACK_TWO = 0.6  # The probability of having exactly one stack of two given that there is at least one stack
PROBA_GRASP = 0.0  # The probability of having a block grasped at initialization
DISTANCE_THRESHOLD = 0.05

class FetchManipulateEnv(fetch_manipulate_env.FetchManipulateEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', num_blocks=3, model_path='fetch/stack3.xml'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_manipulate_env.FetchManipulateEnv.__init__(
            self, model_path, num_blocks=num_blocks, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, predicate_threshold=PREDICATE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, predicates=['close', 'above'],
            p_coplanar=PROBA_COPLANAR, p_stack_two=PROBA_STACK_TWO, p_grasp=PROBA_GRASP,
        )
        utils.EzPickle.__init__(self)
