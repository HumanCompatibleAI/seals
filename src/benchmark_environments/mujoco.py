"""Adaptation of MuJoCo environments for IRL.
"""

import gym
from gym.envs.mujoco import half_cheetah_v3

class HalfCheetahEnv(half_cheetah_v3.HalfCheetahEnv):
    """HalfCheetah with center of mass in observation.
    """
    def __init__(self, *args, **kwargs):
        """
        Args:
            xml_file (str): Asset file describing the half cheetah model.
                Default: 'half_cheetah.xml'
            forward_reward_weight (float): weight in reward for moving forward.
                Default: 1.0
            ctrl_cost_weight (float): weight in reward for action cost. Default: 0.1
            reset_noise_scale (float): Standard deviation of sample for initial
                position/velocity. Default: 0.1
        """
        kwargs['exclude_current_positions_from_observation'] = False
        super().__init__(*args, **kwargs)
