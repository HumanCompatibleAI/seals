"""Environment testing for correct behavior under stochasticity."""

import numpy as np

from seals import base_envs


class RiskyPathEnv(base_envs.TabularModelMDP):
    """Environment with two paths to a goal: one safe and one risky.

    Many LfH algorithms are derived from Maximum Entropy Inverse Reinforcement
    Learning [1], which models the demonstrator as producing trajectories with
    probability p(tau) proportional to exp(R(tau)).  This model implies that a
    demonstrator can "control" the environment well enough to follow any
    high-reward trajectory with high probability [2]. However, in stochastic
    environments, the agent cannot control the probability of each trajectory
    independently.  This misspecification may lead to poor behavior.

    This task tests for this behavior. The agent starts at s_0 and can reach
    the goal s_2 (reward 1.0) by either taking the safe path s_0 to s_1 to s_2,
    or taking a risky action, which has equal chances of going to either s_3
    (reward -100.0) or s_2.  The safe path has the highest expected return, but
    the risky action sometimes reaches the goal s_2 in fewer timesteps, leading
    to higher best-case return. Algorithms that fail to correctly handle
    stochastic dynamics may therefore wrongly believe the reward favors taking
    the risky path.

    [1] Ziebart, Brian D., et al. "Maximum entropy inverse reinforcement
        learning." AAAI. Vol. 8. 2008.
    [2] Ziebart, Brian D. "Modeling purposeful adaptive behavior with the
        principle of maximum causal entropy." (2010); PhD thesis,
        CMU-ML-10-110; page 105.
    """

    def __init__(self):
        """Initialize environment."""
        nS = 4
        nA = 2

        transition_matrix = np.zeros((nS, nA, nS))
        transition_matrix[0, 0, 1] = 1.0
        transition_matrix[0, 1, [2, 3]] = 0.5

        transition_matrix[1, 0, 2] = 1.0
        transition_matrix[1, 1, 1] = 1.0

        transition_matrix[[2, 3], :, [2, 3]] = 1.0

        reward_matrix = np.array([0.0, 0.0, 1.0, -100.0])

        super().__init__(
            transition_matrix=transition_matrix,
            reward_matrix=reward_matrix,
        )
