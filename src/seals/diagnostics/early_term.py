"""Environment checking for correctness under early termination."""

import functools

import numpy as np

from seals import base_envs


class EarlyTerminationEnv(base_envs.TabularModelMDP):
    """Three-state MDP with early termination state.

    Many implementations of imitation learning algorithms incorrectly assign a
    value of zero to terminal states [1]. Depending on the sign of the learned
    reward function in non-terminal states, this can either bias the agent to
    end episodes early or prolong them as long as possible. This confounds
    evaluation as performance is spuriously high in tasks where the termination
    bias aligns with the task objective. These tasks attempt to detect this
    type of bias, and they are adapted from [1].

    The environment is a 3-state MDP, in which the agent can either alternate
    between two initial states until reaching the time horizon, or they can
    move to a terminal state causing the episode to terminate early.

    [1] Kostrikov, Ilya, et al. "Discriminator-actor-critic: Addressing sample
    inefficiency and reward bias in adversarial imitation learning." arXiv
    preprint arXiv:1809.02925 (2018).
    """

    def __init__(self, is_reward_positive: bool = True):
        """Construct environment.

        Args:
            is_reward_positive: whether rewards are positive or negative.
        """
        nS = 3
        nA = 2

        transition_matrix = np.zeros((nS, nA, nS))

        transition_matrix[0, :, 1] = 1.0

        transition_matrix[1, 0, 0] = 1.0
        transition_matrix[1, 1, 2] = 1.0

        transition_matrix[2, :, 2] = 1.0

        reward_sign = 2 * is_reward_positive - 1
        reward_matrix = reward_sign * np.ones((nS,), dtype=float)

        super().__init__(
            transition_matrix=transition_matrix,
            reward_matrix=reward_matrix,
        )

    def terminal(self, state: base_envs.DiscreteSpaceInt, n_actions_taken: int) -> bool:
        """Returns True if (and only if) in state 2."""
        return bool(state == 2)


EarlyTermPosEnv = functools.partial(EarlyTerminationEnv, is_reward_positive=True)
EarlyTermNegEnv = functools.partial(EarlyTerminationEnv, is_reward_positive=False)
