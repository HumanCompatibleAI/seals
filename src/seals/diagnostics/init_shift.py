"""Environment with shift in initial state distribution."""

import functools
import itertools

import numpy as np

from seals import base_envs


class InitShiftEnv(base_envs.TabularModelMDP):
    """Tests for robustness to initial state shift.

    Many LfH algorithms learn from expert demonstrations. This can be
    problematic when the environment the demonstrations were gathered in
    differs even slightly from the learner's environment.

    This task illustrates this problem.  We have a depth-2 full binary tree
    where the agent moves left or right until reaching a leaf. The expert
    starts at the root s_0, whereas the learner starts at the left branch s_1
    and so can only reach leaves s_3 and s_4. Reward is only given at the
    leaves.

    The expert always move to the highest reward leaf s_6, so any algorithm
    that relies on demonstrations will not know whether it is better to go to
    s_3 or s_4. By contrast, feedback such as preference comparison can
    disambiguate this case.
    """

    def __init__(self, initial_state: base_envs.DiscreteSpaceInt):
        """Constructs environment.

        Args:
            initial_state: fixed initial state.

        Raises:
            ValueError: `initial_state` not in [0,6].
        """
        nS = 7
        nA = 2

        if not 0 <= initial_state < nS:
            raise ValueError(f"Initial state {initial_state} must lie in [0,{nS})")

        self._initial_state = initial_state

        non_leaves = np.arange(3)
        leaves = np.arange(3, 7)

        transition_matrix = np.zeros((nS, nA, nS))

        for state, action in itertools.product(non_leaves, range(nA)):
            next_state = 2 * state + 1 + action
            transition_matrix[state, action, next_state] = 1.0

        transition_matrix[leaves, :, leaves] = 1.0

        reward_matrix = np.zeros((nS,))
        reward_matrix[leaves] = [1, -1, -1, 2]

        super().__init__(
            transition_matrix=transition_matrix,
            reward_matrix=reward_matrix,
        )

    def initial_state(self) -> base_envs.DiscreteSpaceInt:
        """Returns initial state defined in constructor."""
        return self._initial_state


InitShiftTrainEnv = functools.partial(InitShiftEnv, initial_state=0)
InitShiftTestEnv = functools.partial(InitShiftEnv, initial_state=1)
