"""Hard-exploration environment."""

import itertools

import numpy as np

from seals import base_envs


class BranchingEnv(base_envs.TabularModelMDP):
    """Long branching environment requiring exploration.

    The agent must traverse a specific path of length L to reach a
    final goal, with B choices at each step. Wrong actions lead to
    dead-ends with zero reward.
    """

    def __init__(self, branch_factor: int = 2, length: int = 10):
        """Construct environment.

        Args:
            branch_factor: number of actions at each state.
            length: path length from initial state to goal.
        """
        nS = 1 + branch_factor * length
        nA = branch_factor

        def get_next(state: int, action: int) -> int:
            can_move = state % branch_factor == 0 and state != nS - 1
            return state + (action + 1) * can_move

        transition_matrix = np.zeros((nS, nA, nS))
        for state, action in itertools.product(range(nS), range(nA)):
            transition_matrix[state, action, get_next(state, action)] = 1.0

        reward_matrix = np.zeros((nS,))
        reward_matrix[-1] = 1.0

        super().__init__(
            transition_matrix=transition_matrix,
            reward_matrix=reward_matrix,
        )
