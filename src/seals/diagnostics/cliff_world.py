"""A cliff world that uses the TabularModelPOMDP."""

import numpy as np

from seals.base_envs import TabularModelPOMDP


class CliffWorldEnv(TabularModelPOMDP):
    """A grid world with a goal next to a cliff the agent may fall into.

    Illustration::

         0 1 2 3 4 5 6 7 8 9
        +-+-+-+-+-+-+-+-+-+-+  Wind:
      0 |S|C|C|C|C|C|C|C|C|G|
        +-+-+-+-+-+-+-+-+-+-+  ^ ^ ^
      1 | | | | | | | | | | |  | | |
        +-+-+-+-+-+-+-+-+-+-+
      2 | | | | | | | | | | |  ^ ^ ^
        +-+-+-+-+-+-+-+-+-+-+  | | |

    Aim is to get from S to G. The G square has reward +10, the C squares
    ("cliff") have reward -10, and all other squares have reward -1. Agent can
    move in all directions (except through walls), but there is 30% chance that
    they will be blown upwards by one more unit than intended due to wind.
    Optimal policy is to go out a bit and avoid the cliff, but still hit goal
    eventually.
    """

    width: int
    height: int

    def __init__(
        self,
        *,
        width: int,
        height: int,
        horizon: int,
        use_xy_obs: bool,
        rew_default: int = -1,
        rew_goal: int = 10,
        rew_cliff: int = -10,
        fail_p: float = 0.3,
    ):
        """Builds CliffWorld with specified dimensions and reward."""
        assert (
            width >= 3 and height >= 2
        ), "degenerate grid world requested; is this a bug?"
        self.width = width
        self.height = height
        succ_p = 1 - fail_p
        n_states = width * height
        O_mat = np.zeros(
            (n_states, 2 if use_xy_obs else n_states),
            dtype=np.float32,
        )
        R_vec = np.zeros((n_states,))
        T_mat = np.zeros((n_states, 4, n_states))

        def to_id_clamp(row, col):
            """Convert (x,y) state to state ID, after clamp x & y to lie in grid."""
            row = min(max(row, 0), height - 1)
            col = min(max(col, 0), width - 1)
            state_id = row * width + col
            assert 0 <= state_id < T_mat.shape[0]
            return state_id

        for row in range(height):
            for col in range(width):
                state_id = to_id_clamp(row, col)

                # start by computing reward
                if row > 0:
                    r = rew_default  # blank
                elif col == 0:
                    r = rew_default  # start
                elif col == width - 1:
                    r = rew_goal  # goal
                else:
                    r = rew_cliff  # cliff
                R_vec[state_id] = r

                # now compute observation
                if use_xy_obs:
                    # (x, y) coordinate scaled to (0,1)
                    O_mat[state_id, :] = [
                        float(col) / (width - 1),
                        float(row) / (height - 1),
                    ]
                else:
                    # our observation matrix is just the identity; observation
                    # is an indicator vector telling us exactly what state
                    # we're in
                    O_mat[state_id, state_id] = 1

                # finally, compute transition matrix entries for each of the
                # four actions
                for drow in [-1, 1]:
                    for dcol in [-1, 1]:
                        action_id = (drow + 1) + (dcol + 1) // 2
                        target_state = to_id_clamp(row + drow, col + dcol)
                        fail_state = to_id_clamp(row + drow - 1, col + dcol)
                        T_mat[state_id, action_id, fail_state] += fail_p
                        T_mat[state_id, action_id, target_state] += succ_p

        assert np.allclose(np.sum(T_mat, axis=-1), 1, rtol=1e-5), (
            "un-normalised matrix %s" % O_mat
        )
        super().__init__(
            transition_matrix=T_mat,
            observation_matrix=O_mat,
            reward_matrix=R_vec,
            horizon=horizon,
            initial_state_dist=None,
        )

    def draw_value_vec(self, D: np.ndarray) -> None:
        """Use matplotlib to plot a vector of values for each state.

        The vector could represent things like reward, occupancy measure, etc.

        Args:
            D: the vector to plot.

        Raises:
            ImportError: if matplotlib is not installed.
        """
        try:  # pragma: no cover
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "matplotlib is not installed in your system, "
                "and is required for this function.",
            ) from exc

        grid = D.reshape(self.height, self.width)
        plt.imshow(grid)
        plt.gca().grid(False)
