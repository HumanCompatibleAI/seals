"""A tabular model MDP with a random transition matrix."""

from typing import Optional

import numpy as np

from seals.base_envs import TabularModelPOMDP


class RandomTransitionEnv(TabularModelPOMDP):
    """AN MDP with a random transition matrix.

    Random matrix is created by `make_random_trans_mat`.
    """

    reward_weights: np.ndarray

    def __init__(
        self,
        *,
        n_states: int,
        n_actions: int,
        branch_factor: int,
        horizon: int,
        random_obs: bool,
        obs_dim: Optional[int] = None,
        generator_seed: Optional[int] = None,
    ):
        """Builds RandomTransitionEnv.

        Args:
            n_states: Number of states.
            n_actions: Number of actions.
            branch_factor: Maximum number of states that can be reached from
                each state-action pair.
            horizon: The horizon of the MDP, i.e. the episode length.
            random_obs: Whether to use random observations (True)
                or one-hot coded (False).
            obs_dim: The size of the observation vectors; must be `None`
                if `random_obs == False`.
            generator_seed: Seed for NumPy RNG.

        Raises:
            ValueError: If ``obs_dim`` is not ``None`` when ``random_obs == False``.
            ValueError: If ``obs_dim`` is ``None`` when ``random_obs == True``.
        """
        # this generator is ONLY for constructing the MDP, not for controlling
        # random outcomes during rollouts
        rand_gen = np.random.default_rng(generator_seed)

        if random_obs:
            if obs_dim is None:
                obs_dim = n_states
        else:
            if obs_dim is not None:
                raise ValueError("obs_dim must be None if random_obs is False")

        observation_matrix = self.make_obs_mat(
            n_states=n_states,
            is_random=random_obs,
            obs_dim=obs_dim,
            rand_state=rand_gen,
        )
        transition_matrix = self.make_random_trans_mat(
            n_states=n_states,
            n_actions=n_actions,
            max_branch_factor=branch_factor,
            rand_state=rand_gen,
        )
        initial_state_dist = self.make_random_state_dist(
            n_avail=branch_factor,
            n_states=n_states,
            rand_state=rand_gen,
        )

        self.reward_weights = rand_gen.normal(size=(observation_matrix.shape[-1],))
        reward_matrix = observation_matrix @ self.reward_weights
        super().__init__(
            transition_matrix=transition_matrix,
            observation_matrix=observation_matrix,
            reward_matrix=reward_matrix,
            horizon=horizon,
            initial_state_dist=initial_state_dist,
        )

    @staticmethod
    def make_random_trans_mat(
        n_states,
        n_actions,
        max_branch_factor,
        rand_state: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Make a 'random' transition matrix.

        Each action goes to at least `max_branch_factor` other states from the
        current state, with transition distribution sampled from Dirichlet(1,1,…,1).

        This roughly apes the strategy from some old Lisp code that Rich Sutton
        left on the internet (http://incompleteideas.net/RandomMDPs.html), and is
        therefore a legitimate way to generate MDPs.

        Args:
            n_states: Number of states.
            n_actions: Number of actions.
            max_branch_factor: Maximum number of states that can be reached from
                each state-action pair.
            rand_state: NumPy random state.

        Returns:
            The transition matrix `mat`, where `mat[s,a,next_s]` gives the probability
            of transitioning to `next_s` after taking action `a` in state `s`.
        """
        if rand_state is None:
            rand_state = np.random.default_rng()
        assert rand_state is not None
        out_mat = np.zeros((n_states, n_actions, n_states), dtype="float32")
        for start_state in range(n_states):
            for action in range(n_actions):
                # uniformly sample a number of successors in [1,max_branch_factor]
                # for this action
                successors = rand_state.integers(1, max_branch_factor + 1)
                next_states = rand_state.choice(
                    n_states,
                    size=(successors,),
                    replace=False,
                )
                # generate random vec in probability simplex
                next_vec = rand_state.dirichlet(np.ones((successors,)))
                next_vec = next_vec / np.sum(next_vec)
                out_mat[start_state, action, next_states] = next_vec
        return out_mat

    @staticmethod
    def make_random_state_dist(
        n_avail: int,
        n_states: int,
        rand_state: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Make a random initial state distribution over n_states.

        Args:
            n_avail: Number of states available to transition into.
            n_states: Total number of states.
            rand_state: NumPy random state.

        Returns:
            An initial state distribution that is zero at all but a uniformly random
            chosen subset of `n_avail` states. This subset of chosen states are set to a
            sample from the uniform distribution over the (n_avail-1) simplex, aka the
            flat Dirichlet distribution.

        Raises:
            ValueError: If `n_avail` is not in the range `(0, n_states]`.
        """  # noqa: DAR402
        if rand_state is None:
            rand_state = np.random.default_rng()
        assert rand_state is not None
        assert 0 < n_avail <= n_states
        init_dist = np.zeros((n_states,))
        next_states = rand_state.choice(n_states, size=(n_avail,), replace=False)
        avail_state_dist = rand_state.dirichlet(np.ones((n_avail,)))
        init_dist[next_states] = avail_state_dist
        assert np.sum(init_dist > 0) == n_avail
        init_dist = init_dist / np.sum(init_dist)
        return init_dist

    @staticmethod
    def make_obs_mat(
        n_states: int,
        is_random: bool,
        obs_dim: Optional[int] = None,
        rand_state: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Makes an observation matrix with a single observation for each state.

        Args:
            n_states (int): Number of states.
            is_random (bool): Are observations drawn at random?
                        If `True`, draw from random normal distribution.
                        If `False`, are unique one-hot vectors for each state.
            obs_dim (int or NoneType): Must be `None` if `is_random == False`.
                    Otherwise, this must be set to the size of the random vectors.
            rand_state (np.random.Generator): Random number generator.

        Returns:
            A matrix of shape `(n_states, obs_dim if is_random else n_states)`.

        Raises:
            ValueError: If ``is_random == False`` and ``obs_dim is not None``.
        """
        if rand_state is None:
            rand_state = np.random.default_rng()
        assert rand_state is not None
        if is_random:
            if obs_dim is None:
                raise ValueError("obs_dim must be set if random_obs is True")
            obs_mat = rand_state.normal(0, 2, (n_states, obs_dim))
        else:
            if obs_dim is not None:
                raise ValueError("obs_dim must be None if random_obs is False")
            obs_mat = np.identity(n_states)
        assert (
            obs_mat.ndim == 2
            and obs_mat.shape[:1] == (n_states,)
            and obs_mat.shape[1] > 0
        )
        return obs_mat.astype(np.float32)
