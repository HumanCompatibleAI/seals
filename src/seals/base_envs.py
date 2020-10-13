"""Base environment classes."""

import abc
from typing import Generic, Optional, Sequence, Tuple, TypeVar

import gym
from gym import spaces
import numpy as np

from seals import util

State = TypeVar("State")
Observation = TypeVar("Observation")
Action = TypeVar("Action")


class ResettablePOMDP(gym.Env, abc.ABC, Generic[State, Observation, Action]):
    """ABC for POMDPs that are resettable.

    Specifically, these environments provide oracle access to sample from
    the initial state distribution and transition dynamics, and compute the
    reward and termination condition. Almost all simulated environments can
    meet these criteria.
    """

    def __init__(
        self,
        *,
        state_space: gym.Space,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        """Build resettable (PO)MDP.

        Args:
            state_space: gym.Space containing possible states.
            observation_space: gym.Space containing possible observations.
            action_space: gym.Space containing possible actions.
        """
        self._state_space = state_space
        self._observation_space = observation_space
        self._action_space = action_space

        self.cur_state: Optional[State] = None
        self._n_actions_taken: Optional[int] = None
        self.seed()

    @abc.abstractmethod
    def initial_state(self) -> State:
        """Samples from the initial state distribution."""

    @abc.abstractmethod
    def transition(self, state: State, action: Action) -> State:
        """Samples from transition distribution."""

    @abc.abstractmethod
    def reward(self, state: State, action: Action, new_state: State) -> float:
        """Computes reward for a given transition."""

    @abc.abstractmethod
    def terminal(self, state: State, step: int) -> bool:
        """Is the state terminal?"""

    @abc.abstractmethod
    def obs_from_state(self, state: State) -> Observation:
        """Sample observation for given state."""

    @property
    def state_space(self) -> gym.Space:
        """State space. Often same as observation_space, but differs in POMDPs."""
        return self._state_space

    @property
    def observation_space(self) -> gym.Space:
        """Observation space. Return type of reset() and component of step()."""
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        """Action space. Parameter type of step()."""
        return self._action_space

    @property
    def n_actions_taken(self) -> int:
        """Number of steps taken so far."""
        assert self._n_actions_taken is not None
        return self._n_actions_taken

    def seed(self, seed=None) -> Sequence[int]:
        """Set random seed."""
        if seed is None:
            # Gym API wants list of seeds to be returned for some reason, so
            # generate a seed explicitly in this case
            seed = np.random.randint(0, 1 << 31)
        self.rand_state = np.random.RandomState(seed)
        return [seed]

    def reset(self) -> Observation:
        """Reset episode and return initial observation."""
        self.cur_state = self.initial_state()
        assert self.cur_state in self.state_space, f"unexpected state {self.cur_state}"
        self._n_actions_taken = 0
        return self.obs_from_state(self.cur_state)

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Transition state using given action."""
        if self.cur_state is None or self._n_actions_taken is None:
            raise ValueError("Need to call reset() before first step()")
        if action not in self.action_space:
            raise ValueError(f"{action} not in {self.action_space}")

        old_state = self.cur_state
        self.cur_state = self.transition(self.cur_state, action)
        assert self.cur_state in self.state_space, f"unexpected state {self.cur_state}"
        obs = self.obs_from_state(self.cur_state)
        assert obs in self.observation_space, f"{obs} not in {self.observation_space}"
        rew = self.reward(old_state, action, self.cur_state)
        done = self.terminal(self.cur_state, self._n_actions_taken)
        self._n_actions_taken += 1

        infos = {"old_state": old_state, "new_state": self.cur_state}
        return obs, rew, done, infos


class ResettableMDP(ResettablePOMDP[State, State, Action], Generic[State, Action]):
    """ABC for MDPs that are resettable."""

    def __init__(
        self,
        *,
        state_space: gym.Space,
        action_space: gym.Space,
    ):
        """Build resettable MDP.

        Args:
            state_space: gym.Space containing possible states.
            action_space: gym.Space containing possible actions.
        """
        super().__init__(
            state_space=state_space,
            observation_space=state_space,
            action_space=action_space,
        )

    def obs_from_state(self, state: State) -> State:
        """Identity since observation == state in an MDP."""
        return state


class TabularModelMDP(ResettableMDP[int, int]):
    """Base class for tabular environments with known dynamics."""

    def __init__(
        self,
        *,
        transition_matrix: np.ndarray,
        reward_matrix: np.ndarray,
        horizon: float = np.inf,
        initial_state_dist: Optional[np.ndarray] = None,
    ):
        """Build tabular environment.

        Args:
            transition_matrix: 3-D array with transition probabilities for a
                given state-action pair, of shape `(n_states,n_actions,n_states)`.
            reward_matrix: 1-D, 2-D or 3-D array corresponding to rewards to a
                given `(state, action, next_state)` triple. A 2-D array assumes
                the `next_state` is not used in the reward, and a 1-D array
                assumes neither the `action` nor `next_state` are used.
                Of shape `(n_states,n_actions,n_states)[:n]` where `n`
                is the dimensionality of the array.
            horizon: Maximum number of timesteps, default `np.inf`.
            initial_state_dist: Distribution from which state is sampled at the
                start of the episode.  If `None`, it is assumed initial state
                is always 0. Shape `(n_states,)`.

        Raises:
            ValueError: `transition_matrix`, `reward_matrix` or
                `initial_state_dist` have shapes different to specified above.
        """
        n_states, n_actions, n_next_states = transition_matrix.shape
        if n_states != n_next_states:
            raise ValueError(
                "Malformed transition_matrix:\n"
                f"transition_matrix.shape: {transition_matrix.shape}\n"
                f"{n_states} != {n_next_states}",
            )

        if initial_state_dist is None:
            initial_state_dist = util.one_hot_encoding(0, n_states)
        if initial_state_dist.ndim != 1:
            raise ValueError(
                "initial_state_dist has multiple dimensions:\n"
                f"{initial_state_dist.ndim} != 1",
            )
        if initial_state_dist.shape[0] != n_states:
            raise ValueError(
                "transition_matrix and initial_state_dist are not compatible:\n"
                f"n_states = {n_states}\n"
                f"len(initial_state_dist) = {len(initial_state_dist)}",
            )

        if reward_matrix.shape != transition_matrix.shape[: len(reward_matrix.shape)]:
            raise ValueError(
                "transition_matrix and reward_matrix are not compatible:\n"
                f"transition_matrix.shape: {transition_matrix.shape}\n"
                f"reward_matrix.shape: {reward_matrix.shape}",
            )

        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix
        self._feature_matrix = None
        self.horizon = horizon
        self.initial_state_dist = initial_state_dist

        super().__init__(
            state_space=spaces.Discrete(n_states),
            action_space=spaces.Discrete(n_actions),
        )

    def initial_state(self) -> int:
        """Samples from the initial state distribution."""
        return util.sample_distribution(
            self.initial_state_dist,
            random=self.rand_state,
        )

    def transition(self, state: int, action: int) -> int:
        """Samples from transition distribution."""
        return util.sample_distribution(
            self.transition_matrix[state, action],
            random=self.rand_state,
        )

    def reward(self, state: int, action: int, new_state: int) -> float:
        """Computes reward for a given transition."""
        inputs = (state, action, new_state)[: len(self.reward_matrix.shape)]
        return self.reward_matrix[inputs]

    def terminal(self, state: int, n_actions_taken: int) -> bool:
        """Checks if state is terminal."""
        return n_actions_taken >= self.horizon

    @property
    def feature_matrix(self):
        """Matrix mapping states to feature vectors."""
        # Construct lazily to save memory in algorithms that don't need features.
        if self._feature_matrix is None:
            n_states = self.state_space.n
            self._feature_matrix = np.eye(n_states)
        return self._feature_matrix
