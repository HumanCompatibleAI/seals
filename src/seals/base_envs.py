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

    _state_space: gym.Space
    _observation_space: gym.Space
    _action_space: gym.Space
    _cur_state: Optional[State]
    _n_actions_taken: Optional[int]

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

        self._cur_state = None
        self._n_actions_taken = None
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

    @property
    def state(self) -> State:
        """Current state."""
        assert self._cur_state is not None
        return self._cur_state

    @state.setter
    def state(self, state: State):
        """Set current state."""
        if state not in self.state_space:
            raise ValueError(f"{state} not in {self.state_space}")
        self._cur_state = state

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
        self.state = self.initial_state()
        self._n_actions_taken = 0
        return self.obs_from_state(self.state)

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Transition state using given action."""
        if self._cur_state is None or self._n_actions_taken is None:
            raise ValueError("Need to call reset() before first step()")
        if action not in self.action_space:
            raise ValueError(f"{action} not in {self.action_space}")

        old_state = self.state
        self.state = self.transition(self.state, action)
        obs = self.obs_from_state(self.state)
        assert obs in self.observation_space
        reward = self.reward(old_state, action, self.state)
        self._n_actions_taken += 1
        done = self.terminal(self.state, self.n_actions_taken)

        infos = {"old_state": old_state, "new_state": self._cur_state}
        return obs, reward, done, infos


class ExposePOMDPStateWrapper(gym.Wrapper, Generic[State, Observation, Action]):
    """A wrapper that exposes the current state of the POMDP as the observation."""

    def __init__(self, env: ResettablePOMDP[State, Observation, Action]) -> None:
        """Build wrapper.

        Args:
            env: POMDP to wrap.
        """
        super().__init__(env)
        self._observation_space = env.state_space

    def reset(self) -> State:
        """Reset environment and return initial state."""
        self.env.reset()
        return self.env.state

    def step(self, action) -> Tuple[State, float, bool, dict]:
        """Transition state using given action."""
        obs, reward, done, info = self.env.step(action)
        return self.env.state, reward, done, info


class ResettableMDP(
    ResettablePOMDP[State, State, Action],
    abc.ABC,
    Generic[State, Action],
):
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


# TODO(juan) this does not implement the .render() method,
#  so in theory it should not be instantiated directly.
#  Not sure why this is not raising an error?
class BaseTabularModelPOMDP(ResettablePOMDP[int, Observation, int]):
    """Base class for tabular environments with known dynamics.

    This is the general class that also allows subclassing for creating
    MDP (where observation == state) or POMDP (where observation != state).
    """

    transition_matrix: np.ndarray
    reward_matrix: np.ndarray

    state_space: spaces.Discrete

    def __init__(
        self,
        *,
        transition_matrix: np.ndarray,
        reward_matrix: np.ndarray,
        horizon: Optional[int] = None,
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
            horizon: Maximum number of timesteps. The default is `None`,
                which represents an infinite horizon.
            initial_state_dist: Distribution from which state is sampled at the
                start of the episode.  If `None`, it is assumed initial state
                is always 0. Shape `(n_states,)`.

        Raises:
            ValueError: `transition_matrix`, `reward_matrix` or
                `initial_state_dist` have shapes different to specified above.
        """
        # The following matrices should conform to the shapes below:

        # transition matrix: n_states x n_actions x n_states
        n_states = transition_matrix.shape[0]
        if n_states != transition_matrix.shape[2]:
            raise ValueError(
                "Malformed transition_matrix:\n"
                f"transition_matrix.shape: {transition_matrix.shape}\n"
                f"{n_states} != {transition_matrix.shape[2]}",
            )

        # reward matrix: n_states x n_actions x n_states
        #   OR n_states x n_actions
        #   OR n_states
        if reward_matrix.shape != transition_matrix.shape[: len(reward_matrix.shape)]:
            raise ValueError(
                "transition_matrix and reward_matrix are not compatible:\n"
                f"transition_matrix.shape: {transition_matrix.shape}\n"
                f"reward_matrix.shape: {reward_matrix.shape}",
            )

        # initial state dist: n_states
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
                f"number of states = {n_states}\n"
                f"len(initial_state_dist) = {len(initial_state_dist)}",
            )

        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix
        self._feature_matrix = None
        self.horizon = horizon
        self.initial_state_dist = initial_state_dist

        super().__init__(
            state_space=self._construct_state_space(),
            action_space=self._construct_action_space(),
            observation_space=self._construct_observation_space(),
        )

    def _construct_state_space(self) -> gym.Space:
        return spaces.Discrete(self.state_dim)

    def _construct_action_space(self) -> gym.Space:
        return spaces.Discrete(self.action_dim)

    @abc.abstractmethod
    def _construct_observation_space(self) -> gym.Space:
        pass  # pragma: no cover

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
        return self.horizon is not None and n_actions_taken >= self.horizon

    @property
    def feature_matrix(self):
        """Matrix mapping states to feature vectors."""
        # Construct lazily to save memory in algorithms that don't need features.
        if self._feature_matrix is None:
            n_states = self.state_space.n
            self._feature_matrix = np.eye(n_states)
        return self._feature_matrix

    @property
    def state_dim(self):
        """Number of states in this MDP (int)."""
        return self.transition_matrix.shape[0]

    @property
    def action_dim(self) -> int:
        """Number of action vectors (int)."""
        return self.transition_matrix.shape[1]


class TabularModelPOMDP(BaseTabularModelPOMDP[np.ndarray]):
    """Tabular model POMDP.

    This class is specifically for environments where observation != state,
    from both a typing perspective but also by defining the method that
    draws observations from the state.

    The tabular model is deterministic in drawing observations from the state,
    in that given a certain state, the observation is always the same;
    a vector with self.obs_dim entries.
    """

    observation_matrix: np.ndarray

    def __init__(
        self,
        *,
        transition_matrix: np.ndarray,
        observation_matrix: np.ndarray,
        reward_matrix: np.ndarray,
        horizon: Optional[int] = None,
        initial_state_dist: Optional[np.ndarray] = None,
    ):
        """Initializes a tabular model POMDP."""
        self.observation_matrix = observation_matrix
        super().__init__(
            transition_matrix=transition_matrix,
            reward_matrix=reward_matrix,
            horizon=horizon,
            initial_state_dist=initial_state_dist,
        )

        # observation matrix: n_states x n_observations
        if observation_matrix.shape[0] != self.state_dim:
            raise ValueError(
                "transition_matrix and observation_matrix are not compatible:\n"
                f"transition_matrix.shape[0]: {self.state_dim}\n"
                f"observation_matrix.shape[0]: {observation_matrix.shape[0]}",
            )

    def _construct_observation_space(self) -> gym.Space:
        min_val: float
        max_val: float
        try:
            dtype_iinfo = np.iinfo(self.obs_dtype)
            min_val, max_val = dtype_iinfo.min, dtype_iinfo.max
        except ValueError:
            min_val = -np.inf
            max_val = np.inf
        return spaces.Box(
            low=min_val,
            high=max_val,
            shape=(self.obs_dim,),
            dtype=self.obs_dtype,
        )

    def obs_from_state(self, state: int) -> np.ndarray:
        """Computes observation from state."""
        # Copy so it can't be mutated in-place (updates will be reflected in
        # self.observation_matrix!)
        obs = self.observation_matrix[state].copy()
        assert obs.ndim == 1, obs.shape
        return obs

    @property
    def obs_dim(self) -> int:
        """Size of observation vectors for this MDP."""
        return self.observation_matrix.shape[1]

    @property
    def obs_dtype(self) -> int:
        """Data type of observation vectors (e.g. np.float32)."""
        return self.observation_matrix.dtype


class TabularModelMDP(BaseTabularModelPOMDP[int]):
    """Tabular model MDP.

    A tabular model MDP is a tabular MDP where the transition and reward
    matrices are constant.
    """

    def __init__(
        self,
        *,
        transition_matrix: np.ndarray,
        reward_matrix: np.ndarray,
        horizon: Optional[int] = None,
        initial_state_dist: Optional[np.ndarray] = None,
    ):
        """Initializes a tabular model MDP.

        Args:
            transition_matrix: Matrix of shape `(n_states, n_actions, n_states)`
                containing transition probabilities.
            reward_matrix: Matrix of shape `(n_states, n_actions, n_states)`
                containing reward values.
            initial_state_dist: Distribution over initial states. Shape `(n_states,)`.
            horizon: Maximum number of steps to take in an episode.
        """
        super().__init__(
            transition_matrix=transition_matrix,
            reward_matrix=reward_matrix,
            horizon=horizon,
            initial_state_dist=initial_state_dist,
        )

    def obs_from_state(self, state: int) -> int:
        """Identity since observation == state in an MDP."""
        return state

    def _construct_observation_space(self) -> gym.Space:
        return self._construct_state_space()
