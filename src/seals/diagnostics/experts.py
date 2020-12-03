"""Expert policies for diagnostic environments."""

import numpy as np
from scipy.special import logsumexp  # type: ignore

from seals.util import sample_distribution


def _get_horizon(env):
    return env._max_episode_steps


def get_noisy_obs_expert_fn(env=None, goal=None):
    """Expert moves towards the center of the grid."""
    if goal is None:
        goal = env.goal

    def predict_fn(ob, state=None, deterministic=False):
        pos = ob[:2]
        dx, dy = goal - pos

        conditions = [
            dx > 0,
            dy > 0,
            dx < 0,
            dy < 0,
            True,
        ]
        act = np.argmax(conditions)

        return act, state

    return predict_fn


def get_proc_goal_expert_fn(env=None):
    """Expert moves in the direction of the goal."""

    def predict_fn(ob, state=None, deterministic=False):
        pos, goal = ob[:2], ob[2:]
        dx, dy = goal - pos

        conditions = [
            dx > 0,
            dy > 0,
            dx < 0,
            dy < 0,
            True,
        ]
        act = np.argmax(conditions)

        return act, state

    return predict_fn


def get_largest_sum_expert_fn(env=None):
    """Expert checks which half has highest sum."""

    def predict_fn(ob, state=None, deterministic=False):
        n = len(ob)
        action = int(np.sum(ob[: n // 2]) > np.sum(ob[n // 2 :]))
        return action, state

    return predict_fn


def get_sort_expert_fn(env=None):
    """Expert performs selection sort on the input.

    Note that in contrast to traditional sorting algorithmic analysis,
    the cost here is not the time taken (which is a function of the number
    of comparisons); rather, the cost here is the number of swaps - which
    is at most n-1 for an optimal policy.
    """

    def predict_fn(ob, state=None, deterministic=False):
        """Performs selection sort."""
        if state is None:
            state = 0
        next_to_sort = state

        act = None
        while act is None and next_to_sort < len(ob):
            pos = next_to_sort + ob[next_to_sort:].argmin()
            if pos != next_to_sort:
                act = (pos, next_to_sort)
            next_to_sort += 1

        if act is None:
            act = (0, 0)

        act = np.array(act)
        return act, next_to_sort

    return predict_fn


def get_early_term_pos_expert_fn(env=None, horizon=None):
    """Alternates between states 0 and 1, moving to state 2 at the last step."""
    if horizon is None:
        horizon = _get_horizon(env)

    def predict_fn(ob, state=None, deterministic=False):
        if state is None:
            state = 0
        t = state

        act = int(horizon - t <= 2)

        state = t + 1
        return act, state

    return predict_fn


def get_early_term_neg_expert_fn(env=None):
    """Moves towards the early termination state."""

    def predict_fn(ob, state=None, deterministic=False):
        act = 1
        return act, state

    return predict_fn


def get_parabola_expert_fn(env=None, x_step=None):
    """Checks corresponding y of the next x and moves to it."""
    if x_step is None:
        x_step = env.x_step

    def predict_fn(ob, state=None, deterministic=False):
        x, y, a, b, c = ob
        x += x_step
        target = a * x ** 2 + b * x + c
        act = target - y
        act = np.array([act])
        return act, state

    return predict_fn


def _policy_matrix_to_predict_fn(policy_matrix):
    def predict_fn(ob, state=None, deterministic=False):
        t = state if state is not None else 0
        action_distribution = policy_matrix[t, ob]
        strategy = np.argmax if deterministic else sample_distribution

        act = strategy(action_distribution)
        new_state = t + 1

        return act, new_state

    return predict_fn


def _force_dim(a, new_dim):
    dim = len(a.shape)
    index = (slice(None),) * dim + (None,) * (new_dim - dim)
    return a[index]


def get_value_iteration_expert_fn(env, discount=1.0, beta=np.inf):
    """Performs value iteration.

    The `beta` parameter allows for "soft value iteration", selecting
    actions using a softmax instead of max (Boltzmann rationality).
    """
    h = _get_horizon(env)
    nS = env.observation_space.n
    nA = env.action_space.n

    R = _force_dim(env.reward_matrix, 3)
    T = env.transition_matrix
    Q = np.empty((h, nS, nA))
    V = np.empty((h + 1, nS))

    V[-1] = np.zeros(nS)
    if np.isinf(beta):
        for t in reversed(range(h)):
            Q[t] = np.einsum("san,san->sa", T, R + discount * V[t + 1][None, None])
            V[t] = np.max(Q[t], axis=1)

        policy = np.eye(nA)[Q.argmax(axis=2)]
    else:
        for t in reversed(range(h)):
            Q[t] = np.einsum("san,san->sa", T, R + discount * V[t + 1][None, None])
            V[t] = logsumexp(beta * Q[t], axis=1) / beta

        policy = np.exp(beta * (Q - V[:-1, :, None]))
        policy /= policy.sum(axis=2, keepdims=True)

    return _policy_matrix_to_predict_fn(policy)


env_name_to_expert_fn = {
    "Branching-v0": get_value_iteration_expert_fn,
    "EarlyTermNeg-v0": get_early_term_neg_expert_fn,
    "EarlyTermPos-v0": get_early_term_pos_expert_fn,
    "InitShiftTrain-v0": get_value_iteration_expert_fn,
    "InitShiftTest-v0": get_value_iteration_expert_fn,
    "LargestSum-v0": get_largest_sum_expert_fn,
    "NoisyObs-v0": get_noisy_obs_expert_fn,
    "Parabola-v0": get_parabola_expert_fn,
    "ProcGoal-v0": get_proc_goal_expert_fn,
    "RiskyPath-v0": get_value_iteration_expert_fn,
    "Sort-v0": get_sort_expert_fn,
}
