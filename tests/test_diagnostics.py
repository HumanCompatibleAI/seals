"""Test the `diagnostics.*` environments."""

import pytest

import gym
import stable_baselines3
from stable_baselines3.common import evaluation

from seals.diagnostics import init_shift
import seals.diagnostics
import seals.diagnostics.experts


def test_init_shift_validation():
    """Test input validation for init_shift.InitShiftEnv."""
    for invalid_state in [-1, 7, 8, 100]:
        with pytest.raises(ValueError, match=r"Initial state.*"):
            init_shift.InitShiftEnv(initial_state=invalid_state)


_env_names = [info[0] for info in seals.diagnostics.envs_info]


@pytest.mark.parametrize("env_name", _env_names)
def test_experts(env_name):
    """Test whether specified expert has non-trivial performance."""
    env = gym.make(f"seals/{env_name}")
    get_return = lambda m: evaluation.evaluate_policy(m, env)[0]
    total_timesteps = 1000

    ppo_model = stable_baselines3.PPO("MlpPolicy", env)
    ppo_model.learn(total_timesteps=total_timesteps)

    expert_fn = seals.diagnostics.experts.env_name_to_expert_fn[env_name](env)

    class ExpertModel:
        def predict(self, *args, **kwargs):
            return expert_fn(*args, **kwargs)

    expert = ExpertModel()

    assert get_return(ppo_model) <= get_return(expert)
