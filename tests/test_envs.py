"""Smoke tests for all environments."""

from typing import List

import gym
from gym.envs import registration
import pytest

import seals  # noqa: F401 required for env registration
from seals.atari import _seals_name
from seals.testing import envs

ENV_NAMES: List[str] = [
    env_spec.id
    for env_spec in registration.registry.all()
    if env_spec.id.startswith("seals/")
]


DETERMINISTIC_ENVS: List[str] = [
    "seals/EarlyTermPos-v0",
    "seals/EarlyTermNeg-v0",
    "seals/Branching-v0",
    "seals/InitShiftTrain-v0",
    "seals/InitShiftTest-v0",
]

ATARI_ENVS: List[str] = [
    _seals_name(gym_spec) for gym_spec in seals.GYM_ATARI_ENV_SPECS
]

ATARI_V5_ENVS: List[str] = list(filter(lambda name: name.endswith("-v5"), ATARI_ENVS))
ATARI_NO_FRAMESKIP_ENVS: List[str] = list(
    filter(lambda name: name.endswith("-v4"), ATARI_ENVS),
)

DETERMINISTIC_ENVS += ATARI_NO_FRAMESKIP_ENVS


env = pytest.fixture(envs.make_env_fixture(skip_fn=pytest.skip))


def test_some_atari_envs():
    """Tests if we succeeded in finding any Atari envs."""
    assert len(seals.GYM_ATARI_ENV_SPECS) > 0


def test_atari_space_invaders():
    """Tests if there's an Atari environment called space invaders."""
    space_invader_environments = list(
        filter(
            lambda name: "SpaceInvaders" in name,
            ATARI_ENVS,
        ),
    )
    assert len(space_invader_environments) > 0


@pytest.mark.parametrize("env_name", ENV_NAMES)
class TestEnvs:
    """Battery of simple tests for environments."""

    def test_seed(self, env: gym.Env, env_name: str):
        """Tests environment seeding.

        Deterministic Atari environments are run with fewer seeds to minimize the number
        of resets done in this test suite, since Atari resets take a long time and there
        are many Atari environments.
        """
        if env_name in ATARI_ENVS:
            # these environments take a while for their non-determinism to show.
            slow_random_envs = [
                "seals/Bowling-v5",
                "seals/Frogger-v5",
                "seals/KingKong-v5",
                "seals/Koolaid-v5",
                "seals/NameThisGame-v5",
            ]
            rollout_len = 100 if env_name not in slow_random_envs else 400
            num_seeds = 2 if env_name in ATARI_NO_FRAMESKIP_ENVS else 10
            envs.test_seed(
                env,
                env_name,
                DETERMINISTIC_ENVS,
                rollout_len=rollout_len,
                num_seeds=num_seeds,
            )
        else:
            envs.test_seed(env, env_name, DETERMINISTIC_ENVS)

    def test_premature_step(self, env: gym.Env):
        """Tests if step() before reset() raises error."""
        envs.test_premature_step(env, skip_fn=pytest.skip, raises_fn=pytest.raises)

    def test_rollout_schema(self, env: gym.Env, env_name: str):
        """Tests if environments have correct types on `step()` and `reset()`.

        Atari environments have a very long episode length (~100k observations), so in
        the interest of time we do not run them to the end of their episodes or check
        the return time of `env.step` after the end of the episode.
        """
        if env_name in ATARI_ENVS:
            envs.test_rollout_schema(env, max_steps=1_000, check_episode_ends=False)
        else:
            envs.test_rollout_schema(env)

    def test_render(self, env: gym.Env):
        """Tests `render()` supports modes specified in environment metadata."""
        envs.test_render(env, raises_fn=pytest.raises)
