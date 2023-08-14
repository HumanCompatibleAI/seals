"""Smoke tests for all environments."""

from typing import List, Union

import gymnasium as gym
from gymnasium.envs import registration
import numpy as np
import pytest

import seals  # noqa: F401 required for env registration
from seals.atari import SCORE_REGIONS, _get_score_region, _seals_name, make_atari_env
from seals.testing import envs
from seals.testing.envs import is_mujoco_env

ENV_NAMES: List[str] = [
    env_id for env_id in registration.registry.keys() if env_id.startswith("seals/")
]


DETERMINISTIC_ENVS: List[str] = [
    "seals/EarlyTermPos-v0",
    "seals/EarlyTermNeg-v0",
    "seals/Branching-v0",
    "seals/InitShiftTrain-v0",
    "seals/InitShiftTest-v0",
]

UNMASKED_ATARI_ENVS: List[str] = [
    _seals_name(gym_spec, masked=False) for gym_spec in seals.GYM_ATARI_ENV_SPECS
]
MASKED_ATARI_ENVS: List[str] = [
    _seals_name(gym_spec, masked=True)
    for gym_spec in seals.GYM_ATARI_ENV_SPECS
    if _get_score_region(gym_spec.id) is not None
]
ATARI_ENVS = UNMASKED_ATARI_ENVS + MASKED_ATARI_ENVS

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
    """Tests for masked and unmasked Atari space invaders environments."""
    masked_space_invader_environments = list(
        filter(
            lambda name: "SpaceInvaders" in name and "Unmasked" not in name,
            ATARI_ENVS,
        ),
    )
    assert len(masked_space_invader_environments) > 0

    unmasked_space_invader_environments = list(
        filter(
            lambda name: "SpaceInvaders" in name and "Unmasked" in name,
            ATARI_ENVS,
        ),
    )
    assert len(unmasked_space_invader_environments) > 0


def test_atari_unmasked_env_naming():
    """Tests that all unmasked Atari envs have the appropriate name qualifier."""
    noncompliant_envs = list(
        filter(
            lambda name: _get_score_region(name) is None and "Unmasked" not in name,
            ATARI_ENVS,
        ),
    )
    assert len(noncompliant_envs) == 0


def test_make_unsupported_masked_atari_env_throws_error():
    """Tests that making an unsupported masked Atari env throws an error."""
    match_str = (
        "Requested environment does not yet support masking. "
        "See https://github.com/HumanCompatibleAI/seals/issues/61."
    )
    with pytest.raises(ValueError, match=match_str):
        make_atari_env("ALE/Bowling-v5", masked=True)


def test_atari_masks_satisfy_spec():
    """Tests that all Atari masks satisfy the spec."""
    masks_satisfy_spec = [
        mask.x[0] < mask.x[1] and mask.y[0] < mask.y[1]
        for env_regions in SCORE_REGIONS.values()
        for mask in env_regions
    ]
    assert all(masks_satisfy_spec)


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
                "seals/Bowling-Unmasked-v5",
                "seals/Frogger-Unmasked-v5",
                "seals/KingKong-Unmasked-v5",
                "seals/Koolaid-Unmasked-v5",
                "seals/NameThisGame-Unmasked-v5",
                "seals/Casino-Unmasked-v5",
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

    def test_render_modes(self, env_name: str):
        """Tests that all render modes specifeid in the metadata work.

        Note: we only check that no exception is thrown.
        There is no test to see if something reasonable is rendered.
        """
        for mode in gym.make(env_name).metadata["render_modes"]:
            # GIVEN
            env = gym.make(env_name, render_mode=mode)
            env.reset(seed=0)

            # WHEN
            if mode == "rgb_array" and not is_mujoco_env(env):
                # The render should not change without calling `step()`.
                # MuJoCo rendering fails this check, ignore -- not much we can do.
                r1: Union[np.ndarray, List[np.ndarray], None] = env.render()
                r2: Union[np.ndarray, List[np.ndarray], None] = env.render()
                assert r1 is not None
                assert r2 is not None
                assert np.allclose(r1, r2)
            else:
                env.render()

            # THEN
            # no error raised

            # CLEANUP
            env.close()
