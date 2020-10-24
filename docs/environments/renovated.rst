.. _renovated:

Renovated Environments
======================

These environments are adaptations of widely-used reinforcement learning benchmarks from
`Gym <https://github.com/openai/gym>`_, modified to be suitable for benchmarking specification
learning algorithms. In particular, we:

    * Make episodes fixed length. Since episode termination conditions are often correlated with
      reward, variable-length episodes provide a side-channel of reward information that algorithms
      can exploit. Critically, episode boundaries do not exist outside of simulation: in the
      real-world, a human must often `"reset" the RL algorithm <https://www.youtube.com/watch?time_continue=125&v=vw3mGAlsT2U>`_.

      Moreover, many algorithms do not properly handle episode termination, and so are
      `biased <https://arxiv.org/abs/1809.02925>`_ towards shorter or longer episode boundaries.
      This confounds evaluation, making some algorithms appear spuriously good or bad depending
      on if their bias aligns with the task objective.

      For most tasks, we make the episode fixed length simply by removing the early termination
      condition. In some environments, such as *MountainCar*, it does not make sense to continue
      after the terminal state: in this case, we make the terminal state an absorbing state that
      is repeated until the end of the episode.
    * Ensure observations include all information necessary to compute the ground-truth reward
      function. For some environments, this has required augmenting the observation space.
      We make this modification to make RL and specification learning of comparable difficulty
      in these environments. While in general both RL and specification learning may need to
      operate in partially observable environments, the observations in these relatively simple
      environments were typically engineered to *make RL easy*: for a fair comparison, we must
      therefore also provide reward learning algorithms with sufficient features to recover the
      reward.

In the future, we intend to add Atari tasks with the score masked, another reward side-channel.

Classic Control
---------------

CartPole
********

**Gym ID**: ``seals/CartPole-v0``

.. autoclass:: seals.classic_control.FixedHorizonCartPole

MountainCar
***********

**Gym ID**: ``seals/MountainCar-v0``

.. autofunction:: seals.classic_control.mountain_car

MuJoCo
------

Ant
***

**Gym ID**: ``seals/Ant-v0``

.. autoclass:: seals.mujoco.AntEnv

HalfCheetah
***********

**Gym ID**: ``seals/HalfCheetah-v0``

.. autoclass:: seals.mujoco.HalfCheetahEnv

Hopper
******

**Gym ID**: ``seals/Hopper-v0``

.. autoclass:: seals.mujoco.HopperEnv

Humanoid
********

**Gym ID**: ``seals/Humanoid-v0``

.. autoclass:: seals.mujoco.HumanoidEnv

Swimmer
*******

**Gym ID**: ``seals/Swimmer-v0``

.. autoclass:: seals.mujoco.SwimmerEnv

Walker2d
********

**Gym ID**: ``seals/Walker2d-v0``

.. autoclass:: seals.mujoco.Walker2dEnv
