.. _install:

Installation Instructions
=========================

To install the latest release from PyPi, run::

    pip install seals

We make releases periodically, but if you wish to use the latest version of the code, you can
always install directly from Git master::

    pip install git+https://github.com/HumanCompatibleAI/seals.git

*seals* has optional dependencies needed by some subset of environments. In particular,
to use MuJoCo environments, you will need to install `MuJoCo <http://www.mujoco.org/>`_ 1.5
and then run::

    pip install seals[mujoco]

You may need to install some other binary dependencies: see the instructions in
`Gym <https://github.com/openai/gym>`_ and `mujoco-py <https://github.com/openai/mujoco-py>`_
for further information.

You can also use our Docker image which includes all necessary binary dependencies. You can either
build it from the ``Dockerfile``, or by downloading a pre-built image::

    docker pull humancompatibleai/seals:base
