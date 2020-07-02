seals User Guide
================

The Suite of Environments for Algorithms that Learn Specifications, or *seals*, is a toolkit for
evaluating specification learning algorithms, such as reward or imitation learning. The environments
are compatible with `Gym <https://github.com/openai/gym/>`_, but are designed to test algorithms
that learn from user data, without requiring a procedurally specified reward function.

There are two types of environments in *seals*:

    * **Diagnostic Tasks** which test individual facets of algorithm performance in isolation.
    * **Renovated Environments**, adaptations of widely-used benchmarks such as MuJoCo continuous
      control tasks to be suitable for specification learning benchmarks. In particular, this
      involves removing any side-channel sources of reward information (such as episode boundaries,
      the score appearing in the observation, etc) and including all the information needed to
      compute the reward in the observation space.

*seals* is under active development and we intend to add more categories of tasks soon.

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   guide/install


.. toctree::
   :maxdepth: 3
   :caption: Environments

   environments/diagnostic
   environments/renovated

.. toctree::
   :maxdepth: 2
   :caption: Common

   common/base_envs
   common/util
   common/testing

Citing seals
------------
To cite this project in publications:

.. code-block:: bibtex

    @misc{seals,
      author = {Adam Gleave and Pedro Freire and Steven Wang and Sam Toyer},
      title = {{seals}: Suite of Environments for Algorithms that Learn Specifications},
      year = {2020},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/HumanCompatibleAI/seals}},
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
