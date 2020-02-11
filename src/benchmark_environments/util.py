"""Miscellaneous utilities."""
import gym


class AutoResetWrapper(gym.Wrapper):
    """Hides done=True and auto-resets at the end of each episode."""

    def step(self, action):
        """When done=True, returns done=False instead and automatically resets.

        When an automatic reset happens, the observation from reset is returned,
        and the overridden observation is stored in
        `info["terminal_observation"]`.
        """
        obs, rew, done, info = self.env.step(action)
        if done:
            info["terminal_observation"] = obs
            obs = self.env.reset()
        return obs, rew, False, info


class EpisodeEndRewardWrapper(gym.Wrapper):
    """Replaces all rewards, with all rewards becoming zero except episode end.

    Useful for converting living rewards into equivalent episode termination
    rewards in environments like CartPole and MountainCar.
    """

    def __init__(self, env: gym.Env, episode_end_reward: float):
        """
        Params:
          env: The wrapped environment.
          episode_end_reward: All rewards are zero except the episode end reward,
            which has this value.
        """
        super().__init__(env)
        self.episode_end_reward = episode_end_reward

    def step(self, action):
        """Wraps `env.step()` to replace rewards."""
        obs, _, done, info = self.env.step(action)
        if done:
            rew = self.episode_end_reward
        else:
            rew = 0
        return obs, rew, done, info
