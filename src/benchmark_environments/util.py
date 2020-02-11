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
