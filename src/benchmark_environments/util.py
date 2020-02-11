import gym


class AutoResetWrapper(gym.Wrapper):
    """Hides done=True and auto-resets at the end of each episode."""

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            info["terminal_observation"] = obs
            obs = self.env.reset()
        return obs, rew, False, info
