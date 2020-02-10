import gym
from gym.spaces import Discrete, Tuple, MultiDiscrete

from run_env import airl

import sys
import time

def _clamp(val, min_val, max_val):
    if val < min_val:
        return min_val
    elif max_val < val:
        return max_val
    else:
        return val


class SimpleCorridor(gym.Env):
    def __init__(self):
        self._corridor_radius = 3
        self._num_positions = 1 + 2 * self._corridor_radius

        self.observation_space = MultiDiscrete((self._num_positions, 2, 2))
        self.action_space = Discrete(3)

        self.max_episode_steps = 10

    def reset(self):
        self.t = 0
        pos = 0
        has_bomb = True
        has_coin = True
        self._state = (pos, has_bomb, has_coin)
        return self._state

    def step(self, act):
        assert act in self.action_space

        pos, has_bomb, has_coin = self._state
        pos = pos + self._get_movement(act)
        pos = _clamp(pos,
                    -self._corridor_radius, self._corridor_radius)

        reward = 0
        if pos == -self._corridor_radius and has_bomb:
            reward = -1
            has_bomb = False
        elif pos == self._corridor_radius and has_coin:
            reward = 1
            has_coin = False

        self._state = (pos, has_bomb, has_coin)

        self.t += 1
        done = self.t >= self.max_episode_steps
        info = {}
        return self._state, reward, done, info

    def _get_movement(self, act):
        # 0 is LEFT, 1 is NO-OP, 2 is RIGHT
        return act - 1

    def render(self, header=True):
        outfile = sys.stdout
        width = 2 + self._num_positions
        height = 3

        out = [
                list(width * '-'),
                list(width * ' '),
                list(width * '-'),
                ]

        out[1][0] = '|'
        out[1][-1] = '|'

        pos, has_bomb, has_coin = self._state
        if has_bomb:
            out[1][1] = 'B'

        if has_coin:
            out[1][-2] = 'C'

        map_pos = 1 + self._corridor_radius + pos

        out[1][map_pos] = 'o'

        out_str = ''
        if header:
            out_str += f'Step {self.t}/{self.max_episode_steps}\n'
        
        for line in out:
            out_str += ''.join(line)
            out_str += '\n'
        out_str += '\n'
        
        outfile.write(out_str) 


def main():
    from stable_baselines import PPO2, DQN
    from stable_baselines.common.evaluation import evaluate_policy
    from stable_baselines.common.policies import MlpPolicy

    env = SimpleCorridor()

    model = PPO2(MlpPolicy, env)
    model.learn(total_timesteps=1000)

    gen, disc = airl(env, model, irl_timesteps=1000)

    def run_and_render_agent(agent):
        ob = env.reset()
        env.render()
        time.sleep(0.4)
        done = False
        while not done:
            ac, _ = agent.predict(ob)
            ob , _, done, _ = env.step(ac)
            env.render()
            time.sleep(0.4)


    print('** EXPERT **')
    for _ in range(10):
        run_and_render_agent(model)

    print('\n** IRL **')
    for _ in range(10):
        run_and_render_agent(gen)

if __name__ == '__main__':
    main()
