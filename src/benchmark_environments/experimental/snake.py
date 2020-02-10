import gym
from gym.spaces import Discrete, Tuple, MultiDiscrete

import random
import sys
import time

def _clamp(val, min_val, max_val):
    if val < min_val:
        return min_val
    elif max_val < val:
        return max_val
    else:
        return val


class SnakeEnv(gym.Env):
    def __init__(self):
        self._corridor_radius = 3
        self._width = 5
        self._height = 5

        self._snake_len = 1

        dims = []
        for _ in range(self._snake_len + 1):
            dims += [self._width, self._height]
        dims = tuple(dims)
        self.observation_space = MultiDiscrete(dims)

        self.action_space = Discrete(4)

        self.max_episode_steps = 50

    def reset(self):
        self.t = 0
        self._snake = [(x, 1) for x in reversed(range(1, 1 + self._snake_len))]
        self._dead = False
        self._generate_new_fruit()
        return self._get_state()

    def _get_state(self):
        state = []
        for x, y in self._snake:
            state.extend([x, y])
        x, y = self._fruit_pos
        state.extend([x, y])
        return tuple(state)

    def step(self, act):
        assert act in self.action_space

        head_pos = self._snake[0]
        new_head_pos = self._move(head_pos, act)

        if new_head_pos != head_pos:
            self._snake.pop()
            self._snake = [new_head_pos] + self._snake

        reward = 0
        if new_head_pos == self._fruit_pos:
            reward += 1
            self._generate_new_fruit()

        def out_of_bounds(pos):
            x, y = pos
            return not (0 <= x < self._width and
                        0 <= y < self._height)

        if new_head_pos in self._snake[1:] or out_of_bounds(new_head_pos):
            pass
            self._dead = True

        self.t += 1
        done = self.t >= self.max_episode_steps
        info = {}
        return self._get_state(), reward, done, info

    def _move(self, head_pos, act):
        if self._dead:
            return head_pos

        moves = [
                (1, 0),
                (0, -1),
                (-1, 0),
                (0, 1),
                ]
        dx, dy = moves[act]
        x, y = head_pos
        x += dx
        y += dy
        # x = _clamp(x, 0, self._width - 1)
        # y = _clamp(y, 0, self._height - 1)
        return (x, y)

    def _generate_new_fruit(self):
        for _ in range(100):
            x = random.randrange(0, self._width)
            y = random.randrange(0, self._height)
            pos = (x, y)
            if pos not in self._snake:
                self._fruit_pos = pos
                return pos
        raise Exception('Could not generate new fruit')

    def render(self, header=True):
        outfile = sys.stdout

        full_width = self._width + 2
        full_height = self._height + 2

        out = [[' ' for _ in range(full_width)] for _ in range(full_height)]

        out[0] = ['-' for _ in range(full_width)]
        out[-1] = ['-' for _ in range(full_width)]
        for i in range(1, full_height-1):
            out[i][0] = '|'
            out[i][-1] = '|'


        def local_to_global_pos(pos):
            x, y = pos
            return (x+1, y+1)

        for i, pos in enumerate(self._snake):
            x, y = local_to_global_pos(pos)
            is_head = i == 0
            out[y][x] = 'S' if is_head else 'o'

        x, y = local_to_global_pos(self._fruit_pos)
        out[y][x] = '@'

        if self._dead:
            x_head, y_head = local_to_global_pos(self._snake[0])
            out[y_head][x_head] = 'X'

        out_str = ''
        if header:
            out_str += f'Step {self.t}/{self.max_episode_steps}\n'
        
        for line in out:
            out_str += ''.join(line)
            out_str += '\n'
        out_str += '\n'

        
        outfile.write(out_str) 

def render_random():
    env = SnakeEnv()

    policy = lambda ob : (env.action_space.sample(), None)

    sleep_delay = 0.1

    def run_and_render_agent(policy):
        ob = env.reset()
        env.render()
        time.sleep(sleep_delay)
        done = False
        while not done:
            ac, _ = policy(ob)
            ob , _, done, _ = env.step(ac)
            env.render()
            time.sleep(sleep_delay)


    print('** RANDOM **')
    for _ in range(10):
        run_and_render_agent(policy)


def exp_and_irl():
    from run_env import airl
    from stable_baselines import PPO2, DQN
    from stable_baselines.common.evaluation import evaluate_policy
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.deepq import MlpPolicy as DqnMlpPolicy

    env = SnakeEnv()

    # model = PPO2(MlpPolicy, env)
    model = DQN(DqnMlpPolicy, env)
    model.learn(total_timesteps=100000)

    def get_dqn_policy_fn(env_):
        return DQN(DqnMlpPolicy, env_)

    gen, disc = airl(env, model, irl_timesteps=300000,
                     gen_batch_size=1000,
                     disc_batch_size=800,
                     gen_policy_fn=get_dqn_policy_fn,
                    )

    def run_and_render_agent(agent, header=None):
        def render():
            if header:
                print(header)
            env.render()

        sleep_delay = 0.1
        ob = env.reset()
        render()
        time.sleep(sleep_delay)
        done = False
        while not done:
            ac, _ = agent.predict(ob)
            ob , _, done, _ = env.step(ac)
            render()
            time.sleep(sleep_delay)


    for i in range(10):
        run_and_render_agent(model, header=f'** EXPERT {i}/10 **')

    for i in range(10):
        run_and_render_agent(gen, header=f'** IRL {i}/10 **')


    exp_reward, _ = evaluate_policy(model, env)
    gen_reward, _ = evaluate_policy(gen, env)

    print(f'exp_reward : {exp_reward}')
    print(f'gen_reward : {gen_reward}')


if __name__ == '__main__':
    # render_random()
    exp_and_irl()
    # main()
