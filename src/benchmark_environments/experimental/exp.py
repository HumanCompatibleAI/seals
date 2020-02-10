from functools import partial

import gym
import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from imitation.policies.base import FeedForward32Policy
from imitation.rewards.discrim_net import DiscrimNetAIRL
from imitation.rewards.reward_net import BasicShapedRewardNet

from imitation.util.buffering_wrapper import BufferingWrapper
from imitation.util import buffer, make_session, reward_wrapper
from imitation.util.rollout import make_sample_until, generate_trajectories, unwrap_traj, flatten_trajectories

import tensorflow as tf

# def run(env_wrapper, irl_fn, eval_fn):
#     expert = env_wrapper.get_expert()
#     irl_result = irl_fn(expert)
#     return eval_fn(expert, irl_result)

def exp1():
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    exp_reward, gen_reward = run_airl_1(env)
    print(f'exp_reward : {exp_reward}')
    print(f'gen_reward : {gen_reward}')


def run_airl_1(env):
    expert = PPO2(MlpPolicy, env)

    expert_timesteps = 1000
    expert.learn(total_timesteps=expert_timesteps)

    generator, discrim = airl(env, expert)

    exp_reward, _ = evaluate_policy(expert, env)
    gen_reward, _ = evaluate_policy(generator, env)

    return exp_reward, gen_reward


class PPOWrapper:
    def __init__(self, env_name, train_timesteps):
        self.env = gym.make(env_name)
        self.model = PPO2(MlpPolicy, env)
        self.train_timesteps = train_timesteps

    def get_expert(self):
        self.model.learn(total_timesteps=self.train_timesteps)
        return self.model


def airl(env, expert):
    venv = DummyVecEnv([lambda: env])

    trajectories = generate_trajectories(
            expert,
            venv,
            sample_until=make_sample_until(n_timesteps=None, n_episodes=10),
           )
    trajectories = flatten_trajectories(trajectories)

    gen_policy, discrim = airl_base(venv, trajectories)
    return gen_policy, discrim


def airl_base(venv, expert_trajectories):
    gen_batch_size = 1000
    disc_batch_size = 1000
    disc_minibatch_size = disc_batch_size // 2
    irl_timesteps = 2000


    with make_session():
        sess = tf.get_default_session()

        gen_policy = PPO2(policy=FeedForward32Policy,
                          env=venv,
                          learning_rate=3e-4,
                          nminibatches=32,
                          noptepochs=10,
                          ent_coef=0.0,
                         )

        rn = BasicShapedRewardNet(venv.observation_space,
                                  venv.action_space,
                                  theta_units=[32, 32],
                                  phi_units=[32, 32],
                                  scale=True)
        discrim = DiscrimNetAIRL(rn, entropy_weight=1.0)

        train_op = tf.train.AdamOptimizer().minimize(
                        tf.reduce_mean(discrim.disc_loss))

        reward_train = partial(discrim.reward_train,
                gen_log_prob_fn=gen_policy.action_probability)
        venv_train = reward_wrapper.RewardVecEnvWrapper(
            venv, reward_train)


        gen_replay_buffer_capacity = 20 * gen_batch_size
        gen_replay_buffer = buffer.ReplayBuffer(gen_replay_buffer_capacity, venv)

        exp_replay_buffer = buffer.ReplayBuffer.from_data(expert_trajectories)

        buffered_venv = BufferingWrapper(venv)
        gen_policy.set_env(buffered_venv)

        sess.run(tf.global_variables_initializer())

        num_epochs = int(irl_timesteps // gen_batch_size)
        for epoch in range(num_epochs):
            # Train gen
            gen_policy.learn(total_timesteps=gen_batch_size,
                             reset_num_timesteps=False)
            gen_samples = buffered_venv.pop_transitions()
            gen_replay_buffer.store(gen_samples)

            # Train disc
            n_updates = disc_batch_size // disc_minibatch_size
            n_samples_per_set = disc_minibatch_size // 2
            gen_samples = gen_replay_buffer.sample(n_samples_per_set)

            expert_samples = exp_replay_buffer.sample(n_samples_per_set)


            obs = np.concatenate([gen_samples.obs,
                                  expert_samples.obs])
            acts = np.concatenate([gen_samples.acts,
                                   expert_samples.acts])
            next_obs = np.concatenate([gen_samples.next_obs,
                                       expert_samples.next_obs])
            labels = np.concatenate(
                    [np.zeros(n_samples_per_set),
                     np.ones(n_samples_per_set)])

            log_act_prob = gen_policy.action_probability(obs, actions=acts, logp=True)
            log_act_prob = log_act_prob.reshape((disc_minibatch_size,))

            sess.run(train_op, feed_dict={
                discrim.obs_ph: obs,
                discrim.act_ph: acts,
                discrim.next_obs_ph: next_obs,
                discrim.labels_ph: labels,
                discrim.log_policy_act_prob_ph: log_act_prob,
                })


    return gen_policy, discrim


def mce_irl(venv, reward_model, expert_trajectories):
    """Maximum Causal Entropy Inverse Reinforcement Learning.

    The idea here is to run an optimization procedure to find the optimal set of policies, given trajectories (i.e. the policies that maximize the Maximum Likelihood of those trajectories being taken).
    Since there can be multiple optimal policies, we select the one that has maximal causal entropy.

    """  
    def compute_occupancy_measure(transitions, state_rewards):
        num_states = len(state_rewards)
        D = np.zeros(horizon, num_states)
        D[0, :] = np.ones(num_states) / num_states

        expected_gain = np.zeros(num_states, num_states)
        for s in range(num_states):
            expected_gain[s, :] = state_rewards @ transitions[s, :, :]

        for t in range(1, horizon + 1):
            D[t, :] = expected_gain @ D[t-1, :]

        return D.sum(axis=0)
        

    visitation_count = Counter(expert_trajectories.obs)

    EPS = 1e-3
    occup_diff = EPS + 1
    grad_norm = EPS + 1
    while occup_diff > EPS and grad_norm > EPS:
        state_rewards, state_rewards_grad = reward_model.get_rewards_and_grads()

        occupancy_measure = compute_occupancy_measure(
                transitions,
                state_rewards,
               )

        # Matrix mult probably wrong, fix this
        num_grad = visitation_count @ state_reward_grads
        den_grad = occupancy_measure @ state_reward_grads
        grad = num_grad - den_grad

        reward_model.params += alpha * grad
        
        occup_diff = np.max(np.abs(occupancy_measure - visitation_count))
        grad_norm = np.linalg.norm(grad)

    return reward_model


def preferences_irl(venv, expert):

    target = expert.reward_model

    n_trajectories = 2 * batch_size

    for _ in range(n_batches):
        trajectories = generate_trajectories(venv, model, n_trajectories) 
        trajectories = trajectories.reshape(batch_size, 2)

        traj_rewards = get_traj_reward(trajectories)
        traj_prefs = traj_rewards[:, 0] >= traj_rewards[:, 1]
        traj_labels = concatenate(traj_prefs, 1 - traj_prefs)
        
        estimated_rewards = model.get_rewards(trajectories)
        traj_log_prob = tf.log_softmax(estimated_rewards, axis=1)

        preferred_log_probs = tf.reduce_sum(traj_labels * traj_log_prob, axis=1)
        cross = (-1) * tf.reduce_mean(preferred_log_probs)
    
        optimizer.minimize(cross)

    return target


def main():
    exp1()


if __name__ == '__main__':
    main()
