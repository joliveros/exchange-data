#!/usr/bin/env python

import tensorflow as tf
from tensorflow_core.python.keras.metrics import Mean
from tensorflow_core.python.ops.summary_ops_v2 import create_file_writer

from exchange_data.agents.a2c.actor_model import ActorModel
from exchange_data.agents.a2c.critic_model import CriticModel
from tensorflow.keras import optimizers, losses

import gym
import numpy as np



class ActorCriticTrain:
    def __init__(self):
        # hyper parameters
        self.env = gym.make('CartPole-v0')
        self.num_action = self.env.action_space.n
        self.lr =0.001
        self.lr2 = 0.001
        self.df = 0.99
        self.en = 0.001

        self.actor_model = ActorModel(num_action=self.num_action)
        self.actor_opt = optimizers.Adam(lr=self.lr, )

        self.critic_model = CriticModel()
        self.critic_opt = optimizers.Adam(lr=self.lr2, )

        # tensorboard
        self.log_dir = 'logs/'
        self.train_summary_writer = create_file_writer(self.log_dir)
        self.reward_board = Mean('reward_board', dtype=tf.float32)
        #self.train_loss = Mean('train_loss', dtype=tf.float32)
        #self.train_loss_c = Mean('train_loss_c', dtype=tf.float32)

    def actor_loss(self, states, actions, advantages):
        policy = self.actor_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))

        # SparseCategoricalCrossentropy = ce loss with not one-hot encoded output
        # from_logits = True  =>  cross_entropy with soft_max
        entropy = losses.categorical_crossentropy(policy, policy, from_logits=False)
        ce_loss = losses.SparseCategoricalCrossentropy(from_logits=False)
        # policy_loss = ce_loss(actions, policy, sample_weight=np.array(advantages))  # same way
        log_pi = ce_loss(actions, policy)
        policy_loss = log_pi * np.array(advantages)
        policy_loss = tf.reduce_mean(policy_loss)

        return policy_loss - self.en * entropy

    def critic_loss(self, states, rewards, dones):
        last_state = states[-1]
        if dones[-1] == True :
            reward_sum = 0
        else :
            reward_sum = self.critic_model(
                tf.convert_to_tensor(last_state[None, :], dtype=tf.float32)
            )

        discounted_rewards = []

        for reward in rewards[::-1]:
            reward_sum = reward + self.df * reward_sum
            discounted_rewards.append(reward_sum)

        discounted_rewards.reverse()
        discounted_rewards = tf.convert_to_tensor(
            np.array(discounted_rewards)[:, None], dtype=tf.float32
        )
        values = self.critic_model(tf.convert_to_tensor(
            np.vstack(states), dtype=tf.float32)
        )
        error = tf.square(values - discounted_rewards)*0.5
        error = tf.reduce_mean(error)
        return error

    def train(self, states, actions, rewards, next_states, dones):

        critic_variable = self.critic_model.trainable_variables

        with tf.GradientTape() as tape_critic:
            tape_critic.watch(critic_variable)
            critic_loss = self.critic_loss(states, rewards, dones)

        # gradient descent will be applied automatically
        critic_grads = tape_critic.gradient(critic_loss, critic_variable)
        self.critic_opt.apply_gradients(zip(critic_grads, critic_variable))

        advantages = self.compute_advantages(states, rewards, dones)
        actor_variable = self.actor_model.trainable_variables

        with tf.GradientTape() as tape:
            tape.watch(actor_variable)
            actor_loss = self.actor_loss(states, actions, advantages)

        actor_grads = tape.gradient(actor_loss, actor_variable)
        self.actor_opt.apply_gradients(zip(actor_grads, actor_variable))

        self.train_loss = tf.reduce_mean(actor_loss)
        self.train_loss_c = tf.reduce_mean(critic_loss)

    def compute_advantages(self, states, rewards, dones):
        last_state = states[-1]

        if dones[-1] == True:
            reward_sum = 0
        else:
            reward_sum = self.critic_model(
                tf.convert_to_tensor(last_state[None, :], dtype=tf.float32)
            )

        discounted_rewards = []

        for reward in rewards[::-1]:
            reward_sum = reward + self.df * reward_sum
            discounted_rewards.append(reward_sum)

        discounted_rewards.reverse()

        values = self.critic_model(tf.convert_to_tensor(
            np.vstack(states), dtype=tf.float32)
        )

        advantages = discounted_rewards - values
        return advantages

    def run(self):
        env = self.env
        t_end = 500
        epi = 100000
        train_size = 20

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        state = env.reset()

        for e in range(epi):
            total_reward = 0
            for t in range(t_end):
                policy = self.actor_model(
                    tf.convert_to_tensor(state[None, :], dtype=tf.float32)
                )
                action = tf.squeeze(tf.random.categorical(policy, 1), axis=-1)
                action = np.array(action)[0]
                next_state, reward, done, _ = env.step(action)

                #env.render()
                if t == t_end :
                    done = True
                    reward += 10
                if t < t_end and done :
                    reward = -1

                total_reward += reward

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state


                if len(states) == train_size or done :
                    self.train(states, actions, rewards, next_states, dones)
                    states = []
                    actions = []
                    rewards = []
                    next_states = []
                    dones = []

                if done:
                    self.reward_board(total_reward)
                    print("e : ", e, " reward : ", total_reward, " step : ", t)
                    env.reset()
                    with self.train_summary_writer.as_default():
                        # tf.summary.scalar('reward', self.reward_board.result(), step=e)
                        tf.summary.scalar('actor_loss', self.train_loss, step=e)
                        tf.summary.scalar('critic_loss', self.train_loss_c, step=e)
                        tf.summary.scalar('reward', total_reward, step=e)
                    break


if __name__ == '__main__':
    ActorCritic = ActorCriticTrain()
    ActorCritic.run()
