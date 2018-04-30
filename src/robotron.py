# -*- coding: utf-8 -*-

import environment
import control
from nn import DQN

import pickle
import copy
import os
import sys
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from scipy.misc import imresize
from builtins import range

MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = (80,110)
K = 64  # 8 movement directions by 8 shooting directions


def downsample_image(img):
    return imresize(img, size=IM_SIZE, interp='nearest')


def update_state(state, obs):
    obs_small = downsample_image(obs)
    return np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)


def split_action(a):
    left = a / 8
    right = a % 8
    return (int(left) + 1, int(right) + 1)


def reset(env, out):
    env.reset()
    out.reset()
    return env.process(True)


def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
    # Sample experiences
    samples = random.sample(experience_replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

    # Calculate targets
    next_Qs = target_model.predict(next_states)
    next_Q = np.amax(next_Qs, axis=1)
    targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q

    # Update model
    loss = model.update(states, actions, targets)
    return loss


def play_one(
        env,
        out,
        total_t,
        experience_replay_buffer,
        model,
        target_model,
        gamma,
        batch_size,
        epsilon,
        epsilon_change,
        epsilon_min):

    t0 = datetime.now()

    # Reset the environment
    obs = reset(env, out)
    obs_small = downsample_image(obs)
    state = np.stack([obs_small] * 4, axis=0)
    # assert(state.shape == (4, 80, 80))
    loss = None

    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = 0

    done = False
    while not done:
        # Update target network
        if total_t % TARGET_UPDATE_PERIOD == 0:
            target_model.copy_from(model)
            print("Copied model parameters to target network. total_t = %s, period = %s              " % (
                total_t, TARGET_UPDATE_PERIOD))

        # Take action
        action = model.sample_action(state, epsilon)
        (l, r) = split_action(action)
        out.move_and_shoot(r, l)
        (active, obs, reward, done) = env.process()

        obs_small = downsample_image(obs)
        next_state = np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)
        # assert(state.shape == (4, 80, 80))

        episode_reward += reward

        # Remove oldest experience if replay buffer is full
        if len(experience_replay_buffer) == MAX_EXPERIENCES:
            experience_replay_buffer.pop(0)

        # Save the latest experience
        experience_replay_buffer.append(
            (state, action, reward, next_state, done))

        # Train the model, keep track of time
        t0_2 = datetime.now()
        loss = learn(model, target_model,
                     experience_replay_buffer, gamma, batch_size)
        dt = datetime.now() - t0_2

        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1

        state = next_state
        total_t += 1

        epsilon = max(epsilon - epsilon_change, epsilon_min)

    print("\n")
    return total_t, episode_reward, (datetime.now() - t0), num_steps_in_episode, \
        total_time_training / num_steps_in_episode, epsilon, loss


def main():
    # hyperparams and initialize stuff
    conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    hidden_layer_sizes = [512]
    gamma = 0.99
    batch_sz = 32
    num_episodes = 1000000
    total_t = 0
    experience_replay_buffer = []
    episode_rewards = np.zeros(num_episodes)
    c_in = control.Controller()
    out = control.Output()
    env = environment.Environment()

    # epsilon
    # decays linearly until 0.1
    epsilon = 0.14
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 500000

    # Create models
    model = DQN(
        K=K,
        conv_layer_sizes=conv_layer_sizes,
        hidden_layer_sizes=hidden_layer_sizes,
        gamma=gamma,
        scope="model")
    target_model = DQN(
        K=K,
        conv_layer_sizes=conv_layer_sizes,
        hidden_layer_sizes=hidden_layer_sizes,
        gamma=gamma,
        scope="target_model"
    )

    # Use controller input to get into the game
    print("Using Controller Input to control game... press the XBox button to continue.")
    c_in.run(out)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        model.set_session(sess)
        target_model.set_session(sess)
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, './models/current.ckpt')
        print("Populating experience replay buffer...")
        obs = reset(env, out)
        print(obs.shape)
        obs_small = downsample_image(obs)
        state = np.stack([obs_small] * 4, axis=0)
        #assert(state.shape == (4, IM_SIZE, IM_SIZE))

        for i in range(MIN_EXPERIENCES):
            (active, obs, reward, done) = env.process()
            while not active:
                out.none()
                (active, obs, reward, done) = env.process()
            action = np.random.choice(8 * 8)
            (l, r) = split_action(action)
            out.move_and_shoot(r, l)
            next_state = update_state(state, obs)
            experience_replay_buffer.append((state, action, reward, next_state, done))

            if done:
                print("\nPopulating experience replay buffer...", i, "of", MIN_EXPERIENCES)
                obs = reset(env, out)
                obs_small = downsample_image(obs)
                state = np.stack([obs_small] * 4, axis=0)
                # assert(state.shape == (4, 80, 80))
            else:
                state = next_state

        # Play a number of episodes and learn!
        print("Beginning DQN learning")
        for i in range(num_episodes):

            total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon, loss = play_one(
                env,
                out,
                total_t,
                experience_replay_buffer,
                model,
                target_model,
                gamma,
                batch_sz,
                epsilon,
                epsilon_change,
                epsilon_min,
            )
            episode_rewards[i] = episode_reward

            last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
            print("Episode:", i,
                  "Duration:", duration,
                  "Num steps:", num_steps_in_episode,
                  "Reward:", episode_reward,
                  "Training time per step:", "%.3f" % time_per_step,
                  "Avg Reward (Last 100):", "%.3f" % last_100_avg,
                  "Epsilon:", "%.3f" % epsilon,
                  "Loss:", "%.3f" % loss,
                  )
            sys.stdout.flush()
            if i % 100 == 0:
                saver.save(sess, './models/current.ckpt')
                saver.save(sess, './models/episode_{}.ckpt'.format(i))

                print("Checkpoint Saved.")


if __name__ == '__main__':
    main()
