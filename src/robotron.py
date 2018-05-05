# -*- coding: utf-8 -*-

import environment
import control
from nn import DQN

import warnings
import sys
import random
import numpy as np
from datetime import datetime
from builtins import range

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from scipy.misc import imresize
    import tensorflow as tf


MAX_EXPERIENCES = 10000
MIN_EXPERIENCES = 1000
# MAX_EXPERIENCES = 500000
# MIN_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = (80, 110)
K = 8  # 8 movement directions by 8 shooting directions


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


def learn(model, target_model, state_buffer, experience_replay_buffer, gamma, batch_size):
    # Sample experiences
    replay_buffer = list(np.concatenate((state_buffer, experience_replay_buffer), axis=1))
    samples = random.sample(replay_buffer, batch_size)
    states, next_states, dones, actions, rewards = map(np.array, zip(*samples))

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
        state_replay_buffer,
        movement_replay_buffer,
        shooting_replay_buffer,
        movement_model,
        movement_target_model,
        shooting_model,
        shooting_target_model,
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

    total_time_training = 0
    num_steps_in_episode = 0
    episode_movement_reward = 0
    episode_shooting_reward = 0

    done = False
    while not done:
        # Update target network
        if total_t % TARGET_UPDATE_PERIOD == 0:
            movement_target_model.copy_from(movement_model)
            shooting_target_model.copy_from(shooting_model)
            print("Copied model parameters to target network. total_t = %s, period = %s              " % (
                total_t, TARGET_UPDATE_PERIOD))

        # Take action
        move = movement_model.sample_action(state, epsilon)
        shoot = shooting_model.sample_action(state, epsilon)
        out.move_and_shoot(move + 1, shoot + 1)
        (active, obs, movement_reward, shooting_reward, done) = env.process()

        obs_small = downsample_image(obs)
        next_state = np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)
        # assert(state.shape == (4, 80, 80))

        episode_movement_reward += movement_reward
        episode_shooting_reward += shooting_reward

        # Remove oldest experience if replay buffer is full
        if len(state_replay_buffer) == MAX_EXPERIENCES:
            state_replay_buffer.pop(0)
            movement_replay_buffer.pop(0)
            shooting_replay_buffer.pop(0)

        # Save the latest experience
        state_replay_buffer.append((state, next_state, done))
        movement_replay_buffer.append((move, episode_movement_reward))
        shooting_replay_buffer.append((shoot, episode_shooting_reward))

        # Train the model, keep track of time
        t0_2 = datetime.now()
        move_loss = learn(movement_model, movement_target_model, state_replay_buffer,
                          movement_replay_buffer, gamma, batch_size)
        shoot_loss = learn(shooting_model, shooting_target_model, state_replay_buffer,
                           shooting_replay_buffer, gamma, batch_size)

        dt = datetime.now() - t0_2

        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1

        state = next_state
        total_t += 1

        epsilon = max(epsilon - epsilon_change, epsilon_min)

    return total_t, episode_movement_reward, episode_shooting_reward, (datetime.now() - t0), num_steps_in_episode, \
        total_time_training / num_steps_in_episode, epsilon, (move_loss, shoot_loss)


def main():
    # hyperparams and initialize stuff
    conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    hidden_layer_sizes = [512]
    gamma = 0.99
    batch_sz = 32
    num_episodes = 1000000
    total_t = 0
    state_replay_buffer = []
    movement_replay_buffer = []
    shooting_replay_buffer = []
    movement_episode_rewards = np.zeros(num_episodes)
    shooting_episode_rewards = np.zeros(num_episodes)
    c_in = control.Controller()
    out = control.Output()
    env = environment.Environment()

    # epsilon
    # decays linearly until 0.1
    epsilon = 1
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 500000

    # Create models
    movement_model = DQN(
        K=K,
        conv_layer_sizes=conv_layer_sizes,
        hidden_layer_sizes=hidden_layer_sizes,
        gamma=gamma,
        scope="movement_model")
    movement_target_model = DQN(
        K=K,
        conv_layer_sizes=conv_layer_sizes,
        hidden_layer_sizes=hidden_layer_sizes,
        gamma=gamma,
        scope="movement_target_model"
    )

    shooting_model = DQN(
        K=K,
        conv_layer_sizes=conv_layer_sizes,
        hidden_layer_sizes=hidden_layer_sizes,
        gamma=gamma,
        scope="shooting_model")
    shooting_target_model = DQN(
        K=K,
        conv_layer_sizes=conv_layer_sizes,
        hidden_layer_sizes=hidden_layer_sizes,
        gamma=gamma,
        scope="target_model"
    )

    # Use controller input to get into the game
    print("Using Controller Input to control game... press the XBox button to continue.")
    # c_in.run(out)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        movement_model.set_session(sess)
        movement_target_model.set_session(sess)
        shooting_model.set_session(sess)
        shooting_target_model.set_session(sess)

        # saver.restore(sess, './models/current.ckpt')
        print("Populating experience replay buffer...")
        obs = reset(env, out)
        obs_small = downsample_image(obs)
        state = np.stack([obs_small] * 4, axis=0)
        # assert(state.shape == (4, IM_SIZE, IM_SIZE))

        for i in range(MIN_EXPERIENCES):
            (active, obs, movement_reward, shooting_reward, done) = env.process()
            while not active:
                out.none()
                (active, obs, movement_reward, shooting_reward, done) = env.process()
            move = np.random.choice(8)
            shoot = np.random.choice(8)
            out.move_and_shoot(move + 1, shoot + 1)
            next_state = update_state(state, obs)

            # Save the latest experience
            state_replay_buffer.append((state, next_state, done))
            movement_replay_buffer.append((move, movement_reward))
            shooting_replay_buffer.append((shoot, shooting_reward))

            if done:
                print("\nPopulating experience replay buffer...", i, "of", MIN_EXPERIENCES)
                obs = reset(env, out)
                obs_small = downsample_image(obs)
                state = np.stack([obs_small] * 4, axis=0)
                # assert(state.shape == (4, 80, 80))
            else:
                state = next_state

        # Play a number of episodes and learn!
        print("\nBeginning DQN learning")
        for i in range(num_episodes):

            (total_t, movement_reward, shooting_reward, duration,
             num_steps_in_episode, time_per_step, epsilon, loss) = play_one(
                env,
                out,
                total_t,
                state_replay_buffer,
                movement_replay_buffer,
                shooting_replay_buffer,
                movement_model,
                movement_target_model,
                shooting_model,
                shooting_target_model,
                gamma,
                batch_sz,
                epsilon,
                epsilon_change,
                epsilon_min,
            )
            movement_episode_rewards[i] = movement_reward
            shooting_episode_rewards[i] = shooting_reward

            last_100_move_avg = movement_episode_rewards[max(0, i - 100):i + 1].mean()
            last_100_shoot_avg = shooting_episode_rewards[max(0, i - 100):i + 1].mean()
            (move_loss, shoot_loss) = loss
            print("Episode:", i,
                  "Duration:", duration,
                  "Num steps:", num_steps_in_episode,
                  "Movement Reward:", movement_reward,
                  "Shooting Reward:", shooting_reward,
                  "Training time per step:", "%.3f" % time_per_step,
                  "Avg Movement Reward (Last 100):", "%.3f" % last_100_move_avg,
                  "Avg Shooting Reward (Last 100):", "%.3f" % last_100_shoot_avg,
                  "Epsilon:", "%.3f" % epsilon,
                  "Move Loss:", "%.3f" % move_loss,
                  "Shooting Loss:", "%.3f" % shoot_loss
                  )
            sys.stdout.flush()
            if i > 0 and i % 100 == 0:
                saver.save(sess, './models/current.ckpt')
                saver.save(sess, './models/episode_{}.ckpt'.format(i))

                print("Checkpoint Saved.")


if __name__ == '__main__':
    main()
