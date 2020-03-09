import cv2
import ai as DQN
import gym
import time
import numpy as np
# from flappy import GameState


def prepare_flappy(image):
    image = image[0:288, 0:404]
    image = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image[image > 0] = 255
    return image


def prepare_image(image):
    image = cv2.resize(image, (84, 110), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[image > 0] = 255
    return image


def test():
    #game_state = GameState()
    env = gym.make('BreakoutDeterministic-v4')

    # ai = DQN.DQNAgent('test', n_actions=2, reward_delay=0, death_delay=0, classifier_input=2048) # flappy
    ai = DQN.DQNAgent('test', n_actions=env.action_space.n, reward_delay=0,
                      death_delay=0, replay_buffer_size=1000000, classifier_input=3072)

    print_freq = 100
    action = 0
    iteration = 0
    cum_reward = 0
    episode = 0
    all_rewards = []

    env.reset()
    try:
        while True:
            # image, reward, terminal = game_state.frame_step(action) #Flappybird
            image, reward, terminal, _ = env.step(action)  # ai gym
            cum_reward += reward
            action, qmax, epsilon, is_random = ai.train(prepare_image(image), action, reward, terminal)

            # We need to focus more on not flying at first.. so lets do our own epsilon check here
            # if is_random:
            #     sample = random.randrange(10)
            #     if sample == 0:
            #         action = 1
            #     else:
            #         action = 0

            #print(f'{iteration}: Action: {action}  QMax: {qmax}, Reward: {reward}, Terminal: {terminal}')

            if episode > 10000:
                env.render()
                time.sleep(0.05)

            if terminal:
                episode += 1
                all_rewards.append(cum_reward)
                if episode % print_freq == 0:
                    print('Episode #{} | Step #{} | Epsilon {:.2f} | Avg. Reward {:.2f}'.format(
                        episode, iteration, epsilon, np.mean(all_rewards[-print_freq:])))

                cum_reward = 0
                env.reset()

            iteration += 1
    except (KeyboardInterrupt, BrokenPipeError):
        print("Interrupt detected.  Exiting...")


if __name__ == "__main__":
    test()
