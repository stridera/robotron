# -*- coding: utf-8 -*-
r"""
The DQN Model

Research:
- https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial
- https://github.com/ejmejm/Q-Learning-Tutorials/blob/master/DQN_target.ipynb
- https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
"""


import random
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# import cv2
# import torchvision
# from torch.utils.tensorboard import SummaryWriter

# from ai.models import TDQN as model
from ai.models import SCM as model

Samples = namedtuple("Samples", ("state", "action", "state_prime", "reward", "done"))


class ReplayMemory:
    """ Holds the replay buffer for the model """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Samples(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, name, device=0, frame_buffer=4, n_actions=64, replay_buffer_size=50000, death_delay=9,
                 image_size=(), reward_delay=3, batch_size=32, classifier_input=12288, gamma=0.98):
        """The DQN Agent

        Arguments:
            name {string} -- [description]

        Keyword Arguments:
            device {int} -- The CUDA device (default: {0})
            frame_buffer {int} -- Number of frames to pass to model (stacked images) (default: {4})
            n_actions {int} -- Number of actions to return (defaults to 9: 8 cardinal, and none) (default: {9})
            replay_buffer_size {int} -- DQN Experienced Replay Buffer (default: {50000})
            death_delay {int} -- Episode Buffer to keep.  Train on info n frames old.  Used to handle death delay. (default: {7})
            reward_delay {int} -- Number of frames to delay the reward.  Used to handle the noise delay. (default: {3})
            batch_size {int} -- Number of items to include in a training batch (default: {32})
            gamma {float} -- The reward discount factor (default: {0.98})
        """
        self.name = name
        self.device = device
        self.frame_buffer = frame_buffer
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma

        self.memory = ReplayMemory(replay_buffer_size)
        self.replay_buffer_size = replay_buffer_size
        self.state_buffer = []
        self.reward_buffer = []
        self.action_buffer = []
        self.last_state = None

        self.death_delay = death_delay
        self.reward_delay = reward_delay

        self.iteration = 0

        self.epsilon = lambda step: np.clip(1 - 0.9 * (step/100000), 0.1, 1)

        self.policy_net = model(frame_buffer, classifier_input, n_actions).to(device)
        self.target_net = model(frame_buffer, classifier_input, n_actions).to(device)
        # self.policy_net = torch.load("models/combined_model_100.pth")
        # self.policy_net.eval()

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.model_update_freq = 100

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.criterion = nn.MSELoss()

        self.writer_name = 'runs/robotron'
        # self.writer = SummaryWriter(self.writer_name)

    def update_network(self):
        """
        Update the network.
        Taken from the pytorch tutorial.
        """
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Samples(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.state_prime)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.state_prime if s is not None]).to(self.device)
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # self.writer.add_scalar('Train/Loss', loss.item(), self.iteration)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        if self.iteration % self.model_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_action(self, state):
        """Get an action with epsilon consideration.

        Arguments:
            state {ndarray} -- Image Stack

        Returns:
            long -- Action with highest Q value. (Or random if under epsilon)
        """

        sample = random.random()
        epsilon = self.epsilon(self.iteration)
        if sample > epsilon:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state).to(self.device)
                # self.writer.add_scalar('Play/Q-Value', torch.argmax(q_values).item(), self.iteration)
                return torch.argmax(q_values).item(), np.max(q_values.cpu().detach().numpy()), epsilon, False
        else:
            return random.randrange(self.n_actions), 0, epsilon, True

    def update_memory(self, state, action, reward, dead):
        """Update the self. after considering delays

        Arguments:
            state {ndarray} -- Image Stack
            action {long} -- The action taken
            reward {float} -- Score clipped between -1 and 1
            dead {bool} -- Did the character die?
        """
        self.state_buffer.append((self.last_state, state))
        self.action_buffer.append(action)

        if len(self.state_buffer) >= self.reward_delay:
            # We'll skip appending rewards until we reach the reward delay
            self.reward_buffer.append(reward)

        if len(self.state_buffer) > self.death_delay:
            # We'll only enact on data once we're past the death delay
            last_state, state = self.state_buffer.pop(0)
            action = self.action_buffer.pop(0)

            if dead:
                # If we're dead, add this data as terminal and clear everything to start again
                self.state_buffer = []
                self.action_buffer = []
                self.reward_buffer = []
            else:
                # Only update the reward if we're not dead, otherwise use the delayed reward
                reward = self.reward_buffer.pop(0)

            self.memory.push(last_state, action, state, reward, dead)

            # Debugging!  Show image with states.
            # img = np.hstack(state.squeeze().cpu().detach().numpy())
            # thresh = np.zeros_like(img)
            # thresh[img > 0] = 255
            # resized = cv2.resize(thresh, (492*4, 665), interpolation=cv2.INTER_LINEAR)
            # movedesc = ["U", "UR", "R", "DR", "D", "DL", "L", "UL"]
            # line = f"Action: {movedesc[action.cpu().detach()//8]}, {movedesc[action.cpu().detach()%8]}  Reward: {reward.cpu().detach()}  Dead: {dead}"
            # cv2.putText(resized, line, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.imwrite(f'/home/strider/Code/robotron/tmp/{self.iteration}.png', resized)
            # img_grid = torchvision.utils.make_grid(state.squeeze())
            # self.writer.add_image('state', img_grid, self.iteration)
            # self.writer.add_scalar('Reward', reward, self.iteration)
            # self.writer.add_scalar('Action', action, self.iteration)

    def train(self, image, action, reward, dead):
        """Play a round.
        Note:
            There are two delays to consider:
            - Reward Delay: In an effort to handle noise (screen flashes and numbers vanish for a frame) it takes 3 frames
                for the scores to update.  Variable for this is score_delay.
            - Death Delay: We don't detect a death until after the character has reloaded and the life marker vanishes.
                Takes around 7 frames.  Variable for this buffer size.

            To handle this, we're going to keep an array of the last (self.buffer_size) rounds.  If we

        Arguments:
            image {ndarray} -- The image data
            action {int} -- The last action taken
            reward {float} -- Reward from -1 to 1
            dead {bool} -- terminal state (character has died)

        Returns:
            int -- new action
        """

        image = torch.from_numpy(np.expand_dims(image, axis=0)).float()
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)

        if self.last_state is None:
            self.last_state = torch.cat(([image]*self.frame_buffer)).unsqueeze(0)

        state = torch.cat((self.last_state.squeeze(0)[1:, :, :], image)).unsqueeze(0)

        self.update_memory(state, action, reward, dead)
        self.update_network()
        action = self.get_action(state)

        self.last_state = state

        self.iteration += 1

        if self.iteration % 1000 == 0 and len(self.memory) != self.replay_buffer_size:
            print('Iteration: ', self.iteration, 'Memory Size:', len(self.memory))

        if self.iteration % 25000 == 0:
            print("Saving model")
            torch.save(self.target_net, f"models/{self.name}_model_" + str(self.iteration) + ".pth")

        # self.writer.close()
        return action

    def play(self, image):
        image = torch.from_numpy(np.expand_dims(image, axis=0)).float()

        if self.last_state is None:
            self.last_state = torch.cat(([image]*self.frame_buffer)).unsqueeze(0)

        state = torch.cat((self.last_state.squeeze(0)[1:, :, :], image)).unsqueeze(0)

        return self.get_action(state)
