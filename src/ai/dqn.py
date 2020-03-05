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
import torch
import torch.nn as nn
import torch.optim as optim

from ai.models import DQN as model


class DQNAgent:
    def __init__(self, name, device=0, frame_buffer=4, n_actions=9, replay_buffer_size=50000, death_delay=9,
                 reward_delay=3, batch_size=32, gamma=0.98):
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
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.memory = []
        self.death_delay = death_delay
        self.reward_delay = reward_delay

        self.last_state = None

        self.state_buffer = []
        self.reward_buffer = []
        self.action_buffer = []

        self.iteration = 0

        self.epsilon = 1.0  # exploration probability at start
        self.epsilon_min = 0.01  # minimum exploration probability
        self.epsilon_decay = 0.00005  # exponential decay rate for exploration prob

        self.model = model(frame_buffer, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-6)
        self.criterion = nn.MSELoss()

    def update_network(self):
        if len(self.memory) < self.batch_size:
            return

        # sample random minibatch
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        # unpack minibatch
        last_state_batch = torch.cat(tuple(d[0] for d in minibatch)).to(self.device)
        action_batch = torch.cat(tuple(d[1] for d in minibatch)).to(self.device)
        reward_batch = torch.cat(tuple(d[2] for d in minibatch)).to(self.device)
        state_batch = torch.cat(tuple(d[3] for d in minibatch)).to(self.device)

        # get output for the state
        output_state_batch = self.model(state_batch).to(self.device)

        # set y_batch to reward for terminal state, otherwise to reward + gamma*max(Q)
        y_batch = torch.stack(tuple(reward_batch[i] if minibatch[i][4]
                                    else reward_batch[i] + self.gamma * torch.max(output_state_batch[i])
                                    for i in range(len(minibatch))))

        # extract Q-value
        q_value = self.model(last_state_batch).gather(1, action_batch).squeeze()

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        self.optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # print("qv", q_value, y_batch.shape)
        # calculate loss
        loss = self.criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
            np.exp(-self.epsilon_decay * self.iteration)

        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.model(state).to(self.device)
                return torch.argmax(q_values).item(), np.max(q_values.cpu().detach().numpy()), epsilon
        else:
            return random.randrange(self.n_actions), 0, epsilon

    def update_memory(self, state, action, reward, dead):
        self.state_buffer.append(state)
        self.action_buffer.append(action)

        if len(self.state_buffer) >= self.reward_delay:
            # We'll skip appending rwards until we reach the reward delay
            self.reward_buffer.append(reward)

        if len(self.state_buffer) > self.death_delay:
            # We'll only enact on data once we're past the death delay
            state = self.state_buffer.pop(0)
            if dead:
                # If we're dead, add this data as terminal and clear everything to start again
                self.state_buffer = []
                self.action_buffer = []
                self.reward_buffer = []
            else:
                action = self.action_buffer.pop(0)
                reward = self.reward_buffer.pop(0)

            episode = (self.last_state, action, reward, state, dead)
            self.memory.append(episode)

        if len(self.memory) > self.replay_buffer_size:
            self.memory.pop(0)

    def play(self, image, action, reward, dead):
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
        self.iteration += 1

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

        if self.iteration % 25000 == 0:
            print("Saving model")
            torch.save(self.model, f"models/{self.name}_model_" + str(self.iteration) + ".pth")

        return action

    def loop(self, in_queue, out_queue):
        args = in_queue.get()
        while args:
            out_queue.put(self.play(*args))
            args = in_queue.get()
