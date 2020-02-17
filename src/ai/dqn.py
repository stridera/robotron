# -*- coding: utf-8 -*-
""" The DQN Model """
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, number_of_actions=9):
        super(DQN, self).__init__()
        self.number_of_actions = number_of_actions

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(65664, 3136)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(3136, 512)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc6 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size()[0], -1)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(x)
        return x


class DQNAgent:
    def __init__(self, device=0, n_actions=9, replay_buffer_size=50000, batch_size=32, gamma=0.98):
        self.device = device
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.memory = []
        self.last_state = None
        self.iteration = 0

        self.epsilon = 1.0  # exploration probability at start
        self.epsilon_min = 0.01  # minimum exploration probability
        self.epsilon_decay = 0.000005  # exponential decay rate for exploration prob

        self.model = DQN(n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-6)
        self.criterion = nn.MSELoss()

    def update_network(self):
        if len(self.memory) < self.batch_size:
            return

        # sample random minibatch
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        # unpack minibatch
        last_state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_batch = torch.cat(tuple(d[3] for d in minibatch))

        last_state_batch = last_state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        state_batch = state_batch.to(self.device)

        # get output for the next state
        output_1_batch = self.model(state_batch).to(self.device)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.stack(tuple(reward_batch[i] if minibatch[i][4]
                                    else reward_batch[i] + self.gamma * torch.max(output_1_batch[i])
                                    for i in range(len(minibatch))))

        # extract Q-value
        q_value = self.model(last_state_batch).gather(1, action_batch).squeeze()

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        self.optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = self.criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
            np.exp(-self.epsilon_decay * self.iteration)

        self.iteration += 1

        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.model(state).to(self.device)
                return torch.argmax(q_values).item(), np.max(q_values.cpu().detach().numpy()), epsilon
        else:
            return random.randrange(self.n_actions), 0, epsilon

    def play(self, name, image, action, reward, done):
        image = torch.from_numpy(np.expand_dims(image, axis=0)).float()
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)

        if self.last_state is None:
            self.last_state = torch.cat((image, image, image, image)).unsqueeze(0)

        state = torch.cat((self.last_state.squeeze(0)[1:, :, :], image)).unsqueeze(0)

        self.memory.append((self.last_state, action, reward, state, done))

        if len(self.memory) > self.replay_buffer_size:
            self.memory.pop(0)

        self.update_network()
        action = self.get_action(state)

        self.last_state = state

        if self.iteration % 25000 == 0:
            torch.save(self.model, f"model/{name}_model_" + str(self.iteration) + ".pth")

        self.iteration += 1

        return action

    def loop(self, name, in_queue, out_queue):
        args = in_queue.get()
        while args:
            out_queue.put(self.play(name, *args))
            args = in_queue.get()
