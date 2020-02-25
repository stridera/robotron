# -*- coding: utf-8 -*-
""" The DQN Model """
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class SpatialCrossMapLRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNELS=True):
        super(SpatialCrossMapLRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(
                kernel_size=(local_size, 1, 1),
                stride=1,
                padding=(int((local_size - 1.0) / 2), 0, 0),
            )
        else:
            self.average = nn.AvgPool2d(
                kernel_size=local_size, stride=1, padding=int((local_size - 1.0) / 2)
            )
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x


class DQN(nn.Module):
    def __init__(self, number_of_actions=9):
        super(DQN, self).__init__()
        self.number_of_actions = number_of_actions
        self.features = nn.Sequential(
            nn.Conv2d(4, 96, (7, 7), (2, 2)),
            nn.ReLU(),
            SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(96, 256, (5, 5), (2, 2), (1, 1)),
            nn.ReLU(),
            SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
        )
        self.classif = nn.Sequential(
            nn.Linear(35840, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, number_of_actions),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classif(x)
        return x


class DQNAgent:
    def __init__(self, name, device=0, n_actions=9, replay_buffer_size=50000, batch_size=32, gamma=0.98):
        self.name = name
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

        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.model(state).to(self.device)
                return torch.argmax(q_values).item(), np.max(q_values.cpu().detach().numpy()), epsilon
        else:
            return random.randrange(self.n_actions), 0, epsilon

    def play(self, image, action, reward, done):
        self.iteration += 1

        image = torch.from_numpy(np.expand_dims(image, axis=0)).float()
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)

        if self.last_state is None:
            self.last_state = torch.cat((image, image, image, image)).unsqueeze(0)

        state = torch.cat((self.last_state.squeeze(0)[1:, :, :], image)).unsqueeze(0)

        self.memory.append((self.last_state, action, reward, state, done))
        # print(self.name, len(self.memory))

        if len(self.memory) > self.replay_buffer_size:
            self.memory.pop(0)

        self.update_network()
        action = self.get_action(state)

        self.last_state = state

        if self.iteration % 25000 == 0:
            print("Saving model")
            torch.save(self.model, f"model/{self.name}_model_" + str(self.iteration) + ".pth")

        return action

    def loop(self, in_queue, out_queue):
        args = in_queue.get()
        while args:
            out_queue.put(self.play(*args))
            args = in_queue.get()
