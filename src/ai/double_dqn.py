# -*- coding: utf-8 -*-
""" The DQN Model """
import random
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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


class DoubleDQN(nn.Module):
    def __init__(self, number_of_actions=9):
        super(DoubleDQN, self).__init__()
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


class DDQNAgent:
    def __init__(self, name, device=0, n_actions=9, replay_buffer_size=50000, batch_size=32, gamma=0.98):
        self.name = name
        self.device = device
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory = ReplayMemory(replay_buffer_size)
        self.last_state = None
        self.iteration = 0
        self.target_update = 10
        self.gamma = gamma

        self.policy_net = DoubleDQN().to(self.device)
        self.target_net = DoubleDQN().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)

        self.epsilon = 0.05
        self.epsilon_min = 0.05
        self.epsilon_decay = 5

    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Samples of batch-arrays.
        batch = Samples(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = 1 - torch.tensor(batch.done, dtype=torch.uint8)
        non_final_next_states = [
            s for nonfinal, s in zip(non_final_mask, batch.state_prime) if nonfinal > 0
        ]
        non_final_next_states = torch.cat(non_final_next_states).to(self.device)
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        self.optimizer.zero_grad()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # about detach(): https://discuss.pytorch.org/t/detach-no-grad-and-requires-grad/16915/7
        next_state_values = torch.zeros(self.batch_size).to(self.device)
        next_state_values[non_final_mask] = (self.target_net(non_final_next_states).max(1)[0].detach())
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

        # Optimize the model
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def get_action(self, state):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay / (self.epsilon_decay + self.epsilon))
        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state).to(self.device)
                return torch.argmax(q_values).item(), np.max(q_values.cpu().detach().numpy()), self.epsilon
        else:
            return random.randrange(self.n_actions), 0, self.epsilon

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def play(self, image, action, reward, done):
        image = torch.from_numpy(np.expand_dims(image, axis=0)).float()
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)

        if self.last_state is None:
            self.last_state = torch.cat((image, image, image, image)).unsqueeze(0)

        state = torch.cat((self.last_state.squeeze(0)[1:, :, :], image)).unsqueeze(0)

        self.memory.push(self.last_state, action, state, reward, done)
        self.update_network()

        self.last_state = state

        # Update the target network, copying all weights and biases in DQN
        if self.iteration % self.target_update == 0:
            self.update_target_network()

        self.iteration += 1

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
