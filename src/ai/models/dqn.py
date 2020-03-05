import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, frames=4, number_of_actions=9):
        super(DQN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(frames, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU()
        )

        self.classif = nn.Sequential(
            nn.Linear(7904, 256),
            nn.ReLU(),
            nn.Linear(256, number_of_actions),
        )

    def forward(self, q):
        q = self.features(q)
        q = q.view(-1, 7904)
        q = self.classif(q)
        return q
