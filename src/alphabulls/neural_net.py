# alphabulls/neural_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from alphabulls.utils import STATE_SIZE, ACTION_SPACE_SIZE

class AlphaBullsNet(nn.Module):
    """
    The neural network for the AlphaZero-inspired agent.
    It takes the game state and outputs a policy (guess probabilities)
    and a value (expected remaining guesses).
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 512)
        self.fc2 = nn.Linear(512, 256)

        # Policy head: outputs probabilities for each possible guess
        self.policy_head = nn.Linear(256, ACTION_SPACE_SIZE)

        # Value head: outputs a single number (expected remaining guesses)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value