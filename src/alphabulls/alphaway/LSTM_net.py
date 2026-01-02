# alphabulls/neural_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from alphabulls.utils import STATE_FEATURE_SIZE, ACTION_SPACE_SIZE


class AlphaBullsNet(nn.Module):
    """
    An LSTM-based network that processes the game history as a sequence.
    """

    def __init__(self, lstm_hidden_size=128, num_lstm_layers=2):
        super().__init__()

        # The LSTM layer will process the sequence of guesses.
        # It takes input of size 6 (4 digits + 2 feedback) for each step.
        self.lstm = nn.LSTM(
            input_size=STATE_FEATURE_SIZE,  # 6
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True  # Crucial for correct shape handling
        )

        # A linear layer to process the final output of the LSTM
        self.fc_shared = nn.Linear(lstm_hidden_size, 128)

        # Policy head remains the same
        self.policy_head = nn.Linear(128, ACTION_SPACE_SIZE)

        # Value head remains the same
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        # x is now expected to have the shape: [batch_size, sequence_length, feature_size]
        # For a single item, this is [1, 10, 6]

        # Pass the sequence through the LSTM
        # We only care about the output of the last time step, not the final hidden/cell states
        lstm_out, _ = self.lstm(x)

        # We take the output from the VERY LAST time step of the sequence
        last_time_step_out = lstm_out[:, -1, :]

        # Pass this condensed "knowledge vector" through the shared layer
        x = F.relu(self.fc_shared(last_time_step_out))

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value