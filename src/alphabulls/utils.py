# alphabulls/utils.py

import itertools
import numpy as np
import torch

# --- Game Constants ---
NUM_DIGITS = 4
MAX_HISTORY = 10  # Max history length for the neural network state
MAX_GUESSES = 15  # Max history length for the neural network state

# --- Action Space ---
# Generate all possible 4-digit codes with unique digits
# This is our action space: every possible guess
ALL_POSSIBLE_CODES = [''.join(p) for p in itertools.permutations('0123456789', NUM_DIGITS)]
ACTION_SPACE_SIZE = len(ALL_POSSIBLE_CODES)

# Create a mapping from code string to an integer index and back
CODE_TO_INDEX = {code: i for i, code in enumerate(ALL_POSSIBLE_CODES)}
INDEX_TO_CODE = {i: code for i, code in enumerate(ALL_POSSIBLE_CODES)}

# --- Neural Network Constants ---
# The state will be a flattened vector of past guesses and their feedback
# For each guess: 4 digits + 1 for bulls + 1 for cows = 6 features
STATE_FEATURE_SIZE = NUM_DIGITS + 2
# NOTE: STATE_SIZE constant is no longer needed by the network but doesn't hurt to keep
STATE_SIZE = STATE_FEATURE_SIZE * MAX_HISTORY

# --- Training Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def code_to_tensor(code_str):
    """Converts a 4-digit string code to a list of ints."""
    return [int(d) for d in code_str]


def state_to_tensor(history):
    """
    Converts game history into a sequential tensor for the LSTM/GRU.
    Output shape: (MAX_GUESSES, STATE_FEATURE_SIZE) -> (10, 6)
    """
    # Create a matrix of zeros. Shape will be (10, 6)
    state_matrix = np.zeros((MAX_HISTORY, STATE_FEATURE_SIZE))

    if history:
        for i, (guess, feedback) in enumerate(history[-MAX_HISTORY:]):
            # Each row is one guess: [d1, d2, d3, d4, bulls, cows]
            row = np.array(code_to_tensor(guess) + [feedback[0], feedback[1]])
            state_matrix[i] = row

    # Return a tensor of shape (10, 6)
    return torch.tensor(state_matrix, dtype=torch.float32).to(DEVICE)
