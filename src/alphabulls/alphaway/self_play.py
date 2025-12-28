# alphabulls/self_play.py

import torch
import torch.nn.functional as F
import numpy as np
from alphabulls.game_env import BullsAndCowsGame
from alphabulls.utils import state_to_tensor, INDEX_TO_CODE, CODE_TO_INDEX


def play_game(model):
    """
    Plays one full game of Bulls and Cows using the model to decide moves.
    Returns a list of training examples: (state, policy, value).
    """
    model.eval()  # Set the model to evaluation mode
    game = BullsAndCowsGame()
    game_history = []  # Stores (state_tensor, action_idx) for the whole game

    while True:
        # 1. Get current state and predict policy/value
        state_tensor = state_to_tensor(game.get_current_state())

        with torch.no_grad():
            policy_logits, _ = model(state_tensor.unsqueeze(0))

        policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        # 2. Choose an action (exploration)
        # We add some noise to encourage exploration, but here we'll just sample
        action_idx = np.random.choice(len(policy_probs), p=policy_probs)
        guess = INDEX_TO_CODE[action_idx]

        game_history.append((state_tensor, action_idx))

        # 3. Make the move in the environment
        _, is_won = game.make_guess(guess)

        if is_won or game.guess_count >= 15:  # End game if won or too long
            break

    # 4. Process game results to create training data
    training_data = []
    num_guesses = game.guess_count

    for i, (state, action_idx) in enumerate(game_history):
        # The "value" is the number of moves it actually took from that point
        target_value = float(num_guesses - i)

        # The target policy is a one-hot vector of the action taken
        target_policy = np.zeros(len(policy_probs), dtype=np.float32)
        target_policy[action_idx] = 1.0

        training_data.append((state, target_policy, target_value))

    return training_data, num_guesses